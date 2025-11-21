"""
Q1大豆贸易模型改进实现
基于综合分析报告的问题修复方案
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ImprovedConfig:
    """改进的模型配置"""
    window_size: int = 6  # 减少到6个月
    forecast_horizon: int = 6  # 减少到6个月
    batch_size: int = 16  # 减少批次大小
    epochs: int = 50  # 减少训练轮次
    learning_rate: float = 0.001  # 降低学习率
    early_stopping_patience: int = 10
    validation_split: float = 0.3  # 增加验证集比例
    lstm_units: int = 64  # LSTM单元数
    dropout_rate: float = 0.2  # Dropout率
    
    # 业务约束参数
    min_price: float = 200.0  # 最低价格（美元/吨）
    max_price: float = 2000.0  # 最高价格（美元/吨）
    min_quantity: float = 1000.0  # 最低进口量（吨）
    max_quantity_growth: float = 2.0  # 最大月度增长率（200%）
    min_quantity_growth: float = -0.5  # 最小月度增长率（-50%）


class DataQualityValidator:
    """数据质量验证器"""
    
    def __init__(self):
        self.validation_rules = {
            'price_range': (200, 2000),  # 合理价格范围
            'quantity_range': (1000, 1e8),  # 合理数量范围
            'tariff_range': (0, 1),  # 关税范围（0-100%）
        }
    
    def validate_monthly_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证月度数据质量"""
        logger.info("验证月度数据质量...")
        
        # 检查必需字段
        required_cols = ['period', 'partnerDesc', 'netWgt', 'primaryValue']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需字段: {missing_cols}")
        
        # 数据类型验证
        df['period'] = df['period'].astype(str)
        df['netWgt'] = pd.to_numeric(df['netWgt'], errors='coerce')
        df['primaryValue'] = pd.to_numeric(df['primaryValue'], errors='coerce')
        
        # 异常值检测和处理
        df = self._handle_extreme_values(df)
        
        # 单位一致性检查
        df = self._validate_unit_consistency(df)
        
        logger.info(f"数据验证完成，剩余记录数: {len(df)}")
        return df
    
    def validate_annual_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证年度数据质量"""
        logger.info("验证年度数据质量...")
        
        # 检查必需字段
        required_cols = ['year', 'exporter', 'import_quantity_tonnes', 'import_value_usd', 'tariff_cn_on_exporter']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需字段: {missing_cols}")
        
        # 计算单位价格并验证合理性
        df['calculated_price'] = df['import_value_usd'] / df['import_quantity_tonnes']
        
        # 价格合理性检查
        invalid_price = (df['calculated_price'] < self.validation_rules['price_range'][0]) | \
                       (df['calculated_price'] > self.validation_rules['price_range'][1])
        
        if invalid_price.any():
            logger.warning(f"发现 {invalid_price.sum()} 个异常价格记录")
            logger.warning(f"异常价格范围: {df.loc[invalid_price, 'calculated_price'].min():.2f} - {df.loc[invalid_price, 'calculated_price'].max():.2f}")
        
        # 关税合理性检查
        invalid_tariff = (df['tariff_cn_on_exporter'] < 0) | (df['tariff_cn_on_exporter'] > 1)
        if invalid_tariff.any():
            logger.warning(f"发现 {invalid_tariff.sum()} 个异常关税记录")
        
        logger.info(f"年度数据验证完成，剩余记录数: {len(df)}")
        return df
    
    def _handle_extreme_values(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """处理极端值"""
        numeric_cols = ['netWgt', 'primaryValue']
        
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                
                # 记录处理前的统计信息
                original_count = len(df)
                extreme_count = ((df[col] < lower) | (df[col] > upper)).sum()
                
                # 裁剪极端值
                df[col] = df[col].clip(lower, upper)
                
                if extreme_count > 0:
                    logger.info(f"字段 {col}: 处理 {extreme_count} 个极端值 ({extreme_count/original_count*100:.1f}%)")
        
        return df
    
    def _validate_unit_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证单位一致性"""
        # 检查数量和金额的比例关系是否合理
        if 'netWgt' in df.columns and 'primaryValue' in df.columns:
            implied_price = df['primaryValue'] / (df['netWgt'] + 1e-6)
            
            # 识别可能的单位错误
            suspicious_records = (implied_price > 10000) | (implied_price < 10)
            if suspicious_records.any():
                logger.warning(f"发现 {suspicious_records.sum()} 个可能的单位不一致记录")
        
        return df


class ImprovedFeatureEngineer:
    """改进的特征工程"""
    
    def __init__(self, config: ImprovedConfig):
        self.config = config
        self.feature_columns = []
        self.scalers = {}
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建改进的特征"""
        logger.info("创建改进的特征...")
        
        features = df.copy()
        
        # 时间特征（更稳健的实现）
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        # 趋势特征（标准化处理）
        year_min, year_max = features['year'].min(), features['year'].max()
        features['year_trend'] = (features['year'] - year_min) / max(1, (year_max - year_min))
        
        # 滞后特征（减少滞后阶数）
        for lag in [1, 3, 6]:  # 移除12个月滞后
            features[f'quantity_lag_{lag}'] = features.groupby('exporter')['import_quantity'].shift(lag)
            features[f'price_lag_{lag}'] = features.groupby('exporter')['unit_price'].shift(lag)
        
        # 移动平均特征（使用更小的窗口）
        for window in [3, 6]:  # 移除12个月窗口
            features[f'quantity_ma_{window}'] = features.groupby('exporter')['import_quantity'].transform(
                lambda s: s.rolling(window, min_periods=1).mean()  # 要求最少1个数据点
            )
            features[f'price_ma_{window}'] = features.groupby('exporter')['unit_price'].transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            )
        
        # 改进的价格弹性计算
        features = self._improved_price_elasticity(features)
        
        # 添加业务逻辑特征
        features['tariff_impact'] = features['tariff_rate'] * features['unit_price']
        features['effective_price'] = features['unit_price'] * (1 + features['tariff_rate'])
        
        # 改进的增长率计算
        features = self._improved_growth_rate(features)
        
        # 处理缺失值和无穷值
        features = self._handle_missing_values(features)
        
        # 定义特征列
        self.feature_columns = [
            'import_quantity', 'unit_price', 'tariff_rate', 'market_share',
            'month_sin', 'month_cos', 'year_trend',
            'quantity_lag_1', 'quantity_lag_3', 'quantity_lag_6',
            'price_lag_1', 'price_lag_3', 'price_lag_6',
            'quantity_ma_3', 'quantity_ma_6',
            'price_ma_3', 'price_ma_6',
            'price_elasticity', 'tariff_impact', 'effective_price', 'volume_growth'
        ]
        
        logger.info(f"特征工程完成，共创建 {len(self.feature_columns)} 个特征")
        return features
    
    def _improved_price_elasticity(self, df: pd.DataFrame) -> pd.DataFrame:
        """改进的价格弹性计算"""
        # 使用更稳健的弹性计算方法
        quantity_pct = df.groupby('exporter')['import_quantity'].pct_change()
        price_pct = df.groupby('exporter')['unit_price'].pct_change()
        
        # 避免除零和极端值
        valid_mask = (abs(price_pct) > 0.001) & (abs(quantity_pct) < 10) & (abs(price_pct) < 10)
        
        elasticity = np.where(valid_mask, quantity_pct / (price_pct + 1e-6), 0)
        
        # 限制弹性范围
        elasticity = np.clip(elasticity, -10, 10)
        
        df['price_elasticity'] = elasticity
        return df
    
    def _improved_growth_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """改进的增长率计算"""
        # 使用对数差分代替简单百分比变化
        log_quantity = np.log(df['import_quantity'] + 1)
        df['volume_growth'] = df.groupby('exporter')['log_quantity'].diff()
        
        # 限制增长率范围
        df['volume_growth'] = df['volume_growth'].clip(-2, 2)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 替换无穷值
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 按出口商分组进行前向填充
        grouped = df.groupby('exporter', group_keys=False)
        fill_cols = [col for col in df.columns if col not in ['exporter', 'date']]
        df[fill_cols] = grouped[fill_cols].ffill()
        
        # 剩余缺失值用0填充
        df[fill_cols] = df[fill_cols].fillna(0)
        
        return df


class ConstrainedLSTMModel(tf.keras.Model):
    """带业务约束的LSTM模型"""
    
    def __init__(self, config: ImprovedConfig, n_features: int, n_targets: int):
        super().__init__()
        self.config = config
        
        # LSTM层
        self.lstm1 = tf.keras.layers.LSTM(config.lstm_units, return_sequences=True)
        self.dropout1 = tf.keras.layers.Dropout(config.dropout_rate)
        self.lstm2 = tf.keras.layers.LSTM(config.lstm_units)
        self.dropout2 = tf.keras.layers.Dropout(config.dropout_rate)
        
        # 输出层（带约束）
        self.price_output = tf.keras.layers.Dense(
            1, 
            activation='relu',  # 确保价格非负
            name='price_output'
        )
        self.quantity_output = tf.keras.layers.Dense(
            1, 
            activation='relu',  # 确保数量非负
            name='quantity_output'
        )
        self.market_share_output = tf.keras.layers.Dense(
            1, 
            activation='sigmoid',  # 确保市场份额在0-1之间
            name='market_share_output'
        )
    
    def call(self, inputs, training=None):
        x = self.lstm1(inputs)
        if training:
            x = self.dropout1(x)
        x = self.lstm2(x)
        if training:
            x = self.dropout2(x)
        
        # 生成预测
        price = self.price_output(x)
        quantity = self.quantity_output(x)
        market_share = self.market_share_output(x)
        
        # 拼接输出
        output = tf.concat([price, quantity, market_share], axis=-1)
        return output
    
    def get_config(self):
        return {"config": self.config}


class ImprovedLSTMPipeline:
    """改进的LSTM预测管道"""
    
    def __init__(self, config: ImprovedConfig):
        self.config = config
        self.model = None
        self.scalers = {}
        self.feature_engineer = ImprovedFeatureEngineer(config)
        self.validator = DataQualityValidator()
    
    def prepare_data(self, monthly_file: str, annual_file: str) -> pd.DataFrame:
        """准备训练数据"""
        logger.info("准备训练数据...")
        
        # 加载和验证数据
        monthly_df = pd.read_csv(monthly_file)
        annual_df = pd.read_csv(annual_file)
        
        monthly_df = self.validator.validate_monthly_data(monthly_df)
        annual_df = self.validator.validate_annual_data(annual_df)
        
        # 数据合并（改进的合并逻辑）
        merged_df = self._improved_data_merge(monthly_df, annual_df)
        
        # 特征工程
        feature_df = self.feature_engineer.create_features(merged_df)
        
        return feature_df
    
    def _improved_data_merge(self, monthly_df: pd.DataFrame, annual_df: pd.DataFrame) -> pd.DataFrame:
        """改进的数据合并"""
        # 确保时间范围一致
        common_years = set(monthly_df['year'].unique()) & set(annual_df['year'].unique())
        logger.info(f"共同年份: {sorted(common_years)}")
        
        # 过滤到共同年份
        monthly_filtered = monthly_df[monthly_df['year'].isin(common_years)]
        annual_filtered = annual_df[annual_df['year'].isin(common_years)]
        
        # 创建关税查找表
        tariff_lookup = annual_filtered[['year', 'exporter', 'tariff_cn_on_exporter']].drop_duplicates()
        
        # 合并数据
        merged = monthly_filtered.merge(
            tariff_lookup, 
            on=['year', 'exporter'], 
            how='left',
            validate='many_to_one'  # 添加验证确保合并正确
        )
        
        return merged
    
    def build_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """构建序列数据"""
        logger.info("构建序列数据...")
        
        # 特征缩放
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        features_scaled = feature_scaler.fit_transform(df[self.feature_engineer.feature_columns])
        targets_scaled = target_scaler.fit_transform(df[['import_quantity', 'unit_price', 'market_share']])
        
        self.scalers['features'] = feature_scaler
        self.scalers['targets'] = target_scaler
        
        # 构建序列
        X, y = [], []
        
        for exporter in df['exporter'].unique():
            exporter_mask = df['exporter'] == exporter
            exporter_features = features_scaled[exporter_mask]
            exporter_targets = targets_scaled[exporter_mask]
            
            # 为每个出口商构建序列
            for i in range(self.config.window_size, len(exporter_features)):
                X.append(exporter_features[i-self.config.window_size:i])
                y.append(exporter_targets[i:i+self.config.forecast_horizon])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"序列数据构建完成: X shape {X.shape}, y shape {y.shape}")
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """训练模型"""
        logger.info("训练模型...")
        
        # 数据分割（时间序列交叉验证）
        tscv = TimeSeriesSplit(n_splits=3)
        
        # 构建模型
        n_features = X.shape[2]
        n_targets = y.shape[2] if len(y.shape) > 2 else 1
        
        self.model = ConstrainedLSTMModel(self.config, n_features, n_targets)
        
        # 编译模型
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # 训练模型（使用早停）
        early_stopping = tf.keras.callbacks.EarlyStopping(
            patience=self.config.early_stopping_patience,
            restore_best_weights=True
        )
        
        # 使用第一个分割作为验证集
        train_idx, val_idx = next(tscv.split(X))
        
        history = self.model.fit(
            X[train_idx], y[train_idx],
            validation_data=(X[val_idx], y[val_idx]),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # 评估模型
        val_loss = min(history.history['val_loss'])
        logger.info(f"模型训练完成，验证损失: {val_loss:.4f}")
        
        return {'val_loss': val_loss, 'history': history.history}
    
    def generate_forecasts(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成预测"""
        logger.info("生成预测...")
        
        predictions = []
        
        for exporter in df['exporter'].unique():
            exporter_mask = df['exporter'] == exporter
            exporter_data = df[exporter_mask]
            
            if len(exporter_data) < self.config.window_size:
                logger.warning(f"出口商 {exporter} 数据不足，跳过预测")
                continue
            
            # 获取最近的历史数据
            recent_data = exporter_data.tail(self.config.window_size)
            
            # 特征缩放
            features_scaled = self.scalers['features'].transform(
                recent_data[self.feature_engineer.feature_columns]
            )
            
            # 重塑输入形状
            X_input = features_scaled.reshape(1, self.config.window_size, -1)
            
            # 生成预测
            pred_scaled = self.model.predict(X_input, verbose=0)
            
            # 反向缩放
            pred_actual = self.scalers['targets'].inverse_transform(pred_scaled[0])
            
            # 应用业务约束
            pred_constrained = self._apply_business_constraints(pred_actual, exporter_data)
            
            # 创建预测DataFrame
            last_date = recent_data['date'].max()
            future_dates = pd.date_range(
                last_date + pd.offsets.MonthBegin(1), 
                periods=self.config.forecast_horizon, 
                freq='MS'
            )
            
            pred_df = pd.DataFrame(pred_constrained, columns=['import_quantity', 'unit_price', 'market_share'])
            pred_df['date'] = future_dates
            pred_df['exporter'] = exporter
            
            predictions.append(pred_df)
        
        # 合并所有预测
        all_predictions = pd.concat(predictions, ignore_index=True)
        
        # 重新计算市场份额（确保总和为1）
        all_predictions = self._normalize_market_share(all_predictions)
        
        return all_predictions
    
    def _apply_business_constraints(self, predictions: np.ndarray, historical_data: pd.DataFrame) -> np.ndarray:
        """应用业务约束"""
        constrained = predictions.copy()
        
        # 价格约束
        constrained[:, 1] = np.clip(constrained[:, 1], self.config.min_price, self.config.max_price)
        
        # 数量约束
        constrained[:, 0] = np.maximum(constrained[:, 0], self.config.min_quantity)
        
        # 增长率约束（基于历史数据）
        if len(historical_data) > 0:
            last_quantity = historical_data['import_quantity'].iloc[-1]
            
            for i in range(len(constrained)):
                if i == 0:
                    prev_quantity = last_quantity
                else:
                    prev_quantity = constrained[i-1, 0]
                
                # 计算增长率
                growth = (constrained[i, 0] - prev_quantity) / (prev_quantity + 1e-6)
                
                # 限制增长率
                growth = np.clip(growth, self.config.min_quantity_growth, self.config.max_quantity_growth)
                
                # 应用约束后的数量
                constrained[i, 0] = prev_quantity * (1 + growth)
        
        # 市场份额约束
        constrained[:, 2] = np.clip(constrained[:, 2], 0.0, 1.0)
        
        return constrained
    
    def _normalize_market_share(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """归一化市场份额"""
        def normalize_group(group):
            total_share = group['market_share'].sum()
            if total_share > 0:
                group['market_share'] = group['market_share'] / total_share
            return group
        
        predictions = predictions.groupby('date').apply(normalize_group)
        return predictions.reset_index(drop=True)
    
    def evaluate_model(self, predictions: pd.DataFrame, actual_data: pd.DataFrame) -> Dict[str, float]:
        """评估模型性能"""
        logger.info("评估模型性能...")
        
        # 合并预测和实际数据
        merged = predictions.merge(
            actual_data, 
            on=['date', 'exporter'], 
            suffixes=('_pred', '_actual'),
            how='inner'
        )
        
        if len(merged) == 0:
            logger.warning("无法找到匹配的预测和实际数据进行评估")
            return {}
        
        metrics = {}
        target_cols = ['import_quantity', 'unit_price', 'market_share']
        
        for col in target_cols:
            pred_col = f'{col}_pred'
            actual_col = f'{col}_actual'
            
            if pred_col in merged.columns and actual_col in merged.columns:
                # 计算各种指标
                mae = np.mean(np.abs(merged[pred_col] - merged[actual_col]))
                rmse = np.sqrt(np.mean((merged[pred_col] - merged[actual_col]) ** 2))
                
                # 避免除零的MAPE计算
                mape = np.mean(np.abs((merged[pred_col] - merged[actual_col]) / (merged[actual_col] + 1e-6))) * 100
                
                # sMAPE（对称MAPE）
                smape_denom = (np.abs(merged[pred_col]) + np.abs(merged[actual_col])) + 1e-6
                smape = np.mean(2.0 * np.abs(merged[pred_col] - merged[actual_col]) / smape_denom) * 100
                
                metrics[col] = {
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'smape': smape
                }
                
                logger.info(f"{col}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, sMAPE={smape:.2f}%")
        
        return metrics


def run_improved_q1_analysis(monthly_file: str, annual_file: str, output_dir: str) -> Dict:
    """运行改进的Q1分析"""
    logger.info("开始改进的Q1分析...")
    
    # 配置
    config = ImprovedConfig()
    
    # 创建管道
    pipeline = ImprovedLSTMPipeline(config)
    
    # 准备数据
    feature_df = pipeline.prepare_data(monthly_file, annual_file)
    
    # 构建序列
    X, y = pipeline.build_sequences(feature_df)
    
    # 训练模型
    train_results = pipeline.train_model(X, y)
    
    # 生成预测
    predictions = pipeline.generate_forecasts(feature_df)
    
    # 评估模型（如果有实际数据）
    metrics = pipeline.evaluate_model(predictions, feature_df)
    
    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存预测结果
    predictions.to_csv(output_path / 'improved_predictions.csv', index=False)
    
    # 保存评估指标
    if metrics:
        import json
        with open(output_path / 'improved_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    
    logger.info("改进的Q1分析完成")
    
    return {
        'predictions': predictions,
        'metrics': metrics,
        'train_results': train_results
    }


if __name__ == "__main__":
    # 示例用法
    monthly_file = "data/processed/q1/q1_1.csv"
    annual_file = "data/processed/q1/q1_0.csv"
    output_dir = "results/q1/improved"
    
    results = run_improved_q1_analysis(monthly_file, annual_file, output_dir)
    
    # 打印结果摘要
    if results['metrics']:
        print("\n=== 模型评估结果 ===")
        for target, metrics in results['metrics'].items():
            print(f"{target}: MAPE={metrics['mape']:.2f}%, sMAPE={metrics['smape']:.2f}%")
    
    print(f"\n预测结果已保存到: {output_dir}")