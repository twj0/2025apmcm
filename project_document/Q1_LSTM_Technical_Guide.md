# Q1: LSTM时间序列预测技术实现指南

## 1. 问题背景与数据分析

### 1.1 数据状况
- **数据来源**: 中国大豆进口数据（2015-2024）
- **关键变量**:
  - 进口量：年度数据显示美国进口量从2750万吨(2015)降至2200万吨(2024)
  - 关税率：美国关税从3%(2015)升至27.5%(2018-2022)，2024降至13%
  - 价格：单价波动范围335-609美元/吨
  - 月度细粒度数据：`q1_soybean_imports_comtrade_monthly.csv`包含394个月度观察值

### 1.2 建模目标
- 预测未来12-24个月的大豆进口量、价格和市场份额
- 捕捉关税政策变化对贸易流的非线性影响
- 考虑季节性和长期趋势

## 2. LSTM模型架构设计

### 2.1 数据预处理Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

class SoybeanDataProcessor:
    def __init__(self, window_size=12, forecast_horizon=6):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.scalers = {}
        
    def prepare_features(self, df):
        """构建特征工程"""
        # 基础特征
        features = df[['import_quantity', 'unit_price', 'tariff_rate']].copy()
        
        # 时间特征
        features['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        features['year_trend'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
        
        # 滞后特征
        for lag in [1, 3, 6, 12]:
            features[f'quantity_lag_{lag}'] = features['import_quantity'].shift(lag)
            features[f'price_lag_{lag}'] = features['unit_price'].shift(lag)
        
        # 移动平均
        for window in [3, 6, 12]:
            features[f'quantity_ma_{window}'] = features['import_quantity'].rolling(window).mean()
            features[f'price_ma_{window}'] = features['unit_price'].rolling(window).mean()
        
        # 价格弹性指标
        features['price_elasticity'] = features['import_quantity'].pct_change() / features['unit_price'].pct_change()
        
        # 关税影响指标
        features['tariff_impact'] = features['tariff_rate'] * features['unit_price']
        features['effective_price'] = features['unit_price'] * (1 + features['tariff_rate'])
        
        return features.fillna(method='ffill').fillna(0)
    
    def create_sequences(self, data, target_col):
        """创建LSTM输入序列"""
        generator = TimeseriesGenerator(
            data=data,
            targets=data[target_col],
            length=self.window_size,
            sampling_rate=1,
            batch_size=32
        )
        return generator
```

### 2.2 LSTM模型架构

```python
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

class SoybeanLSTMModel:
    def __init__(self, input_shape, output_steps=6):
        self.input_shape = input_shape
        self.output_steps = output_steps
        self.model = self.build_model()
        
    def build_model(self):
        """构建多层LSTM模型"""
        model = models.Sequential([
            # 第一层LSTM - 捕捉短期模式
            layers.LSTM(128, return_sequences=True, 
                       input_shape=self.input_shape,
                       dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            
            # 第二层LSTM - 捕捉中期趋势
            layers.LSTM(64, return_sequences=True,
                       dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            
            # 第三层LSTM - 整合长期依赖
            layers.LSTM(32, return_sequences=False,
                       dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            
            # 全连接层
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # 输出层 - 多步预测
            layers.Dense(self.output_steps * 3)  # 预测量、价、份额
        ])
        
        # 自定义损失函数
        def custom_loss(y_true, y_pred):
            mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
            # 添加关税敏感性惩罚
            tariff_weight = tf.where(y_true[:, 2] > 0.1, 2.0, 1.0)  # 高关税期权重更高
            return mse * tariff_weight
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=custom_loss,
            metrics=['mae', 'mape']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        """训练模型"""
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        return history
```

### 2.3 注意力增强LSTM

```python
class AttentionLSTM(tf.keras.layers.Layer):
    """带注意力机制的LSTM层"""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.lstm = layers.LSTM(units, return_sequences=True, return_state=True)
        self.attention = layers.MultiHeadAttention(
            num_heads=4, key_dim=units//4
        )
        
    def call(self, inputs):
        lstm_out, final_h, final_c = self.lstm(inputs)
        
        # 自注意力
        attention_out = self.attention(
            query=lstm_out, value=lstm_out, key=lstm_out
        )
        
        # 残差连接
        output = layers.Add()([lstm_out, attention_out])
        output = layers.LayerNormalization()(output)
        
        return output, final_h, final_c

class EnhancedSoybeanModel:
    """增强版模型：LSTM + Attention + 多任务学习"""
    def __init__(self, feature_dim, sequence_length):
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        
    def build_model(self):
        # 输入层
        inputs = layers.Input(shape=(self.sequence_length, self.feature_dim))
        
        # 特征提取分支
        x, h, c = AttentionLSTM(128)(inputs)
        x = layers.GlobalAveragePooling1D()(x)
        
        # 多任务输出头
        # 1. 进口量预测
        quantity_branch = layers.Dense(64, activation='relu')(x)
        quantity_output = layers.Dense(6, name='quantity')(quantity_branch)
        
        # 2. 价格预测
        price_branch = layers.Dense(64, activation='relu')(x)
        price_output = layers.Dense(6, name='price')(price_branch)
        
        # 3. 市场份额预测
        share_branch = layers.Dense(64, activation='relu')(x)
        share_output = layers.Dense(18, activation='softmax', name='share')(share_branch)  # 3国×6月
        
        model = models.Model(
            inputs=inputs,
            outputs=[quantity_output, price_output, share_output]
        )
        
        # 多任务损失权重
        model.compile(
            optimizer='adam',
            loss={
                'quantity': 'mse',
                'price': 'mse',
                'share': 'categorical_crossentropy'
            },
            loss_weights={
                'quantity': 1.0,
                'price': 0.5,
                'share': 0.3
            }
        )
        
        return model
```

## 3. 情景模拟与预测

### 3.1 政策情景生成器

```python
class TariffScenarioGenerator:
    """关税政策情景生成器"""
    def __init__(self, base_model):
        self.base_model = base_model
        
    def generate_scenarios(self):
        scenarios = {
            'baseline': {
                'us_tariff': 0.13,  # 2024年水平
                'retaliation': False
            },
            'reciprocal': {
                'us_tariff': 0.275,  # 恢复高关税
                'retaliation': False
            },
            'trade_war': {
                'us_tariff': 0.50,  # 极端情景
                'retaliation': True,
                'china_counter': 0.25
            },
            'de_escalation': {
                'us_tariff': 0.05,  # 缓和情景
                'retaliation': False
            }
        }
        return scenarios
    
    def predict_scenario(self, scenario, horizon=24):
        """预测特定情景下的贸易流"""
        # 构建情景特征
        scenario_features = self.encode_scenario(scenario)
        
        predictions = {}
        current_state = self.get_current_state()
        
        for month in range(horizon):
            # 递归预测
            next_pred = self.base_model.predict(
                np.concatenate([current_state, scenario_features])
            )
            
            predictions[f'month_{month+1}'] = {
                'quantity': next_pred[0],
                'price': next_pred[1],
                'market_share': next_pred[2]
            }
            
            # 更新状态
            current_state = self.update_state(current_state, next_pred)
        
        return predictions
```

### 3.2 不确定性量化

```python
class BayesianLSTM:
    """贝叶斯LSTM用于不确定性估计"""
    def __init__(self, n_ensemble=5):
        self.n_ensemble = n_ensemble
        self.models = []
        
    def build_bayesian_model(self, input_shape):
        """构建带MC Dropout的模型"""
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3, training=True),  # MC Dropout
            layers.LSTM(64),
            layers.Dropout(0.3, training=True),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3, training=True),
            layers.Dense(6)  # 6个月预测
        ])
        return model
    
    def predict_with_uncertainty(self, X, n_samples=100):
        """生成预测分布"""
        predictions = []
        
        for _ in range(n_samples):
            pred = np.mean([model.predict(X) for model in self.models], axis=0)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # 计算统计量
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # 95%置信区间
        lower_bound = np.percentile(predictions, 2.5, axis=0)
        upper_bound = np.percentile(predictions, 97.5, axis=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'ci_lower': lower_bound,
            'ci_upper': upper_bound
        }
```

## 4. 实现路线图

### 4.1 第一阶段：基础LSTM实现（24小时）
1. 数据预处理：月度数据整合、特征工程
2. 简单LSTM模型：单任务进口量预测
3. 基础评估：MAE、RMSE、MAPE

### 4.2 第二阶段：模型增强（48小时）
1. 注意力机制集成
2. 多任务学习框架
3. 贝叶斯不确定性量化

### 4.3 第三阶段：政策分析（72小时）
1. 情景模拟器开发
2. 敏感性分析
3. 可视化仪表板

## 5. 代码实现示例

```python
# main_q1_lstm.py
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # 1. 加载数据
    monthly_data = pd.read_csv('2025/data/external/q1_soybean_imports_comtrade_monthly.csv')
    annual_data = pd.read_csv('2025/data/processed/q1/q1_0.csv')
    
    # 2. 数据处理
    processor = SoybeanDataProcessor(window_size=12, forecast_horizon=6)
    features = processor.prepare_features(monthly_data)
    
    # 3. 模型训练
    model = EnhancedSoybeanModel(
        feature_dim=features.shape[1],
        sequence_length=12
    )
    
    # 4. 情景分析
    scenario_gen = TariffScenarioGenerator(model)
    scenarios = scenario_gen.generate_scenarios()
    
    results = {}
    for name, scenario in scenarios.items():
        predictions = scenario_gen.predict_scenario(scenario, horizon=24)
        results[name] = predictions
    
    # 5. 结果输出
    save_results(results, 'q1_lstm_predictions.json')
    plot_scenario_comparison(results)

if __name__ == '__main__':
    main()
```

## 6. 评估指标

### 6.1 预测精度
- MAE: 平均绝对误差 < 5%
- RMSE: 均方根误差
- MAPE: 平均绝对百分比误差 < 10%
- 方向准确率 > 80%

### 6.2 业务指标
- 市场份额预测准确度
- 关税弹性估计精度
- 转折点预测能力

## 7. 技术栈

```yaml
dependencies:
  - tensorflow==2.13.0
  - keras==2.13.1
  - scikit-learn==1.3.0
  - pandas==2.0.3
  - numpy==1.24.3
  - matplotlib==3.7.2
  - seaborn==0.12.2
  - plotly==5.15.0
```

## 8. 预期成果

1. **预测精度提升30%**：相比传统计量模型
2. **多步预测能力**：可靠的6-12个月预测
3. **不确定性量化**：提供预测置信区间
4. **政策洞察**：量化不同关税情景的影响

## 9. 参考文献

1. Hochreiter & Schmidhuber (1997). Long Short-Term Memory. Neural Computation.
2. Bahdanau et al. (2014). Neural Machine Translation by Jointly Learning to Align and Translate.
3. Lim et al. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.
