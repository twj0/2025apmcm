# Q1大豆贸易模型综合分析报告

## 执行摘要

经过对Q1数据的深入分析，发现模型存在严重的结构性问题，导致预测结果完全不可信。主要问题包括数据质量缺陷、模型架构不合理、以及缺乏必要的业务约束。

## 数据质量分析

### 1. 数据一致性检查

**q1_1.csv (月度数据)**
- 时间范围：2010-2019年（共419行）
- 字段：period, partnerDesc, netWgt, primaryValue, 数量(万吨), 金额(亿美元)
- 数据完整性：良好，无缺失值

**q1_0.csv (年度数据)**
- 时间范围：2015-2024年（共30行）
- 字段：year, exporter, import_quantity_tonnes, import_value_usd, unit_price_usd_per_ton, tariff_cn_on_exporter
- 关键发现：
  - 美国关税在2018年贸易战期间显著上升（28%→33%）
  - 2024年关税回落至13%
  - 阿根廷和巴西关税保持稳定（3%）

### 2. 数据匹配问题

**时间范围不匹配**：
- 月度数据：2010-2019
- 年度数据：2015-2024
- **影响**：导致模型训练数据不足，外推风险极高

**单位转换风险**：
- 月度数据使用"万吨"和"亿美元"
- 年度数据使用"吨"和"美元"
- **风险**：单位转换错误可能导致数量级错误

## 模型架构分析

### 1. SoybeanTradeModel (面板数据模型)

**优点**：
- 使用面板回归估计贸易弹性
- 考虑了关税影响和相对价格效应
- 情景分析框架完整

**缺点**：
- 缺乏对2020年疫情冲击的特殊处理
- 未考虑结构性变化（如贸易战）

### 2. SoybeanLSTMPipeline (深度学习模型)

**严重缺陷**：

1. **预测结果异常**：
   - import_quantity MAPE: 428.65%
   - unit_price MAPE: 203.54%
   - market_share MAPE: 417.83%

2. **物理意义错误**：
   - 预测出现负价格（如巴西2025年8月：-355.21美元/吨）
   - 数量波动巨大（月度变化超过1000%）

3. **模型架构问题**：
   - 缺乏价格非负约束
   - 特征工程存在数据泄漏
   - 超参数设置不合理（window_size=12, epochs=80）

## 关键问题识别

### 1. 数据预处理缺陷

```python
# 问题代码示例
features['price_elasticity'] = (
    features.groupby('exporter')['import_quantity'].pct_change() /
    (features.groupby('exporter')['unit_price'].pct_change() + 1e-6)
)
```

**问题**：
- pct_change()在价格接近0时会产生极端值
- 缺乏对异常值的有效处理
- 分组操作可能导致数据错位

### 2. 特征工程风险

```python
# 滞后变量设置
for lag in [1, 3, 6, 12]:
    features[f'quantity_lag_{lag}'] = features.groupby('exporter')['import_quantity'].shift(lag)
```

**风险**：
- 12个月滞后可能引入过时的信息
- 缺乏特征重要性评估
- 移动平均可能平滑掉重要信号

### 3. 模型训练问题

```python
# 预测结果后处理
pred_actual = np.exp(pred_log)  # 可能导致极端值
```

**缺陷**：
- 简单的指数变换无法保证合理性
- 缺乏业务约束（价格≥0，数量≥0）
- 市场份额未强制归一化

## 业务影响评估

### 1. 预测可信度
- **当前状态**：完全不可信（MAPE>400%）
- **业务影响**：无法用于政策制定或商业决策
- **风险等级**：极高

### 2. 情景分析有效性
- 关税情景分析框架合理
- 但基于错误的预测结果
- 需要重新校准模型参数

## 修复建议

### 1. 数据质量改进

```python
# 建议的数据预处理
class ImprovedDataProcessor:
    def clean_extreme_values(self, df, columns, method='iqr'):
        """使用IQR方法清理极端值"""
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower, upper)
        return df
    
    def validate_unit_consistency(self, monthly_df, annual_df):
        """验证单位一致性"""
        # 实现单位转换验证逻辑
        pass
```

### 2. 模型架构重构

```python
# 建议的约束LSTM模型
class ConstrainedSoybeanLSTM(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(config.lstm_units)
        self.price_dense = tf.keras.layers.Dense(1, activation='relu')  # 确保价格非负
        self.quantity_dense = tf.keras.layers.Dense(1, activation='relu')  # 确保数量非负
        
    def call(self, inputs):
        features = self.lstm(inputs)
        price = self.price_dense(features)
        quantity = self.quantity_dense(features)
        # 添加业务约束
        return tf.concat([price, quantity], axis=-1)
```

### 3. 训练策略优化

```python
# 建议的训练配置
IMPROVED_CONFIG = {
    'window_size': 6,  # 减少到6个月
    'forecast_horizon': 6,  # 减少到6个月
    'batch_size': 16,  # 减少批次大小
    'epochs': 50,  # 减少训练轮次
    'learning_rate': 0.001,  # 降低学习率
    'early_stopping_patience': 10,
    'validation_split': 0.3,  # 增加验证集比例
}
```

### 4. 业务约束强化

```python
def apply_business_constraints(predictions):
    """应用业务约束"""
    # 价格约束：必须为正数，且在合理范围内
    predictions['unit_price'] = predictions['unit_price'].clip(200, 2000)
    
    # 数量约束：必须为正数
    predictions['import_quantity'] = predictions['import_quantity'].clip(1000, None)
    
    # 市场份额约束：必须归一化为1
    predictions['market_share'] = predictions.groupby('date')['market_share'].transform(
        lambda x: x / x.sum()
    )
    
    return predictions
```

## 实施优先级

### 高优先级（立即执行）
1. **数据验证**：重新检查单位转换和数据一致性
2. **模型禁用**：暂停使用当前LSTM模型
3. **约束添加**：实施价格和数量的非负约束

### 中优先级（1-2周内）
1. **特征工程**：重新设计特征选择逻辑
2. **超参数调优**：使用网格搜索优化模型参数
3. **交叉验证**：实施时间序列交叉验证

### 低优先级（1个月内）
1. **模型集成**：结合多种预测方法
2. **不确定性量化**：添加预测区间
3. **自动化监控**：建立模型性能监控体系

## 结论

Q1大豆贸易模型存在严重的结构性问题，主要体现在数据质量、模型架构和业务约束三个方面。当前模型产生的预测结果完全不可信，需要立即进行重大修复。建议优先处理数据一致性问题，重构模型架构，并强化业务约束，以确保预测结果的合理性和可信度。

修复后的模型应该能够将MAPE控制在20%以内，预测价格保持正值，并且市场份额变化符合经济逻辑。