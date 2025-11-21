# 模型升级与机器学习增强指南

## 概述

本文档说明了Q2-Q5模型的升级方案，包括机器学习增强、数据管理和可视化系统。

**升级日期**: 2025-11-21  
**升级目标**: 在保留原有方法基础上，为每个模型添加机器学习增强，建立统一的数据导出和可视化系统

---

## 核心升级内容

### 1. 模型增强方案

#### Q2: 汽车贸易模型 (Auto Trade)
**原有方法**:
- Econometric OLS: 进口结构估计
- MARL: Nash均衡博弈分析

**新增方法**:
- **Transformer**: 基于注意力机制的进口预测
  - 多头注意力层捕捉跨国贸易模式
  - 时序特征工程（滞后项、移动平均）
  - 输出: `2025/results/q2/transformer/`

**技术路线**:
```python
from models.q2_autos import run_q2_analysis

# 运行完整分析（包含Transformer）
run_q2_analysis(use_transformer=True)
```

**输出数据**:
- `training_results.json`: 训练指标和历史
- `predictions.csv`: 预测结果
- `analysis_report.md`: 分析报告

---

#### Q3: 半导体模型 (Semiconductors)
**原有方法**:
- Econometric: 分段回归分析
- GNN: 供应链图网络风险分析
- Random Forest: 贸易预测

**已有ML增强**:
- **Random Forest**: 分段贸易预测
- **Time Series Forecasting**: 供应链风险预测

**输出结构**:
```
2025/results/q3/
├── econometric/     # OLS回归结果
├── gnn/             # 图网络风险分析
└── ml/              # ML预测结果
    ├── trade_predictions.json
    ├── risk_forecasting.json
    └── comparison_report.md
```

---

#### Q4: 关税收入模型 (Tariff Revenue)
**原有方法**:
- Static Laffer Curve: 静态收入最大化
- Dynamic Import Response: 动态进口响应

**已有ML增强**:
- **Gradient Boosting**: 收入预测
- **ARIMA**: 时间序列预测

**输出结构**:
```
2025/results/q4/
├── econometric/     # Laffer曲线和动态模型
└── ml/              # ML预测
    ├── gb_model_metrics.json
    ├── arima_model_metrics.json
    ├── ml_revenue_forecasts.csv
    └── model_comparison.json
```

---

#### Q5: 宏观金融模型 (Macro/Financial)
**原有方法**:
- OLS Regression: 宏观效应回归
- VAR: 向量自回归
- Event Study: 制造业回流分析

**已有ML增强**:
- **VAR-LSTM Hybrid**: 混合脉冲响应预测
- **Random Forest + Gradient Boosting**: 制造业回流预测

**输出结构**:
```
2025/results/q5/
├── econometric/     # VAR和回归结果
└── ml/              # ML混合模型
    ├── var_lstm_hybrid.json
    ├── reshoring_ml_models.json
    ├── feature_importance.json
    └── model_comparison.json
```

---

## 2. 统一数据管理系统

### 数据导出接口

创建了 `utils/data_exporter.py` 提供统一的数据导出功能：

```python
from utils.data_exporter import ModelResultsManager

# 初始化管理器
manager = ModelResultsManager(question_number=2, results_base_dir=RESULTS_DIR)

# 注册方法
manager.register_method('transformer')

# 保存结果（自动导出JSON、CSV、MD）
results = {
    'metrics': {'rmse': 0.5, 'r2': 0.85},
    'predictions': predictions_df
}
manager.save_results('transformer', results, 'training_results')

# 生成摘要报告
manager.generate_summary()
```

### 数据格式标准

所有结果文件遵循以下格式：

**JSON格式**:
```json
{
  "metadata": {
    "exported_at": "2025-11-21T17:23:00",
    "method": "transformer",
    "question": "q2"
  },
  "data": {
    "metrics": {...},
    "results": [...]
  }
}
```

**CSV格式**:
- 表格数据，包含所有预测和实际值
- 第一行为列名
- UTF-8编码

**Markdown格式**:
- 人类可读的分析报告
- 包含关键指标和可视化建议

---

## 3. 可视化系统

### 可视化模板

创建了 `visualization/viz_template.py` 提供标准化可视化：

```python
from visualization.viz_template import create_all_visualizations

# 生成所有可视化
all_figures = create_all_visualizations(RESULTS_DIR, FIGURES_DIR)

# 结果: {'q2': [fig1.pdf, fig2.pdf], 'q3': [...], ...}
```

### 可视化类型

1. **模型对比图** (`ModelComparisonVisualizer`)
   - 指标对比柱状图
   - 预测vs实际散点图

2. **时间序列图** (`TimeSeriesVisualizer`)
   - 历史数据+预测
   - 多场景对比

3. **网络图** (`NetworkVisualizer`)
   - 风险热力图
   - 供应链网络图

### 自定义可视化

```python
from visualization.viz_template import ResultsVisualizer

viz = ResultsVisualizer(RESULTS_DIR / 'q2', FIGURES_DIR)

# 加载数据
data = viz.load_json('transformer', 'training_results')
df = viz.load_csv('transformer', 'predictions')

# 创建自定义图表
fig, ax = plt.subplots()
# ... 绘图代码 ...
viz.save_figure(fig, 'custom_plot.pdf')
```

---

## 4. 运行流程

### 方式1: 运行单个模型

```bash
# Q2模型（包含Transformer）
cd 2025/src
python -m models.q2_autos

# Q3模型（包含GNN和ML）
python -m models.q3_semiconductors

# Q4模型（包含ML）
python -m models.q4_tariff_revenue

# Q5模型（包含VAR-LSTM）
python -m models.q5_macro_finance
```

### 方式2: 运行所有模型

```bash
# 运行所有模型并生成可视化
python run_all_models.py --questions 2 3 4 5 --visualize

# 仅运行Q2和Q3
python run_all_models.py --questions 2 3

# 禁用ML增强（仅运行原方法）
python run_all_models.py --no-ml

# 调试模式
python run_all_models.py --log-level DEBUG
```

### 方式3: Python脚本

```python
from models.q2_autos import run_q2_analysis
from models.q3_semiconductors import run_q3_analysis
from visualization.viz_template import create_all_visualizations

# 运行模型
run_q2_analysis(use_transformer=True)
run_q3_analysis()

# 生成可视化
create_all_visualizations(RESULTS_DIR, FIGURES_DIR)
```

---

## 5. 目录结构

### 完整结构

```
2025/
├── results/                    # 所有结果数据（珍贵资产）
│   ├── q2/
│   │   ├── econometric/       # OLS结果
│   │   │   ├── summary.json
│   │   │   ├── scenario_imports.csv
│   │   │   └── scenario_industry.csv
│   │   ├── marl/              # Nash均衡结果
│   │   │   ├── nash_equilibrium.json
│   │   │   ├── payoff_matrix.csv
│   │   │   └── analysis_report.md
│   │   ├── transformer/       # Transformer ML结果
│   │   │   ├── training_results.json
│   │   │   ├── predictions.csv
│   │   │   └── analysis_report.md
│   │   └── SUMMARY.md         # Q2汇总报告
│   ├── q3/
│   │   ├── econometric/
│   │   ├── gnn/
│   │   ├── ml/
│   │   └── SUMMARY.md
│   ├── q4/
│   │   ├── econometric/
│   │   ├── ml/
│   │   └── SUMMARY.md
│   └── q5/
│       ├── econometric/
│       ├── ml/
│       └── SUMMARY.md
├── figures/                    # 所有可视化图表
│   ├── q2_import_structure.pdf
│   ├── q2_industry_impact.pdf
│   ├── q3_efficiency_security_tradeoff.pdf
│   ├── q4_revenue_time_path.pdf
│   ├── q5_time_series_overview.pdf
│   └── ...
└── src/
    ├── models/                 # 模型代码
    │   ├── q2_autos.py
    │   ├── q3_semiconductors.py
    │   ├── q4_tariff_revenue.py
    │   └── q5_macro_finance.py
    ├── utils/                  # 工具模块
    │   ├── data_exporter.py   # 统一数据导出
    │   ├── data_loader.py
    │   └── config.py
    ├── visualization/          # 可视化模块
    │   └── viz_template.py
    └── run_all_models.py       # 主运行脚本
```

---

## 6. 依赖要求

### Python包

```bash
# 核心依赖
pip install pandas numpy scipy statsmodels scikit-learn

# 可视化
pip install matplotlib seaborn

# 深度学习（可选，用于Transformer和LSTM）
pip install tensorflow>=2.10

# 或使用PyTorch
pip install torch torchvision
```

### 环境配置

```bash
# 使用uv（推荐）
uv pip install -r requirements.txt

# 或使用pip
pip install -r requirements.txt
```

---

## 7. 数据完整性检查

### 检查脚本

```python
from pathlib import Path
from utils.data_exporter import ModelResultsManager

def check_data_integrity(question_number: int):
    """检查指定问题的数据完整性"""
    manager = ModelResultsManager(question_number, RESULTS_DIR)
    
    # 列出所有结果
    all_results = manager.list_all_results()
    
    print(f"\n=== Q{question_number} Data Integrity Check ===")
    for method, files in all_results.items():
        print(f"\n{method.upper()}:")
        print(f"  Total files: {len(files)}")
        
        # 检查必需文件
        has_json = any(f.endswith('.json') for f in files)
        has_csv = any(f.endswith('.csv') for f in files)
        has_md = any(f.endswith('.md') for f in files)
        
        print(f"  ✓ JSON: {has_json}")
        print(f"  ✓ CSV: {has_csv}")
        print(f"  ✓ Markdown: {has_md}")

# 检查所有问题
for q in [2, 3, 4, 5]:
    check_data_integrity(q)
```

---

## 8. 常见问题

### Q: TensorFlow未安装，Transformer模型无法运行？
**A**: Transformer是可选增强。如果TensorFlow未安装，模型会自动跳过Transformer部分，仍然运行原有的Econometric和MARL方法。

安装TensorFlow:
```bash
pip install tensorflow>=2.10
# 或GPU版本
pip install tensorflow-gpu>=2.10
```

### Q: 如何只运行原方法，不运行ML增强？
**A**: 使用 `--no-ml` 参数：
```bash
python run_all_models.py --no-ml
```

或在代码中：
```python
run_q2_analysis(use_transformer=False)
run_q4_analysis(use_ml=False)
```

### Q: 结果数据保存在哪里？
**A**: 所有结果保存在 `2025/results/q{N}/{method}/` 目录下，包括：
- JSON: 结构化数据
- CSV: 表格数据
- MD: 分析报告

### Q: 如何添加自定义可视化？
**A**: 继承 `ResultsVisualizer` 类：
```python
from visualization.viz_template import ResultsVisualizer

class MyCustomVisualizer(ResultsVisualizer):
    def plot_custom(self, ...):
        # 自定义绘图逻辑
        pass
```

### Q: 模型运行失败怎么办？
**A**: 
1. 检查日志文件: `2025/results/run_all_models.log`
2. 确认数据文件存在: `2025/data/processed/`
3. 检查依赖包是否安装完整
4. 使用DEBUG模式: `--log-level DEBUG`

---

## 9. 性能优化建议

### 数据缓存
- 首次运行会处理原始数据，后续运行使用缓存
- 缓存位置: `2025/data/processed/`

### 并行处理
- 不同问题可以并行运行
- 使用多进程: `python run_all_models.py --questions 2 3 &`

### 内存管理
- 大数据集使用分块处理
- 及时释放不需要的DataFrame

---

## 10. 下一步计划

### 短期（1-2周）
- [ ] 完善Q2 Transformer方法的实现
- [ ] 添加更多特征工程
- [ ] 优化超参数

### 中期（1个月）
- [ ] 集成更多ML模型（XGBoost, LightGBM）
- [ ] 添加模型解释性分析（SHAP）
- [ ] 开发交互式可视化（Plotly）

### 长期（2-3个月）
- [ ] 构建模型集成（Ensemble）
- [ ] 添加不确定性量化
- [ ] 开发Web界面展示结果

---

## 11. 贡献指南

### 添加新方法
1. 在对应的模型文件中添加方法类
2. 在 `results_base` 下创建新的方法目录
3. 使用 `DataExporter` 导出结果
4. 在 `viz_template.py` 中添加可视化

### 代码规范
- 遵循PEP 8
- 添加类型注解
- 编写docstring
- 添加单元测试

---

## 12. 联系方式

如有问题或建议，请联系项目维护者。

**最后更新**: 2025-11-21  
**版本**: 1.0.0
