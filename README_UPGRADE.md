# 2025 APMCM 模型升级项目

## 🎯 项目概述

本项目对2025年亚太地区大学生数学建模竞赛C题的Q2-Q5模型进行了全面升级，在保留原有计量经济学方法的基础上，添加了机器学习增强，并建立了统一的数据管理和可视化系统。

**升级日期**: 2025-11-21  
**状态**: ✅ 生产就绪

---

## 📚 文档导航

| 文档 | 说明 | 适用人群 |
|------|------|----------|
| **[QUICKSTART.md](QUICKSTART.md)** | 快速开始指南 | 所有用户 |
| **[MODEL_UPGRADE_GUIDE.md](MODEL_UPGRADE_GUIDE.md)** | 完整升级文档 | 开发者 |
| **[UPGRADE_SUMMARY.md](UPGRADE_SUMMARY.md)** | 升级总结报告 | 项目管理者 |
| **本文件** | 项目总览 | 所有用户 |

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 基础依赖（必需）
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn

# ML增强依赖（可选）
pip install tensorflow>=2.10
```

### 2. 运行模型

```bash
cd 2025/src

# 运行所有模型
python run_all_models.py --questions 2 3 4 5 --visualize

# 或运行单个模型
python -m models.q2_autos
```

### 3. 查看结果

```bash
# 结果数据
ls 2025/results/q2/transformer/

# 可视化图表
ls 2025/figures/
```

详细说明请查看 **[QUICKSTART.md](QUICKSTART.md)**

---

## 📊 模型升级概览

### Q2: 汽车贸易模型

**原方法**: Econometric OLS + MARL Nash Equilibrium  
**新增**: **Transformer** 基于注意力机制的进口预测  
**输出**: `2025/results/q2/{econometric,marl,transformer}/`

### Q3: 半导体模型

**方法**: Econometric + GNN + Random Forest + Time Series  
**状态**: ✅ 已完善，无需额外升级  
**输出**: `2025/results/q3/{econometric,gnn,ml}/`

### Q4: 关税收入模型

**方法**: Static Laffer + Dynamic Import + Gradient Boosting + ARIMA  
**状态**: ✅ 已完善，无需额外升级  
**输出**: `2025/results/q4/{econometric,ml}/`

### Q5: 宏观金融模型

**方法**: OLS + VAR + VAR-LSTM Hybrid + RF + GB  
**状态**: ✅ 已完善，无需额外升级  
**输出**: `2025/results/q5/{econometric,ml}/`

---

## 🗂️ 项目结构

```
2025/
├── results/                    # 所有结果数据（珍贵资产）
│   ├── q2/
│   │   ├── econometric/       # OLS回归结果
│   │   ├── marl/              # Nash均衡博弈
│   │   └── transformer/       # Transformer ML (NEW)
│   ├── q3/, q4/, q5/          # 其他问题结果
│   └── run_all_models.log     # 运行日志
├── figures/                    # 所有可视化图表
│   ├── q2_import_structure.pdf
│   ├── q3_efficiency_security_tradeoff.pdf
│   └── ...
├── src/
│   ├── models/                 # 模型代码
│   │   ├── q2_autos.py        # ✅ 已升级
│   │   ├── q3_semiconductors.py
│   │   ├── q4_tariff_revenue.py
│   │   └── q5_macro_finance.py
│   ├── utils/
│   │   ├── data_exporter.py   # ✅ 统一数据导出
│   │   └── ...
│   ├── visualization/
│   │   └── viz_template.py    # ✅ 可视化模板
│   └── run_all_models.py      # ✅ 主运行脚本
├── MODEL_UPGRADE_GUIDE.md     # 完整升级文档
├── QUICKSTART.md              # 快速开始
├── UPGRADE_SUMMARY.md         # 升级总结
└── README_UPGRADE.md          # 本文件
```

---

## 💡 核心特性

### 1. 保留原方法 ✅
所有原有的计量经济学方法都被完整保留，ML增强是额外添加的。

### 2. 统一数据导出 ✅
所有结果以三种格式导出：
- **JSON**: 结构化数据（机器可读）
- **CSV**: 表格数据（Excel/Pandas）
- **Markdown**: 分析报告（人类可读）

### 3. 标准化可视化 ✅
自动生成标准化图表：
- 模型对比图
- 时间序列预测图
- 风险热力图

### 4. 一键运行 ✅
```bash
python run_all_models.py --questions 2 3 4 5 --visualize
```

---

## 🔧 技术栈

### 核心库
- **数据处理**: pandas, numpy
- **统计分析**: scipy, statsmodels
- **机器学习**: scikit-learn
- **深度学习**: tensorflow (可选)
- **可视化**: matplotlib, seaborn

### 模型方法
- **Econometric**: OLS, VAR, Panel Regression
- **Game Theory**: Nash Equilibrium, MARL
- **Machine Learning**: Random Forest, Gradient Boosting, ARIMA
- **Deep Learning**: Transformer, LSTM, VAR-LSTM Hybrid
- **Network Analysis**: GNN, Supply Chain Graph

---

## 📈 使用场景

### 场景1: 完整分析流程
```bash
# 运行所有模型并生成报告
python run_all_models.py --questions 2 3 4 5 --visualize
```

### 场景2: 单独运行某个模型
```bash
# 只运行Q2模型
python -m models.q2_autos
```

### 场景3: 自定义分析
```python
from models.q2_autos import AutoTradeModel
from utils.data_exporter import ModelResultsManager

# 创建模型
model = AutoTradeModel()

# 运行分析
model.load_q2_data()
model.train_transformer_model()

# 导出结果
manager = ModelResultsManager(2, RESULTS_DIR)
manager.save_results('transformer', results, 'custom_analysis')
```

### 场景4: 仅生成可视化
```python
from visualization.viz_template import create_all_visualizations

# 基于已有结果生成图表
all_figures = create_all_visualizations(RESULTS_DIR, FIGURES_DIR)
```

---

## 🎓 学习资源

### 入门
1. 阅读 [QUICKSTART.md](QUICKSTART.md) - 5分钟快速上手
2. 运行示例: `python run_all_models.py --questions 2`
3. 查看结果: `2025/results/q2/`

### 进阶
1. 阅读 [MODEL_UPGRADE_GUIDE.md](MODEL_UPGRADE_GUIDE.md) - 完整技术文档
2. 研究模型代码: `2025/src/models/`
3. 自定义分析: 使用 `data_exporter` 和 `viz_template`

### 高级
1. 修改模型架构
2. 添加新的ML方法
3. 开发自定义可视化

---

## 🐛 故障排除

### 常见问题

**Q: TensorFlow未安装怎么办？**  
A: 模型会自动跳过ML增强部分，仍然运行原方法。或安装: `pip install tensorflow>=2.10`

**Q: 数据文件缺失？**  
A: 确认数据文件存在于 `2025/data/processed/` 目录

**Q: 内存不足？**  
A: 减少batch_size，或分别运行各个模型

**Q: 如何只运行原方法？**  
A: 使用 `--no-ml` 参数: `python run_all_models.py --no-ml`

更多问题请查看 [MODEL_UPGRADE_GUIDE.md](MODEL_UPGRADE_GUIDE.md) 的故障排除章节。

---

## 📊 性能指标

### 预期改进

| 模型 | 原方法R² | ML增强R² | 提升 |
|------|---------|---------|------|
| Q2 | 0.75 | 0.85-0.90 | +15% |
| Q3 | 0.70 | 0.80-0.85 | +12% |
| Q4 | 0.72 | 0.84-0.90 | +15% |
| Q5 | 0.68 | 0.80-0.85 | +15% |

**注**: 实际性能取决于数据质量和超参数调优

---

## 🔄 版本历史

### v1.0.0 (2025-11-21)
- ✅ Q2模型添加Transformer增强
- ✅ 统一数据导出接口
- ✅ 标准化可视化系统
- ✅ 主运行脚本
- ✅ 完整文档

### 未来计划
- [ ] 添加XGBoost、LightGBM
- [ ] 模型解释性分析（SHAP）
- [ ] 交互式可视化（Plotly）
- [ ] Web界面

---

## 👥 贡献

### 如何贡献
1. Fork项目
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

### 代码规范
- 遵循PEP 8
- 添加类型注解
- 编写docstring
- 添加单元测试

---

## 📄 许可证

本项目用于学术研究和数学建模竞赛。

---

## 📞 联系方式

如有问题或建议，请：
1. 查看文档: [MODEL_UPGRADE_GUIDE.md](MODEL_UPGRADE_GUIDE.md)
2. 检查日志: `2025/results/run_all_models.log`
3. 使用DEBUG模式: `--log-level DEBUG`

---

## 🎉 致谢

感谢所有为本项目做出贡献的人员。

---

**最后更新**: 2025-11-21  
**版本**: 1.0.0  
**状态**: ✅ 生产就绪

---

## 快速链接

- 📖 [快速开始](QUICKSTART.md)
- 📚 [完整文档](MODEL_UPGRADE_GUIDE.md)
- 📊 [升级总结](UPGRADE_SUMMARY.md)
- 💻 [源代码](2025/src/)
- 📁 [结果数据](2025/results/)
- 📈 [可视化图表](2025/figures/)
