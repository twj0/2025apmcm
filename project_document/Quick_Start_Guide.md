# APMCM 2025 C题 快速启动指南

**更新时间:** 2025-11-21  
**目标:** 5分钟内运行所有模型

---

## 🚀 快速开始（5分钟）

### Step 1: 环境准备（1分钟）
```bash
# 确保在项目根目录 SPEC/
cd "D:/Mathematical Modeling/2025APMCM/SPEC"

# 安装依赖
uv sync
```

### Step 2: 数据准备（1分钟）
```bash
# 运行数据预处理脚本
uv run python 2025/src/preprocessing/prepare_data.py
```

预期输出：
```
✓ Q1 data preparation completed
✓ Q2 data preparation completed  
✓ Q3 data preparation completed
✓ Q4 data preparation completed
✓ Q5 data preparation completed
✅ All data preparation completed successfully!
```

### Step 3: 运行模型（3分钟）

#### 选项A: 运行单个问题（快速测试）
```bash
# 测试Q1 LSTM模型
uv run python 2025/src/main.py --questions Q1
```

#### 选项B: 运行所有模型（无ML增强，速度快）
```bash
uv run python 2025/src/main.py --no-ml
```

#### 选项C: 运行完整分析（含ML增强，较慢）
```bash
uv run python 2025/src/main.py
```

### Step 4: 生成可视化
```bash
# 生成所有图表
uv run python 2025/src/main.py --visualize
```

---

## 📊 输出位置

运行完成后，结果保存在：

```
SPEC/
├── 2025/
│   ├── results/          # 模型运行结果
│   │   ├── q1/           # Q1大豆贸易结果
│   │   │   ├── econometric/   # 计量经济模型
│   │   │   └── lstm/          # LSTM预测结果
│   │   ├── q2/           # Q2汽车产业结果
│   │   │   ├── econometric/   # OLS回归
│   │   │   └── marl/          # MARL博弈分析
│   │   ├── q3/           # Q3半导体结果
│   │   │   ├── econometric/   # 分段回归
│   │   │   └── gnn/           # GNN风险分析
│   │   ├── q4/           # Q4关税收入结果
│   │   │   └── econometric/   # Laffer曲线
│   │   └── q5/           # Q5宏观经济结果
│   │       ├── econometric/   # VAR模型
│   │       └── transformer/   # Transformer预测
│   │
│   └── figures/          # 可视化图表
│       ├── q1/          # Q1图表（PDF格式）
│       ├── q2/          # Q2图表
│       ├── q3/          # Q3图表
│       ├── q4/          # Q4图表
│       └── q5/          # Q5图表
```

---

## 🔧 常用命令

### 基础命令
```bash
# 查看帮助
uv run python 2025/src/main.py --help

# 运行特定问题
uv run python 2025/src/main.py --questions Q1 Q3 Q5

# 调试模式
uv run python 2025/src/main.py --log-level DEBUG
```

### 高级选项
```bash
# 运行Q2的MARL分析
uv run python 2025/src/main.py --questions Q2

# 运行Q4的DRL优化（需要先实现）
uv run python 2025/src/main.py --questions Q4

# 生成所有可视化（不运行模型）
uv run python 2025/src/visualization/run_all_visualizations.py
```

---

## ⚠️ 常见问题

### 1. 数据文件缺失
**错误:** `FileNotFoundError: q1_1.csv`

**解决:** 运行数据准备脚本
```bash
uv run python 2025/src/preprocessing/prepare_data.py
```

### 2. 依赖包缺失
**错误:** `ModuleNotFoundError: No module named 'tensorflow'`

**解决:** 重新安装依赖
```bash
uv sync
# 或手动安装
uv pip install tensorflow scikit-learn statsmodels
```

### 3. 内存不足
**错误:** `MemoryError`

**解决:** 使用无ML模式
```bash
uv run python 2025/src/main.py --no-ml
```

### 4. Q4 DRL未实现
**错误:** `AttributeError: 'TariffRevenueModel' object has no attribute 'run_drl_analysis'`

**解决:** Q4的DRL增强尚未实现，使用基础模式
```bash
uv run python 2025/src/main.py --questions Q4 --no-ml
```

---

## 📈 检查结果

### 验证输出文件
```bash
# 检查Q1结果
ls -la 2025/results/q1/

# 检查所有结果
find 2025/results -name "*.json" -o -name "*.csv"

# 检查图表
ls -la 2025/figures/
```

### 查看汇总报告
```bash
# 每个问题都有SUMMARY.md
cat 2025/results/q1/SUMMARY.md
cat 2025/results/q2/SUMMARY.md
```

### 查看日志
```bash
# 查看运行日志
cat 2025/results/logs/analysis.log

# 查看错误
grep ERROR 2025/results/logs/analysis.log
```

---

## 🎯 性能基准

| 模型 | 运行时间 | 内存占用 | GPU需求 |
|------|---------|---------|---------|
| Q1 LSTM | ~30秒 | 2GB | 可选 |
| Q2 MARL | ~45秒 | 3GB | 不需要 |
| Q3 GNN | ~20秒 | 1.5GB | 不需要 |
| Q4 基础 | ~15秒 | 1GB | 不需要 |
| Q5 Transformer | ~60秒 | 4GB | 推荐 |
| **总计（无ML）** | **~2分钟** | **4GB** | **不需要** |
| **总计（含ML）** | **~5分钟** | **8GB** | **推荐** |

---

## 🔄 下一步

1. **添加Q4 DRL增强**
   - 参考 `project_document/Q4_DRL_Technical_Guide.md`
   - 实现SAC算法

2. **优化可视化**
   - 调整图表样式
   - 添加交互式图表

3. **撰写论文**
   - 使用 `2025/paper/` 目录下的模板
   - 引用 `results/` 中的数据

---

## 📞 支持

- 查看完整文档: `project_document/`
- 技术指南: `Q1-Q5_*_Technical_Guide.md`
- 代码结构: `PROJECT_STRUCTURE.md`

---

*快速启动指南 v1.0 - 2024.11.21*
