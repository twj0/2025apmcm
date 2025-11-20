# APMCM 2025 Project Structure

## 完整目录树

```
2025APMCM-ProblemC/
│
├── README.md                          # 项目总览（人类 + .0modeler 首选入口）
├── requirements.txt                   # Python依赖（或 pyproject.toml）
├── .gitignore                         # Git忽略文件
│
├── data/                              # 数据目录（.0mathcoder 负责可复现管道）
│   ├── README.md                      # 数据说明（.0modeler 提供变量含义）
│   ├── raw/                           # 原始数据（只读）
│   ├── processed/                     # 清洗 & 特征工程结果
│   └── external/                      # 外部数据（可选）
│
├── src/                               # 源代码目录（主战场：.0mathcoder）
│   ├── preprocessing/                 # 数据预处理：加载、清洗、特征工程
│   ├── models/                        # 模型实现：时间序列 / 回归 / 优化 / 集成
│   ├── analysis/                      # 分析模块：EDA / 验证 / 敏感性分析
│   ├── visualization/                 # 可视化：绘图函数与样式
│   └── utils/                         # 工具函数：配置、指标、辅助方法
│
├── notebooks/                         # Jupyter笔记本（可选，探索性分析）
│
├── scripts/                           # 执行脚本（面向人类 + .0checker）
│   ├── run_q1.py                      # 问题1主流程
│   ├── run_q2.py                      # 问题2主流程
│   ├── run_q3.py                      # 问题3主流程
│   ├── run_q4.py                      # 问题4主流程（如有）
│   ├── run_all.py                     # 一键运行所有问题
│   └── generate_figures.py            # 统一生成图表
│
├── results/                           # 结果输出（.0checker 对齐论文 & 代码）
│   ├── README.md                      # 结果说明
│   ├── predictions/                   # 各问题预测结果
│   ├── metrics/                       # 评估指标（单模型 & 多模型对比）
│   ├── tables/                        # LaTeX表格（供论文直接引用）
│   └── logs/                          # 运行日志
│
├── figures/                           # 图表输出（PDF，供论文直接引用）
│
├── paper/                             # 论文目录（.0paperwriter 主战场）
│   ├── README.md                      # 论文说明
│   ├── main.tex                       # 主论文文件
│   ├── apmcmthesis.cls                # 模板类文件
│   ├── references.bib                 # 参考文献
│   ├── sections/                      # 论文章节（可选分离）
│   ├── figures/                       # 论文图片（通常链接到 ../figures/）
│   └── tables/                        # 论文表格（通常链接到 ../results/tables/）
│
├── tests/                             # 测试代码（.0checker 重点目录，可选）
│
├── docs/                              # 文档目录（.0modeler 维护大纲与说明）
│   ├── modeling_outline.md            # 建模大纲
│   ├── data_dictionary.md             # 数据字典
│   ├── methodology.md                 # 方法论说明
│   └── team_notes.md                  # 团队笔记
│
└── submission/                        # 提交文件
    ├── paper.pdf                      # 最终论文PDF
    ├── code/                          # 代码压缩包
    └── supplementary/                 # 补充材料
```

## 目录说明

### 1. 根目录文件
- **README.md**: 项目概述、快速开始、团队信息
- **requirements.txt**: Python依赖包列表
- **.gitignore**: Git版本控制忽略文件

### 2. data/ - 数据管理
- **raw/**: 原始数据，不修改
- **processed/**: 清洗后的数据
- **external/**: 额外收集的数据

### 3. src/ - 源代码（模块化）
- **preprocessing/**: 数据预处理管道
- **models/**: 各类模型实现
- **analysis/**: 分析工具
- **visualization/**: 可视化工具
- **utils/**: 通用工具函数

### 4. scripts/ - 执行脚本
- 每个问题一个独立脚本
- `run_all.py` 一键运行所有分析

### 5. results/ - 结果存储
- **predictions/**: 预测结果CSV
- **metrics/**: 评估指标JSON
- **tables/**: LaTeX格式表格
- **logs/**: 执行日志

### 6. figures/ - 图表输出
- 按问题分类存储
- 所有图表PDF格式（矢量图）
- 300 DPI以上

### 7. paper/ - 论文写作
- LaTeX源文件
- 可选：章节分离便于协作
- 符号链接到figures和tables

### 8. tests/ - 单元测试（可选）
- 确保代码质量
- 便于调试

### 9. docs/ - 文档
- 建模大纲
- 数据说明
- 团队协作笔记

### 10. submission/ - 最终提交
- 编译好的PDF
- 打包的代码
- 补充材料

## 工作流程

### Phase 1: 项目初始化
```bash
# 创建目录结构
python scripts/setup_project.py

# 安装依赖
pip install -r requirements.txt
```

### Phase 2: 数据处理
```bash
# 运行数据预处理
python scripts/run_preprocessing.py
```

### Phase 3: 模型开发
```bash
# 运行各问题
python scripts/run_q1.py
python scripts/run_q2.py
python scripts/run_q3.py
# 如有问题4：
# python scripts/run_q4.py

# 或一次运行全部
python scripts/run_all.py
```

### Phase 4: 结果生成
```bash
# 生成所有结果和图表
python scripts/generate_results.py
```

### Phase 5: 论文编写
```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Phase 6: 提交准备
```bash
# 打包提交文件
python scripts/prepare_submission.py
```

## 命名规范

### 文件命名
- Python文件: `snake_case.py`
- 数据文件: `descriptive_name.csv`
- 图表文件: `q1_forecast.pdf`, `sensitivity_analysis.pdf`
- LaTeX文件: `main.tex`, `section_name.tex`

### 变量命名
- 变量: `snake_case`
- 常量: `UPPER_CASE`
- 类名: `PascalCase`
- 函数: `snake_case`

### 代码注释
- 文件头: 说明文件用途、作者、日期
- 函数: docstring (Args, Returns, Examples)
- 复杂逻辑: 行内注释

## Python 环境管理（uv 推荐方案）

本仓库推荐使用 **uv** 统一管理 Python 依赖和虚拟环境：

- 项目根目录: `SPEC/`
- 依赖定义文件: 根目录 `pyproject.toml`
- 虚拟环境目录: 根目录 `.venv/`（由 uv 自动创建）

基本使用方式：

```bash
# 在 SPEC/ 根目录下，同步依赖并创建 .venv/
uv sync

# 在 uv 管理的环境中运行脚本
uv run python scripts/run_all.py
uv run python 2025/src/your_module.py
```

注意：
- 不再推荐在项目内额外创建 `venv/` 或使用 `pip install -r requirements.txt` 作为主方案；
- 现有的 `requirements.txt` / pip 命令可以作为**回退选项**，但标准环境以 `pyproject.toml + uv` 为准；
- `.venv/` 目录应加入 `.gitignore`，避免提交到版本库。

## Git版本控制

### .gitignore 建议
```
# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/

# Jupyter
.ipynb_checkpoints/

# Data (large files)
data/raw/*.csv
data/external/*.csv

# Results (generated)
results/predictions/*.csv
results/logs/*.log

# LaTeX
*.aux
*.log
*.out
*.toc
*.bbl
*.blg
*.synctex.gz

# OS
.DS_Store
Thumbs.db
```

### 提交规范
```bash
git commit -m "feat: add ARIMA forecasting model"
git commit -m "fix: correct data preprocessing bug"
git commit -m "docs: update README with usage"
```

## 最佳实践

1. **数据不可变**: raw/目录数据只读
2. **可复现性**: 设置随机种子
3. **模块化**: 每个功能独立模块
4. **文档化**: README和注释完整
5. **版本控制**: 定期提交Git
6. **结果验证**: 交叉检查所有输出

## AI 智能体与目录结构（短指针示例）

- **.0modeler → 负责问题理解与建模大纲**  
  - 关注目录：`docs/`, `data/README.md`  
  - 示例短指针：`.0modeler -> update modeling_outline.md with assumptions and model list for all questions.`

- **.0mathcoder → 负责代码实现与数据管道**  
  - 关注目录：`src/`, `scripts/`, `data/processed/`, `results/`  
  - 示例短指针：`.0mathcoder -> implement preprocessing pipeline in src/preprocessing/ and expose entrypoints in scripts/run_q1.py.`

- **.0checker → 负责质量检查与一致性**  
  - 关注目录：`tests/`, `results/`, `figures/`, `paper/`  
  - 示例短指针：`.0checker -> verify results/tables/*.tex numbers match latest predictions and metrics; report any mismatch >1%.`

- **.0paperwriter → 负责论文撰写与LaTeX**  
  - 关注目录：`paper/`, `results/tables/`, `figures/`  
  - 示例短指针：`.0paperwriter -> draft Results section in LaTeX using tables from results/tables/ and figures from figures/.`

- **提示词生成者 / 项目检察长（当前AI）**  
  - 根据 `PROJECT_STRUCTURE.md`, `WORKFLOW.md`, 各 agent README 生成上述短指针。  
  - 示例短指针：`human -> upload all competition attachments into data/raw/ and confirm schema in data_dictionary.md.`
