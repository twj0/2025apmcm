# Project Development Document 项目开发笔记规范

## 📋 Purpose 目的

`project_document/` 文件夹用于存放**开发过程中的思考、决策、实验和笔记**。

这里的文档记录的是**开发过程**，而非最终交付物。

## 🆚 与spec/的区别

| 维度 | project_document/ | spec/ |
|------|-------------------|-------|
| **内容性质** | 开发笔记、实验记录、思考过程 | AI代理的工作产出和交接文档 |
| **写作对象** | 给人类看的，记录开发历程 | 给AI看的，定义工作规范 |
| **更新频率** | 持续更新，随时添加想法 | 关键节点更新，正式交接 |
| **格式要求** | 灵活，可以是草稿 | 严格，使用模板 |
| **文件命名** | 按主题命名 `topic_name.md` | 按角色和时间戳 `role_YYYYMMDDHHmmss.md` |

## 📝 文档规范

### 命名规则

```
project_document/
├── 2025-01-15_problem_analysis.md        # 按日期_主题
├── data_exploration_notes.md             # 按主题
├── model_selection_discussion.md         # 讨论类
├── experiment_arima_vs_prophet.md        # 实验类
├── debugging_convergence_issue.md        # 问题记录
└── lessons_learned.md                    # 总结类
```

### 文件结构模板

```markdown
# [文档标题]

**创建时间**: YYYY-MM-DD HH:MM
**最后更新**: YYYY-MM-DD HH:MM
**相关问题**: Q1 / Q2 / Q3 / Q4
**状态**: 🟢 进行中 / 🟡 待确认 / 🔴 已解决 / ⚪ 已归档

---

## 背景 / Context

[为什么需要这个文档？遇到了什么问题？]

## 目标 / Objective

[这次要搞清楚什么？要解决什么？]

## 思考过程 / Thought Process

### 方案 A
- **想法**: [描述]
- **优点**: [...]
- **缺点**: [...]
- **结论**: [...]

### 方案 B
- **想法**: [描述]
- **优点**: [...]
- **缺点**: [...]
- **结论**: [...]

## 实验结果 / Experiments (if applicable)

### Experiment 1: [标题]
```python
# 代码片段
```

**结果**:
- RMSE: 0.25
- R²: 0.78

**观察**:
[你看到了什么？]

## 决策 / Decision

**最终选择**: [...]
**理由**: [...]

## 待办事项 / TODO

- [ ] 任务1
- [ ] 任务2

## 参考资料 / References

- [链接或文献]

---

**Tags**: #modeling #data_analysis #experiment
```

## 📚 文档类型

### 1. 问题分析类 (Problem Analysis)

**目的**: 深入理解和分解问题
**文件名**: `YYYYMMDD_problem_analysis.md`

```markdown
# 2025年C题问题分析

## 问题原文摘录
[...]

## 关键信息提取
- 数据: [...]
- 时间范围: [...]
- 要求: [...]

## 四个问题分解

### Q1: [标题]
- **问题类型**: 时间序列预测
- **难度**: ⭐⭐⭐
- **依赖**: 无
- **初步想法**: [...]
```

### 2. 数据探索类 (Data Exploration)

**目的**: 记录数据分析发现
**文件名**: `data_exploration_notes.md`

```markdown
# 数据探索笔记

## Attachment 1: [数据描述]

### 基本统计
- 行数: 100
- 列数: 5
- 缺失值: 无

### 发现
1. 变量X呈现上升趋势
2. 变量Y有明显季节性
3. 2020年有异常值

### 可视化
![](../figures/eda/scatter_plot.png)
```

### 3. 模型选择讨论 (Model Selection)

**目的**: 记录为什么选择某个模型
**文件名**: `model_selection_qX.md`

```markdown
# Q1 模型选择讨论

## 候选模型

### 1. ARIMA
**理论**: [...]
**适用性**: ✅ 数据平稳
**复杂度**: 低
**预期性能**: 中等

### 2. Prophet
**理论**: [...]
**适用性**: ⚠️ 需要更多数据
**复杂度**: 低
**预期性能**: 高

### 3. LSTM
**理论**: [...]
**适用性**: ❌ 数据量不足
**复杂度**: 高
**预期性能**: 未知

## 最终决策
**选择**: ARIMA
**理由**: [...]
```

### 4. 实验记录 (Experiments)

**目的**: 记录参数调优和模型对比
**文件名**: `experiment_[description].md`

```markdown
# 实验: ARIMA参数选择

## 实验设置
- 数据: 2010-2020
- 验证集: 2018-2020
- 指标: RMSE, MAE

## 实验结果

| (p,d,q) | RMSE | MAE | 备注 |
|---------|------|-----|------|
| (1,1,1) | 0.25 | 0.18 | ✅ 最佳 |
| (2,1,1) | 0.27 | 0.20 | 过拟合 |
| (1,1,2) | 0.26 | 0.19 | 次优 |

## 结论
选择 (1,1,1)
```

### 5. 调试记录 (Debugging)

**目的**: 记录遇到的问题和解决方案
**文件名**: `debugging_[issue].md`

```markdown
# 调试: 模型不收敛

## 问题描述
ARIMA模型在第1000次迭代后仍未收敛

## 尝试的方案

### 方案1: 增加迭代次数
❌ 无效，仍不收敛

### 方案2: 调整初始值
❌ 无效

### 方案3: 数据标准化
✅ 有效！收敛于第50次迭代

## 根本原因
数据量级差异大（10^6 vs 10^2），导致梯度不稳定

## 解决方案
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
```

## 经验教训
数值稳定性很重要，先标准化再建模
```

### 6. 会议记录 (Meeting Notes)

**目的**: 记录团队讨论和决策
**文件名**: `meeting_YYYYMMDD_HHmm.md`

```markdown
# 团队会议记录

**时间**: 2025-01-15 14:00
**参与者**: modeler, mathcoder, checker
**议题**: Q2模型选择

## 讨论要点

### modeler提出
- 考虑使用多元回归
- 需要特征工程

### mathcoder反馈
- 担心计算时间
- 建议先试简单模型

### checker意见
- 优先保证可复现性
- 建议固定随机种子

## 决策
1. 先用线性回归baseline
2. 再尝试Random Forest
3. 截止时间: Hour 14

## 待办
- [ ] modeler: 更新建模大纲
- [ ] mathcoder: 实现baseline
- [ ] checker: 准备审查清单
```

### 7. 经验总结 (Lessons Learned)

**目的**: 记录可复用的经验
**文件名**: `lessons_learned.md`

```markdown
# 经验教训

## ✅ 做得好的

### 1. 提前做数据探索
**经验**: 花1小时做EDA，节省了5小时调试
**下次**: 继续保持

### 2. 使用Git频繁提交
**经验**: 每完成一个功能就提交，bug好追溯
**下次**: 提交信息写更清楚

## ❌ 需要改进的

### 1. 模型复杂度控制不好
**问题**: 一开始就用复杂模型，浪费时间
**教训**: 先简单后复杂
**下次**: 永远从baseline开始

### 2. 文档更新不及时
**问题**: 等到最后才写文档，很多细节忘了
**教训**: 随手记录
**下次**: 每完成一步就更新文档

## 💡 技巧分享

### 快速验证模型
```python
# 用10%数据快速测试
sample = data.sample(frac=0.1)
quick_test(sample)
```

### 自动保存中间结果
```python
import pickle
with open('checkpoint.pkl', 'wb') as f:
    pickle.dump(model, f)
```
```

## 🔍 文档检索

### 按标签查找

使用文档底部的Tags进行分类：

```markdown
**Tags**: #modeling #timeseries #arima #experiment
```

常用标签：
- `#problem_analysis` - 问题分析
- `#data_exploration` - 数据探索
- `#modeling` - 建模相关
- `#debugging` - 调试记录
- `#experiment` - 实验记录
- `#decision` - 重要决策
- `#lesson_learned` - 经验教训
- `#timeseries` - 时间序列
- `#regression` - 回归分析
- `#optimization` - 优化问题

### 按状态筛选

- 🟢 进行中 - 正在处理
- 🟡 待确认 - 需要验证
- 🔴 已解决 - 已有结论
- ⚪ 已归档 - 历史记录

## 📁 推荐的文件组织

```
project_document/
│
├── 00_overview/
│   ├── project_timeline.md           # 项目时间线
│   └── team_responsibilities.md      # 团队职责
│
├── 01_problem_analysis/
│   ├── 20250115_problem_reading.md
│   └── questions_breakdown.md
│
├── 02_data_exploration/
│   ├── attachment1_analysis.md
│   ├── attachment2_analysis.md
│   └── data_quality_issues.md
│
├── 03_modeling/
│   ├── q1_model_selection.md
│   ├── q2_model_selection.md
│   ├── q3_model_selection.md
│   └── q4_model_selection.md
│
├── 04_experiments/
│   ├── exp01_arima_tuning.md
│   ├── exp02_prophet_test.md
│   └── exp03_ensemble.md
│
├── 05_debugging/
│   ├── issue01_convergence.md
│   └── issue02_memory_error.md
│
├── 06_meetings/
│   ├── meeting_20250115_1400.md
│   └── meeting_20250116_0900.md
│
└── 07_summary/
    ├── lessons_learned.md
    └── best_practices.md
```

## ⚠️ 注意事项

### DO ✅

1. **随时记录**: 想法出现时立即记录，不要等
2. **诚实记录**: 失败的尝试也要记录，这是宝贵经验
3. **代码片段**: 关键代码直接粘贴，方便复用
4. **截图保存**: 重要的图表截图保存
5. **交叉引用**: 相关文档互相链接

### DON'T ❌

1. **不要过度整理**: 追求完美格式，重要的是内容
2. **不要只记成功**: 失败和弯路同样重要
3. **不要拖延**: 不要想着"等会再写"
4. **不要全删除**: 即使过时的文档也保留，加上 `⚪ 已归档`
5. **不要太正式**: 这是笔记，可以用口语化表达

## 🚀 快速开始

### 创建新文档

```bash
# 使用模板创建
cp project_document/templates/general_template.md \
   project_document/my_new_document.md
```

### 模板内容

```markdown
# [标题]

**时间**: 2025-01-15 14:00
**问题**: Q1
**状态**: 🟢 进行中

---

## 背景
[...]

## 目标
[...]

## 内容
[...]

## 结论
[...]

---

**Tags**: #tag1 #tag2
```

## 📊 文档生命周期

```
1. 🟢 创建
   ↓
2. 🟢 持续更新
   ↓
3. 🟡 等待确认 (如果需要)
   ↓
4. 🔴 得出结论
   ↓
5. ⚪ 归档
```

---

## 总结

**project_document/** = 你的思考实验室

- 记录过程，不是结果
- 写给自己看，不是别人
- 保持真实，不追求完美
- 持续更新，不一次写完

**记住**:
> "好记性不如烂笔头。三天后的你会感谢现在做笔记的你。" 📝

---

**Happy Documenting! 🎉**
