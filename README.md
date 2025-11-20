# 2025 APMCM Competition Project

## Project Overview

This repository contains the complete framework for the 2025 Asia-Pacific Mathematical Contest in Modeling (APMCM) Problem C, featuring four specialized AI agents working collaboratively.

## 🤖 AI Agent Team

### Core Agents

| Agent | Role | Primary Responsibilities |
|-------|------|-------------------------|
| **[.0modeler](/.0modeler/)** 🧮 | Mathematical Modeling Expert | Problem analysis, model selection, strategy design |
| **[.0mathcoder](/.0mathcoder/)** 💻 | Code Implementation Specialist | Algorithm implementation, data analysis, visualization |
| **[.0checker](/.0checker/)** ✅ | Quality Assurance Lead | Code review, validation, project coordination |
| **[.0paperwriter](/.0paperwriter/)** 📝 | Academic Writing Expert | LaTeX paper composition, English writing |

## 📁 Project Structure

```
.
├── .0modeler/              # Modeling expert agent
│   ├── modeler.agent.yaml  # Agent configuration
│   └── README.md           # Modeling guidelines
│
├── .0mathcoder/            # Code implementation agent
│   ├── mathcoder.agent.yaml
│   ├── README.md
│   └── template_main.py    # Main execution template
│
├── .0checker/              # Quality assurance agent
│   ├── checker.agent.yaml
│   └── README.md
│
├── .0paperwriter/          # Paper writing agent
│   ├── paperwriter.agent.yaml
│   ├── README.md
│   └── template_paper.tex  # LaTeX paper template
│
├── PreviousProblems/       # Historical competition problems
│   ├── 2023APMCM/
│   │   ├── problem/
│   │   └── latex-tmpl/
│   └── 2024APMCM/
│       ├── problem/
│       └── latex-tmpl/
│
├── prompt/                 # Project documentation
│   ├── 1.md               # Project overview
│   └── 2.md               # Detailed framework
│
├── WORKFLOW.md            # Complete 72-hour workflow
├── CLAUDE.md              # Claude Code integration guide
└── IFLOW.md               # Project workflow (Chinese)
```

## 🚀 Quick Start

### 1. Problem Released (Hour 0)

```bash
# .0modeler takes the lead
# Read problem statement
# Create modeling outline within 30 minutes
```

**Deliverable**: Modeling outline with problem decomposition and model selection

### 2. Data Analysis (Hour 2-8)

```bash
# .0mathcoder implements
cd project/
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run data preprocessing
python data_preprocessing.py
```

**Deliverable**: Cleaned data, EDA report, engineered features

### 3. Model Implementation (Hour 8-20)

```bash
# .0mathcoder codes, .0checker reviews
python model_timeseries.py
python model_regression.py
python main.py  # Run all models
```

**Deliverable**: Model implementations, predictions, evaluation metrics

### 4. Results Generation (Hour 20-30)

```bash
# Generate all outputs
python visualization.py
python main.py  # Generate final results
```

**Deliverable**: Figures (PDF), tables (LaTeX), sensitivity analysis

### 5. Paper Writing (Hour 30-48)

```bash
# .0paperwriter composes
cd paper/
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

**Deliverable**: Complete LaTeX paper with all sections

### 6. Final Review (Hour 48-72)

```bash
# Full team review
# .0checker validates everything
# Final revisions and submission
```

## 📋 72-Hour Timeline

| Phase | Hours | Lead Agent | Key Activities |
|-------|-------|------------|----------------|
| **Analysis** | 0-2 | .0modeler | Problem decomposition, modeling strategy |
| **Data & EDA** | 2-8 | .0mathcoder | Data cleaning, exploration, features |
| **Modeling** | 8-20 | .0mathcoder | Model implementation, validation |
| **Results** | 20-30 | .0mathcoder | Predictions, figures, sensitivity |
| **Writing** | 30-48 | .0paperwriter | LaTeX paper composition |
| **Review** | 48-72 | .0checker | Quality assurance, submission |

See [WORKFLOW.md](./WORKFLOW.md) for detailed timeline and collaboration protocol.

## 🛠️ Technology Stack

### Python (Primary)
- **Numerical**: NumPy, SciPy, Pandas
- **ML**: scikit-learn, XGBoost, TensorFlow/PyTorch
- **Stats**: statsmodels (ARIMA, regression)
- **Viz**: Matplotlib, Seaborn, Plotly
- **Optimization**: cvxpy, PuLP

### R (Secondary)
- forecast, ggplot2, dplyr, caret

### LaTeX
- APMCM template (see PreviousProblems/)
- Packages: amsmath, graphicx, booktabs, algorithm2e

## 📚 Documentation

### Agent Guides
- [.0modeler README](/.0modeler/README.md) - Modeling strategies and patterns
- [.0mathcoder README](/.0mathcoder/README.md) - Code templates and best practices
- [.0checker README](/.0checker/README.md) - Review checklists and protocols
- [.0paperwriter README](/.0paperwriter/README.md) - Writing guidelines and LaTeX tips

### Workflow
- [WORKFLOW.md](./WORKFLOW.md) - Complete 72-hour collaboration workflow
- [CLAUDE.md](./CLAUDE.md) - Claude Code integration guide
- [IFLOW.md](./IFLOW.md) - Project overview (Chinese)

### Historical Problems
- [2023 APMCM Problem C](./PreviousProblems/2023APMCM/problem/) - Electric Vehicles
- [2024 APMCM Problem C](./PreviousProblems/2024APMCM/problem/) - Pet Industry

## 🎯 Competition Requirements

### Paper Requirements
- **Language**: English
- **Format**: LaTeX (APMCM template)
- **Length**: Typically 20-30 pages
- **Sections**: Abstract, Introduction, Problem Analysis, Assumptions, Model Development, Results, Sensitivity Analysis, Strengths/Weaknesses, Conclusions

### Submission Package
```
submission.zip
├── paper.pdf           # Main paper
├── paper.tex          # LaTeX source
├── figures/           # All figures
├── code/              # All code
└── data/              # Processed data
```

## 🔍 Quality Standards

### Code Quality (.0checker)
- ✅ Random seeds set (reproducibility)
- ✅ No hardcoded paths
- ✅ Comprehensive error handling
- ✅ Type hints and docstrings
- ✅ PEP 8 compliance

### Mathematical Correctness (.0modeler)
- ✅ Formulas match specifications
- ✅ Numerical stability verified
- ✅ Convergence achieved
- ✅ Results in expected range

### Paper Quality (.0paperwriter)
- ✅ All sections complete
- ✅ Formal academic English
- ✅ LaTeX compiles without errors
- ✅ All figures/tables referenced
- ✅ Bibliography complete

## 🚨 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Model doesn't converge | Try simpler model, adjust hyperparameters |
| Data quality problems | Validate early, implement robust cleaning |
| Time overrun | Strict time boxing, parallel work |
| LaTeX errors | Use tested template, compile early |
| Code bugs | Code review before execution |

## 📞 Agent Communication

### Handoff Protocol
1. Notify recipient
2. Provide clear deliverables
3. Specify format
4. Set deadline
5. Confirm receipt

### Status Updates (Every 6 Hours)
Each agent reports:
- What I completed
- What I'm working on
- Any blockers
- ETA for current task

## 🏆 Success Criteria

### Phase Completion
- [ ] Phase 1: Modeling outline approved
- [ ] Phase 2: Data cleaned and explored
- [ ] Phase 3: Models implemented and validated
- [ ] Phase 4: Results generated and visualized
- [ ] Phase 5: Paper written and formatted
- [ ] Phase 6: Reviewed and submitted

### Final Checklist
- [ ] All questions answered
- [ ] Code runs from clean environment
- [ ] Results reproducible
- [ ] Paper compiles to PDF
- [ ] Submission package complete
- [ ] Submitted before deadline

## 📖 References

### APMCM Resources
- Official website: [APMCM](http://www.apmcm.org/)
- Historical problems: See `PreviousProblems/`
- LaTeX templates: See `PreviousProblems/*/latex-tmpl/`

### Modeling Resources
- Time Series: Box-Jenkins methodology, ARIMA
- Regression: Multiple regression, regularization
- Optimization: Linear programming, genetic algorithms
- Validation: Cross-validation, sensitivity analysis

## 📄 License

This project framework is for educational purposes in the APMCM competition.

## 🙏 Acknowledgments

This framework integrates best practices from:
- BMAD-METHOD (AI-driven agile development)
- OpenSpec (Spec-driven development)
- PromptX (AI agent context platform)
- spec-kit (Spec-driven toolkit)

---

**Ready for Competition!** 🎯

Follow the [WORKFLOW.md](./WORKFLOW.md) for step-by-step execution.
