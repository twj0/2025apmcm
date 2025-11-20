# APMCM Competition Workflow

## Team Collaboration Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                    72-Hour Competition Timeline                  │
└─────────────────────────────────────────────────────────────────┘

Hour 0 ──────────────────────────────────────────────────────► Hour 72
│                                                                    │
├─ Phase 1: Analysis (0-2h)                                        │
├─ Phase 2: Data & EDA (2-8h)                                      │
├─ Phase 3: Modeling (8-20h)                                       │
├─ Phase 4: Results (20-30h)                                       │
├─ Phase 5: Writing (30-48h)                                       │
└─ Phase 6: Review (48-72h)                                        │
```

## Phase 1: Problem Analysis (Hour 0-2)

### Objective
Understand problem, decompose into sub-problems, create modeling strategy.

### Workflow
```
Problem Released
    │
    ├─► .0modeler (Lead)
    │   ├─ Read problem statement (15 min)
    │   ├─ Identify question types (15 min)
    │   ├─ Draft modeling outline (30 min)
    │   └─ Create handoff document (10 min)
    │
    ├─► .0checker (Support)
    │   ├─ Create project timeline
    │   ├─ Set up milestones
    │   └─ Initialize tracking
    │
    └─► Team Meeting (10 min)
        └─ Align on strategy
```

### Deliverables
- [ ] Modeling outline (Markdown with LaTeX)
- [ ] Problem decomposition
- [ ] Model selection rationale
- [ ] Project timeline

### Example Timeline
```
00:00 - Problem released
00:15 - Initial reading complete
00:30 - Problem decomposition done
01:00 - Modeling outline drafted
01:30 - Team alignment meeting
02:00 - Phase 1 complete ✓
```

---

## Phase 2: Data & Exploratory Analysis (Hour 2-8)

### Objective
Load data, clean, explore patterns, engineer features.

### Workflow
```
Modeling Outline Ready
    │
    ├─► .0mathcoder (Lead)
    │   ├─ Set up project structure (30 min)
    │   ├─ Load and validate data (1 hour)
    │   ├─ Data cleaning (1 hour)
    │   ├─ Exploratory analysis (2 hours)
    │   └─ Feature engineering (1.5 hours)
    │
    ├─► .0checker (Review)
    │   ├─ Review data loading code
    │   ├─ Validate data quality
    │   └─ Check preprocessing logic
    │
    └─► .0modeler (Consult)
        └─ Validate features align with models
```

### Deliverables
- [ ] Project structure created
- [ ] Data cleaned and validated
- [ ] EDA report with visualizations
- [ ] Engineered features documented

### Code Structure
```
project/
├── data/
│   ├── raw/              # Original data
│   └── processed/        # Cleaned data
├── models/               # Model implementations
├── results/              # Outputs
├── figures/              # Visualizations
├── requirements.txt
└── README.md
```

---

## Phase 3: Model Implementation (Hour 8-20)

### Objective
Implement models, validate, generate predictions.

### Workflow
```
Data Ready
    │
    ├─► .0mathcoder (Lead)
    │   ├─ Implement Model 1 (3 hours)
    │   │   ├─ Code implementation
    │   │   ├─ Unit testing
    │   │   └─ Initial run
    │   │
    │   ├─ Code Review Checkpoint ◄─── .0checker
    │   │
    │   ├─ Implement Model 2 (3 hours)
    │   ├─ Implement Model 3 (2 hours)
    │   │
    │   ├─ Model comparison (2 hours)
    │   └─ Generate predictions (2 hours)
    │
    ├─► .0checker (Review - 3 checkpoints)
    │   ├─ Review before execution
    │   ├─ Validate mathematical correctness
    │   ├─ Check reproducibility
    │   └─ Verify results sanity
    │
    └─► .0modeler (Validate)
        ├─ Check model assumptions
        ├─ Validate convergence
        └─ Interpret results
```

### Critical Checkpoints

#### Checkpoint 1: Before First Model Run (Hour 10)
```markdown
.0checker Review:
- [ ] Random seeds set
- [ ] No hardcoded paths
- [ ] Error handling present
- [ ] Mathematical formulas correct
```

#### Checkpoint 2: After Model 1 Complete (Hour 13)
```markdown
.0checker + .0modeler Review:
- [ ] Model converged
- [ ] Results in expected range
- [ ] Validation metrics acceptable
- [ ] Ready for Model 2
```

#### Checkpoint 3: All Models Complete (Hour 18)
```markdown
Full Team Review:
- [ ] All models executed successfully
- [ ] Results consistent across models
- [ ] Predictions generated
- [ ] Ready for paper writing
```

### Deliverables
- [ ] All model implementations
- [ ] Model comparison results
- [ ] Predictions for all questions
- [ ] Evaluation metrics
- [ ] Diagnostic plots

---

## Phase 4: Results & Validation (Hour 20-30)

### Objective
Generate final results, create visualizations, perform sensitivity analysis.

### Workflow
```
Models Complete
    │
    ├─► .0mathcoder (Lead)
    │   ├─ Generate all predictions (2 hours)
    │   ├─ Create publication figures (3 hours)
    │   ├─ Format tables for LaTeX (2 hours)
    │   ├─ Sensitivity analysis (2 hours)
    │   └─ Export results (1 hour)
    │
    ├─► .0checker (Validate)
    │   ├─ Cross-check all results
    │   ├─ Verify figure quality
    │   ├─ Validate table formats
    │   └─ Reproducibility test
    │
    └─► .0modeler (Interpret)
        ├─ Validate sensitivity analysis
        ├─ Interpret findings
        └─ Prepare result descriptions
```

### Deliverables
- [ ] `results/predictions.csv` - All predictions
- [ ] `results/metrics.json` - Evaluation metrics
- [ ] `results/latex_tables.tex` - LaTeX tables
- [ ] `figures/*.pdf` - All figures (vector format)
- [ ] Sensitivity analysis results

### Figure Checklist
- [ ] Data exploration plots
- [ ] Model comparison plots
- [ ] Prediction plots (with confidence intervals)
- [ ] Sensitivity analysis plots
- [ ] All figures 300 DPI, vector format
- [ ] Consistent style and fonts

---

## Phase 5: Paper Writing (Hour 30-48)

### Objective
Write complete APMCM paper in English with LaTeX.

### Workflow
```
Results Ready
    │
    ├─► .0paperwriter (Lead)
    │   ├─ Setup LaTeX project (1 hour)
    │   ├─ Write Introduction (2 hours)
    │   ├─ Write Problem Analysis (2 hours)
    │   ├─ Write Assumptions (1 hour)
    │   ├─ Write Model Development (4 hours)
    │   ├─ Write Results (3 hours)
    │   ├─ Write Sensitivity Analysis (2 hours)
    │   ├─ Write Strengths/Weaknesses (1 hour)
    │   ├─ Write Conclusions (1 hour)
    │   └─ Write Abstract (1 hour)
    │
    ├─► .0modeler (Provide Content)
    │   ├─ Model descriptions
    │   ├─ Mathematical formulations
    │   ├─ Assumptions and justifications
    │   └─ Results interpretation
    │
    ├─► .0mathcoder (Provide Materials)
    │   ├─ All figures
    │   ├─ All tables
    │   └─ Code for appendix
    │
    └─► .0checker (Review)
        ├─ Technical accuracy
        ├─ Consistency with code
        └─ Completeness check
```

### Writing Schedule
```
Hour 30-32:  Setup + Introduction
Hour 32-34:  Problem Analysis
Hour 34-35:  Assumptions
Hour 35-39:  Model Development (most important)
Hour 39-42:  Results
Hour 42-44:  Sensitivity Analysis
Hour 44-45:  Strengths/Weaknesses
Hour 45-46:  Conclusions
Hour 46-47:  Abstract (write last!)
Hour 47-48:  Format, compile, initial review
```

### Deliverables
- [ ] Complete LaTeX paper
- [ ] Compiled PDF
- [ ] All figures integrated
- [ ] All tables formatted
- [ ] Bibliography complete
- [ ] Appendices with code

---

## Phase 6: Review & Submission (Hour 48-72)

### Objective
Final review, revisions, quality assurance, submission.

### Workflow
```
Draft Complete
    │
    ├─► Full Team Review (Hour 48-54)
    │   ├─ .0checker: Technical review
    │   ├─ .0modeler: Mathematical review
    │   ├─ .0mathcoder: Code/results review
    │   └─ .0paperwriter: Language review
    │
    ├─► Revision Round 1 (Hour 54-60)
    │   ├─ Fix critical issues
    │   ├─ Improve clarity
    │   └─ Polish language
    │
    ├─► Revision Round 2 (Hour 60-66)
    │   ├─ Final improvements
    │   ├─ Proofread
    │   └─ Format check
    │
    ├─► Final Checks (Hour 66-70)
    │   ├─ Reproducibility test
    │   ├─ PDF quality check
    │   ├─ File size check
    │   └─ Submission package
    │
    └─► Submit (Hour 70-72)
        ├─ Create submission ZIP
        ├─ Upload to platform
        └─ Verify submission
```

### Final Checklist

#### Content Quality
- [ ] All questions answered completely
- [ ] Models clearly explained
- [ ] Results properly interpreted
- [ ] Assumptions justified
- [ ] Limitations discussed

#### Technical Quality
- [ ] Code runs from clean environment
- [ ] Results reproducible
- [ ] All figures visible in PDF
- [ ] All references resolved
- [ ] No LaTeX errors

#### Language Quality
- [ ] No grammatical errors
- [ ] Consistent terminology
- [ ] Formal academic tone
- [ ] Clear and concise

#### Format Compliance
- [ ] APMCM template used
- [ ] Page limit met
- [ ] File naming correct
- [ ] Submission package complete

---

## Communication Protocol

### Daily Standups (Every 6 Hours)
```
Hour 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66:

Each agent reports:
1. What I completed
2. What I'm working on
3. Any blockers
4. ETA for current task
```

### Handoff Protocol
```
When passing work between agents:

1. Notify recipient
2. Provide clear deliverables
3. Specify expected format
4. Set deadline
5. Confirm receipt
```

### Escalation Protocol
```
If blocked:

1. Identify blocker type
   - Technical issue
   - Missing information
   - Time constraint

2. Notify team immediately

3. Propose solutions

4. Get team decision

5. Document resolution
```

---

## Risk Management

### Common Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Model doesn't converge | Medium | High | Have backup simpler model |
| Data quality issues | Medium | High | Validate early, have cleaning plan |
| Time overrun | High | High | Strict time boxing, parallel work |
| Code bugs | Medium | Medium | Code review before execution |
| LaTeX compilation errors | Low | Medium | Use tested template, compile early |
| Language issues | Low | Low | Use standard phrases, proofread |

### Contingency Plans

#### If Model Fails (Hour 15)
```
1. Switch to simpler model immediately
2. Don't spend > 2 hours debugging
3. Document attempt in paper
4. Move forward with working model
```

#### If Behind Schedule (Hour 36)
```
1. Assess remaining work
2. Prioritize critical sections
3. Accept technical debt
4. Parallel work where possible
5. Reduce polish, focus on completeness
```

#### If Major Bug Found (Hour 50)
```
1. Assess impact on paper
2. If fixable in < 2 hours: fix
3. If not: document limitation
4. Don't rewrite entire paper
5. Focus on what works
```

---

## Success Metrics

### Phase Completion Criteria

#### Phase 1 ✓
- Modeling outline approved by team
- Timeline created
- Roles assigned

#### Phase 2 ✓
- Data loaded and validated
- EDA complete
- Features engineered

#### Phase 3 ✓
- All models implemented
- Code reviewed and approved
- Predictions generated

#### Phase 4 ✓
- All results validated
- Figures publication-ready
- Tables formatted

#### Phase 5 ✓
- Complete draft written
- All sections present
- PDF compiles

#### Phase 6 ✓
- All reviews complete
- Revisions done
- Submitted on time

---

## Tools & Resources

### Project Management
- Timeline tracker (spreadsheet)
- Task checklist (Markdown)
- Status reports (every 6 hours)

### Code Management
- Git for version control
- pyproject.toml + uv for dependencies and virtual environment
- README.md for documentation

### Document Management
- LaTeX for paper
- BibTeX for references
- Git for version control

### Communication
- Shared document for handoffs
- Status update template
- Issue tracking

---

## Post-Competition Review

After submission, conduct brief review:

### What Went Well
- [List successes]

### What Could Improve
- [List improvements]

### Lessons Learned
- [Key takeaways]

### Action Items for Next Time
- [Specific improvements]
