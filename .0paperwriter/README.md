# Paper Writer Agent 📝

## Role
Academic Paper Writing Specialist - Transforms technical results into professional APMCM papers.

## Quick Start

### Paper Writing Timeline (20 hours)
```
Hour 0-2:   Collect materials, setup LaTeX
Hour 2-4:   Introduction & Problem Analysis
Hour 4-8:   Model Development section
Hour 8-11:  Results & Sensitivity Analysis
Hour 11-14: Other sections (Assumptions, Strengths/Weaknesses)
Hour 14-17: Formatting, figures, tables
Hour 17-19: Abstract, polish, proofread
Hour 19-20: Final review, compile PDF
```

## APMCM Paper Structure

### Required Sections (8 sections)
```
1. Title & Team Information
2. Abstract (200-300 words)
3. Keywords (3-5 terms)
4. Introduction
5. Problem Analysis
6. Assumptions and Justifications
7. Model Development
8. Model Solution and Results
9. Sensitivity Analysis
10. Strengths and Weaknesses
11. Conclusions
12. References
13. Appendices
```

## LaTeX Template

### Document Setup
```latex
\documentclass[12pt,a4paper]{article}

% Essential packages
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{algorithm2e}
\usepackage{hyperref}
\usepackage[margin=2.5cm]{geometry}

% Custom commands
\newcommand{\R}{\mathbb{R}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Var}{\text{Var}}

\title{Development Analysis and Forecasting of [Topic]}
\author{Team \#[Number]}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
[200-300 words summarizing problem, approach, results, conclusions]
\end{abstract}

\textbf{Keywords:} [keyword1], [keyword2], [keyword3]

\section{Introduction}
[Content]

\section{Problem Analysis}
[Content]

% ... more sections ...

\bibliographystyle{plain}
\bibliography{references}

\appendix
\section{Code}
[Code listings]

\end{document}
```

## Section Templates

### 1. Abstract Template
```latex
\begin{abstract}
% Background (1-2 sentences)
[Topic] has become increasingly important due to [reason].
Understanding [specific aspect] is crucial for [stakeholders].

% Problem (1 sentence)
This paper addresses the problem of [specific problem statement].

% Approach (2-3 sentences)
We develop a comprehensive framework combining [method1], [method2],
and [method3]. Our approach involves [key steps]. We validate our
models using [validation method].

% Results (2-3 sentences)
Our analysis reveals that [key finding 1]. The forecasting results
indicate [key finding 2]. Sensitivity analysis demonstrates [key finding 3].

% Conclusion (1 sentence)
These findings provide valuable insights for [stakeholders] and suggest
that [main conclusion].
\end{abstract}

\textbf{Keywords:} time series forecasting, regression analysis,
sensitivity analysis, [domain-specific term]
```

### 2. Introduction Template
```latex
\section{Introduction}

\subsection{Background}
In recent years, [topic] has attracted considerable attention from
researchers and policymakers. According to [source], [statistic or fact].
This trend is driven by [factors].

\subsection{Problem Statement}
This paper addresses the following questions:
\begin{enumerate}
    \item [Question 1 from problem statement]
    \item [Question 2 from problem statement]
    \item [Question 3 from problem statement]
\end{enumerate}

\subsection{Our Approach}
To address these challenges, we develop a multi-faceted approach:
\begin{itemize}
    \item \textbf{Data Analysis:} We analyze [data description]
          spanning [time period].
    \item \textbf{Modeling:} We employ [model1] for [purpose1] and
          [model2] for [purpose2].
    \item \textbf{Validation:} We validate our models using [method].
\end{itemize}

\subsection{Paper Organization}
The remainder of this paper is organized as follows. Section 2 presents
our problem analysis. Section 3 lists our assumptions. Section 4 develops
our mathematical models. Section 5 presents results. Section 6 conducts
sensitivity analysis. Section 7 discusses strengths and weaknesses.
Section 8 concludes.
```

### 3. Model Development Template
```latex
\section{Model Development}

\subsection{Model 1: [Model Name]}

\subsubsection{Model Formulation}
We formulate the problem as follows. Let $y_t$ denote [variable] at
time $t$, where $t = 1, 2, \ldots, T$. Our objective is to [objective].

The model is defined as:
\begin{equation}
    y_t = f(x_t; \theta) + \epsilon_t
    \label{eq:main_model}
\end{equation}
where $x_t \in \R^p$ represents [features], $\theta$ denotes [parameters],
and $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$ is the error term.

\subsubsection{Parameter Estimation}
We estimate parameters using [method]. Specifically, we minimize:
\begin{equation}
    \hat{\theta} = \arg\min_{\theta} \sum_{t=1}^{T} (y_t - f(x_t; \theta))^2
    \label{eq:estimation}
\end{equation}

\subsubsection{Model Justification}
We choose this model because:
\begin{enumerate}
    \item [Reason 1 with theoretical support]
    \item [Reason 2 with empirical evidence]
    \item [Reason 3 with practical consideration]
\end{enumerate}
```

### 4. Results Template
```latex
\section{Model Solution and Results}

\subsection{Question 1: [Question]}

\subsubsection{Data Analysis}
Figure~\ref{fig:data_q1} shows the historical trend of [variable].
We observe that [observation 1] and [observation 2].

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/data_analysis_q1.pdf}
    \caption{Historical trend of [variable] from [year1] to [year2].
             The data shows [key pattern].}
    \label{fig:data_q1}
\end{figure}

\subsubsection{Model Results}
Table~\ref{tab:results_q1} presents the forecasting results. Our model
predicts that [variable] will [trend] from [value1] in [year1] to
[value2] in [year2], representing a [percentage] change.

\begin{table}[htbp]
    \centering
    \caption{Forecasting results for Question 1}
    \label{tab:results_q1}
    \begin{tabular}{lrrr}
        \toprule
        Year & Actual & Predicted & Error (\%) \\
        \midrule
        2023 & 1000 & 1020 & 2.0 \\
        2024 & 1100 & 1095 & -0.5 \\
        2025 & -- & 1180 & -- \\
        \bottomrule
    \end{tabular}
\end{table}

\subsubsection{Model Validation}
We validate our model using [method]. The RMSE is [value], MAE is [value],
and $R^2$ is [value], indicating [interpretation].
```

### 5. Sensitivity Analysis Template
```latex
\section{Sensitivity Analysis}

To assess the robustness of our model, we conduct sensitivity analysis
on key parameters.

\subsection{Parameter: [Parameter Name]}
We vary [parameter] from [min] to [max] while holding other parameters
constant. Figure~\ref{fig:sensitivity_param1} shows the results.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.7\textwidth]{figures/sensitivity_param1.pdf}
    \caption{Sensitivity analysis for [parameter]. The output is
             [stable/sensitive] to changes in this parameter.}
    \label{fig:sensitivity_param1}
\end{figure}

Our analysis reveals that:
\begin{itemize}
    \item When [parameter] increases by 10\%, [output] changes by [X]\%.
    \item The model is [most/least] sensitive to [parameter].
    \item The relationship is [linear/nonlinear].
\end{itemize}
```

## Writing Guidelines

### Academic English

#### Avoid ❌
- Contractions: "don't" → "do not"
- Informal: "a lot of" → "many", "numerous"
- Vague: "very good" → "excellent", "superior"
- Personal: "I think" → "The analysis suggests"

#### Prefer ✅
- Formal vocabulary
- Precise technical terms
- Quantitative statements
- Passive voice (when appropriate)

### Common Phrases

#### Introducing
- "In recent years, ... has attracted considerable attention"
- "This paper addresses the problem of ..."
- "We propose a comprehensive approach that ..."

#### Describing Methods
- "To address this challenge, we develop ..."
- "The model is formulated as follows:"
- "We employ ... to estimate ..."

#### Presenting Results
- "Figure X illustrates ..."
- "As shown in Table Y, ..."
- "The results demonstrate that ..."
- "Our analysis reveals ..."

#### Discussing
- "These findings suggest that ..."
- "One possible explanation is ..."
- "This is consistent with ..."
- "However, it should be noted that ..."

#### Concluding
- "In summary, this paper ..."
- "Our results indicate that ..."
- "Future work could explore ..."

## LaTeX Tips & Tricks

### Equations
```latex
% Inline math
The variable $x$ represents ...

% Numbered equation
\begin{equation}
    y = ax + b
    \label{eq:linear}
\end{equation}

% Multi-line equations
\begin{align}
    y_1 &= a_1 x + b_1 \label{eq:line1} \\
    y_2 &= a_2 x + b_2 \label{eq:line2}
\end{align}

% Reference equation
As shown in Equation~\eqref{eq:linear}, ...
```

### Figures
```latex
% Single figure
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/plot.pdf}
    \caption{Caption text.}
    \label{fig:plot}
\end{figure}

% Side-by-side figures
\begin{figure}[htbp]
    \centering
    \begin{subfigure}{0.45\textwidth}
        \includegraphics[width=\textwidth]{fig1.pdf}
        \caption{First figure}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.45\textwidth}
        \includegraphics[width=\textwidth]{fig2.pdf}
        \caption{Second figure}
    \end{subfigure}
    \caption{Overall caption}
    \label{fig:combined}
\end{figure}
```

### Tables
```latex
% Use booktabs for professional tables
\begin{table}[htbp]
    \centering
    \caption{Table caption}
    \label{tab:results}
    \begin{tabular}{lrrr}
        \toprule
        Item & Value 1 & Value 2 & Value 3 \\
        \midrule
        Row 1 & 10.5 & 20.3 & 30.1 \\
        Row 2 & 15.2 & 25.8 & 35.4 \\
        \bottomrule
    \end{tabular}
\end{table}
```

### Algorithms
```latex
\begin{algorithm}[htbp]
    \caption{Algorithm Name}
    \label{alg:name}
    \KwIn{Input parameters}
    \KwOut{Output results}

    Initialize $\theta \leftarrow \theta_0$\;
    \For{$t = 1$ to $T$}{
        Compute $\nabla L(\theta)$\;
        Update $\theta \leftarrow \theta - \alpha \nabla L(\theta)$\;
        \If{converged}{
            \textbf{break}\;
        }
    }
    \Return $\theta$\;
\end{algorithm}
```

## Quality Checklist

### Content ✓
- [ ] Abstract summarizes entire paper (200-300 words)
- [ ] Introduction provides context and motivation
- [ ] All assumptions stated and justified
- [ ] Models clearly formulated with equations
- [ ] Results properly presented with figures/tables
- [ ] Sensitivity analysis included
- [ ] Strengths and weaknesses discussed honestly
- [ ] Conclusions match results

### Formatting ✓
- [ ] APMCM template applied
- [ ] All equations numbered and referenced
- [ ] All figures have captions and labels
- [ ] All tables formatted with booktabs
- [ ] Bibliography complete
- [ ] Page numbers present
- [ ] No LaTeX errors or warnings

### Language ✓
- [ ] No grammatical errors
- [ ] Consistent terminology
- [ ] Formal academic tone
- [ ] Clear and concise sentences
- [ ] Smooth transitions

### Technical ✓
- [ ] Mathematical notation consistent
- [ ] Variable definitions clear
- [ ] Equations properly formatted
- [ ] Results match code output
- [ ] Figures readable (300 DPI)

## Common LaTeX Errors

| Error | Cause | Solution |
|-------|-------|----------|
| Undefined reference | Label not found | Compile twice |
| Missing $ | Math mode not closed | Add closing $ |
| Overfull hbox | Line too long | Rephrase or use `\linebreak` |
| Figure not found | Wrong path | Check file path and extension |
| Bibliography empty | .bib file issue | Run bibtex, then pdflatex twice |

## Collaboration

### With .0modeler
- Get model descriptions and formulations
- Clarify mathematical notation
- Understand assumptions and limitations

### With .0mathcoder
- Collect all figures (PDF format)
- Get LaTeX-formatted tables
- Verify numerical values

### With .0checker
- Submit draft for technical review
- Verify consistency with code
- Confirm all claims supported

## Final Steps

```bash
# Compile LaTeX
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex

# Check PDF
# - All figures visible
# - All references resolved
# - Page count within limit
# - File size reasonable

# Create submission package
zip submission.zip paper.pdf paper.tex figures/* code/*
```
