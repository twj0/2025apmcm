"""
main.py - Main execution script for APMCM Competition
Orchestrates all models and generates final results

Author: .0mathcoder
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Import custom modules
from data_preprocessing import load_data, clean_data, engineer_features
from model_timeseries import TimeSeriesForecaster
from model_regression import RegressionModeler
from visualization import plot_comparison, plot_sensitivity


def setup_project():
    """Create project directory structure."""
    dirs = ['data/raw', 'data/processed', 'models', 'results', 'figures']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✓ Project structure created")


def question_1():
    """
    Question 1: [Problem statement]
    Approach: [Brief description]
    """
    print("\n" + "="*60)
    print("QUESTION 1")
    print("="*60)

    # Load data
    df = pd.read_csv('data/processed/data_q1.csv')
    print(f"Loaded {len(df)} rows")

    # Fit model
    forecaster = TimeSeriesForecaster(order=(1, 1, 1))
    forecaster.fit(df['value'])

    # Predict
    predictions = forecaster.predict(steps=3)
    print(f"Predictions: {predictions}")

    # Save results
    results = pd.DataFrame({
        'Year': [2026, 2027, 2028],
        'Prediction': predictions
    })
    results.to_csv('results/q1_predictions.csv', index=False)

    # Visualize
    plt.figure(figsize=(10, 6))
    plt.plot(df['year'], df['value'], 'o-', label='Historical')
    plt.plot([2026, 2027, 2028], predictions, 's--', label='Forecast')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title('Question 1: Forecast')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/q1_forecast.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Question 1 complete")
    return results


def question_2():
    """
    Question 2: [Problem statement]
    Approach: [Brief description]
    """
    print("\n" + "="*60)
    print("QUESTION 2")
    print("="*60)

    # Load data
    df = pd.read_csv('data/processed/data_q2.csv')

    # Prepare features
    X = df[['feature1', 'feature2', 'feature3']].values
    y = df['target'].values

    # Fit model
    modeler = RegressionModeler()
    comparison = modeler.fit_compare(X, y)

    # Save results
    comparison.to_csv('results/q2_model_comparison.csv', index=False)

    # Feature importance
    importance = modeler.feature_importance(['feature1', 'feature2', 'feature3'])
    if importance is not None:
        importance.to_csv('results/q2_feature_importance.csv', index=False)

    print("✓ Question 2 complete")
    return comparison


def sensitivity_analysis():
    """
    Perform sensitivity analysis on key parameters.
    """
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS")
    print("="*60)

    # Example: vary parameter from -50% to +50%
    param_values = np.linspace(0.5, 1.5, 21)
    results = []

    for param in param_values:
        # Run model with modified parameter
        # result = run_model_with_param(param)
        result = param ** 2  # Placeholder
        results.append(result)

    results = np.array(results)

    # Save
    sensitivity_df = pd.DataFrame({
        'Parameter': param_values,
        'Output': results
    })
    sensitivity_df.to_csv('results/sensitivity_analysis.csv', index=False)

    # Plot
    plot_sensitivity(param_values, results, 'Parameter α',
                     save_path='figures/sensitivity_analysis.pdf')

    print("✓ Sensitivity analysis complete")
    return sensitivity_df


def generate_latex_tables():
    """
    Generate LaTeX-formatted tables for paper.
    """
    print("\n" + "="*60)
    print("GENERATING LATEX TABLES")
    print("="*60)

    # Example: Question 1 results
    df = pd.read_csv('results/q1_predictions.csv')

    latex_code = r"""\begin{table}[htbp]
    \centering
    \caption{Forecasting Results for Question 1}
    \label{tab:q1_results}
    \begin{tabular}{lr}
        \toprule
        Year & Predicted Value \\
        \midrule
"""

    for _, row in df.iterrows():
        latex_code += f"        {int(row['Year'])} & {row['Prediction']:.2f} \\\\\n"

    latex_code += r"""        \bottomrule
    \end{tabular}
\end{table}
"""

    with open('results/latex_tables.tex', 'w') as f:
        f.write(latex_code)

    print("✓ LaTeX tables generated")


def generate_summary():
    """
    Generate summary of all results.
    """
    print("\n" + "="*60)
    print("GENERATING SUMMARY")
    print("="*60)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'random_seed': RANDOM_SEED,
        'questions': {
            'q1': {
                'model': 'ARIMA(1,1,1)',
                'predictions_file': 'results/q1_predictions.csv',
                'figure': 'figures/q1_forecast.pdf'
            },
            'q2': {
                'model': 'Multiple Regression',
                'results_file': 'results/q2_model_comparison.csv',
                'figure': 'figures/q2_comparison.pdf'
            }
        },
        'sensitivity_analysis': {
            'file': 'results/sensitivity_analysis.csv',
            'figure': 'figures/sensitivity_analysis.pdf'
        }
    }

    with open('results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("✓ Summary generated")
    print(f"\nAll results saved to results/")
    print(f"All figures saved to figures/")


def main():
    """
    Main execution function.
    """
    print("="*60)
    print("APMCM COMPETITION - MAIN EXECUTION")
    print("="*60)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Start time: {datetime.now()}")

    # Setup
    setup_project()

    # Execute questions
    try:
        q1_results = question_1()
        q2_results = question_2()
        sensitivity_results = sensitivity_analysis()

        # Generate outputs
        generate_latex_tables()
        generate_summary()

        print("\n" + "="*60)
        print("ALL TASKS COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"End time: {datetime.now()}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
