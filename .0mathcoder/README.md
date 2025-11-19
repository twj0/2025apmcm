# Mathematical Coder Agent 💻

## Role
Algorithm Implementation Specialist - Translates mathematical models into production-quality code.

## Quick Start

### Project Setup (15 min)
```bash
# Create structure
mkdir -p data/{raw,processed} models results figures
touch requirements.txt README.md

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install base packages
pip install numpy pandas scipy scikit-learn matplotlib seaborn
pip freeze > requirements.txt
```

## Code Templates

### 1. Data Preprocessing Template
```python
"""
data_preprocessing.py
Data loading, cleaning, and feature engineering
"""
import numpy as np
import pandas as pd
from typing import Tuple

# Set random seed for reproducibility
np.random.seed(42)

def load_data(filepath: str) -> pd.DataFrame:
    """Load raw data from file.

    Args:
        filepath: Path to data file

    Returns:
        DataFrame with raw data
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data: handle missing values, outliers.

    Args:
        df: Raw dataframe

    Returns:
        Cleaned dataframe
    """
    # Missing values
    print(f"Missing values:\n{df.isnull().sum()}")
    df = df.dropna()  # or fillna()

    # Outliers (example: IQR method)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features.

    Args:
        df: Cleaned dataframe

    Returns:
        DataFrame with engineered features
    """
    # Example: time-based features
    if 'date' in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year
        df['month'] = pd.to_datetime(df['date']).dt.month

    # Example: interaction terms
    # df['feature_interaction'] = df['feature1'] * df['feature2']

    return df

if __name__ == "__main__":
    # Pipeline
    df_raw = load_data('data/raw/data.csv')
    df_clean = clean_data(df_raw)
    df_final = engineer_features(df_clean)
    df_final.to_csv('data/processed/data_processed.csv', index=False)
    print("Preprocessing complete!")
```

### 2. Time Series Forecasting Template
```python
"""
model_timeseries.py
Time series forecasting with ARIMA
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error

np.random.seed(42)

class TimeSeriesForecaster:
    """ARIMA-based time series forecasting."""

    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        """Initialize forecaster.

        Args:
            order: (p, d, q) for ARIMA model
        """
        self.order = order
        self.model = None
        self.fitted = None

    def fit(self, data: pd.Series):
        """Fit ARIMA model.

        Args:
            data: Time series data
        """
        self.model = ARIMA(data, order=self.order)
        self.fitted = self.model.fit()
        print(self.fitted.summary())

    def predict(self, steps: int) -> np.ndarray:
        """Forecast future values.

        Args:
            steps: Number of steps ahead

        Returns:
            Array of predictions
        """
        forecast = self.fitted.forecast(steps=steps)
        return forecast

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate evaluation metrics.

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }

def plot_forecast(data: pd.Series, forecast: np.ndarray,
                  steps: int, save_path: str = None):
    """Plot historical data and forecast.

    Args:
        data: Historical time series
        forecast: Forecasted values
        steps: Number of forecast steps
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 6))

    # Historical data
    plt.plot(data.index, data.values, label='Historical', marker='o')

    # Forecast
    forecast_index = pd.date_range(start=data.index[-1], periods=steps+1, freq='Y')[1:]
    plt.plot(forecast_index, forecast, label='Forecast',
             marker='s', linestyle='--', color='red')

    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Time Series Forecast', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/processed/data_processed.csv', index_col='year')

    # Train/test split
    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]

    # Fit model
    forecaster = TimeSeriesForecaster(order=(1, 1, 1))
    forecaster.fit(train['value'])

    # Predict
    predictions = forecaster.predict(steps=len(test))

    # Evaluate
    metrics = forecaster.evaluate(test['value'].values, predictions)
    print(f"Metrics: {metrics}")

    # Forecast future
    future_forecast = forecaster.predict(steps=3)
    print(f"Future forecast: {future_forecast}")

    # Plot
    plot_forecast(df['value'], future_forecast, steps=3,
                  save_path='figures/forecast.pdf')
```

### 3. Regression Model Template
```python
"""
model_regression.py
Multiple regression with feature selection
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(42)

class RegressionModeler:
    """Multiple regression with model comparison."""

    def __init__(self):
        self.models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.scaler = StandardScaler()

    def fit_compare(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Fit all models and compare performance.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            DataFrame with model comparison
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        results = []
        for name, model in self.models.items():
            # Cross-validation
            scores = cross_val_score(model, X_scaled, y, cv=5,
                                     scoring='neg_mean_squared_error')
            rmse = np.sqrt(-scores.mean())

            # Fit on full data
            model.fit(X_scaled, y)

            results.append({
                'Model': name,
                'RMSE': rmse,
                'R²': model.score(X_scaled, y)
            })

        results_df = pd.DataFrame(results).sort_values('RMSE')
        print(results_df)

        # Select best model
        best_name = results_df.iloc[0]['Model']
        self.best_model = self.models[best_name]
        print(f"\nBest model: {best_name}")

        return results_df

    def feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Get feature importance (for tree-based models).

        Args:
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance
        """
        if hasattr(self.best_model, 'feature_importances_'):
            importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': self.best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            return importance
        else:
            print("Model does not have feature_importances_")
            return None

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/processed/data_processed.csv')

    # Prepare X, y
    feature_cols = ['feature1', 'feature2', 'feature3']
    X = df[feature_cols].values
    y = df['target'].values

    # Fit and compare
    modeler = RegressionModeler()
    comparison = modeler.fit_compare(X, y)

    # Feature importance
    importance = modeler.feature_importance(feature_cols)
    if importance is not None:
        print(f"\n{importance}")
```

### 4. Visualization Template
```python
"""
visualization.py
Publication-quality figures
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

def plot_comparison(data: pd.DataFrame, save_path: str = None):
    """Compare multiple models/scenarios.

    Args:
        data: DataFrame with columns ['Year', 'Model1', 'Model2', ...]
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for col in data.columns[1:]:
        ax.plot(data['Year'], data[col], marker='o', label=col, linewidth=2)

    ax.set_xlabel('Year')
    ax.set_ylabel('Value')
    ax.set_title('Model Comparison')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_sensitivity(param_values: np.ndarray, results: np.ndarray,
                     param_name: str, save_path: str = None):
    """Sensitivity analysis plot.

    Args:
        param_values: Array of parameter values
        results: Array of corresponding results
        param_name: Name of parameter
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(param_values, results, marker='o', linewidth=2, color='steelblue')
    ax.axhline(y=results[len(results)//2], color='red', linestyle='--',
               label='Baseline', alpha=0.7)

    ax.set_xlabel(param_name)
    ax.set_ylabel('Output')
    ax.set_title(f'Sensitivity Analysis: {param_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
```

## Best Practices Checklist

### Before Coding
- [ ] Read modeling specification from .0modeler
- [ ] Understand input/output requirements
- [ ] Set up project structure
- [ ] Create virtual environment

### During Coding
- [ ] Set random seeds (np.random.seed(42))
- [ ] Add type hints to functions
- [ ] Write docstrings (Args, Returns, Examples)
- [ ] Use meaningful variable names
- [ ] Add inline comments for complex logic

### After Coding
- [ ] Test with sample data
- [ ] Check for NaN/Inf values
- [ ] Verify results make sense
- [ ] Generate all required outputs
- [ ] Submit to .0checker for review

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Convergence failure | Adjust learning rate, increase iterations |
| Memory error | Use batch processing, reduce data size |
| Slow execution | Vectorize loops, use multiprocessing |
| NaN in results | Check for division by zero, log of negative |
| Poor predictions | Check data quality, try different model |

## Output Checklist

- [ ] `results/predictions.csv` - All predictions
- [ ] `results/metrics.json` - Evaluation metrics
- [ ] `results/latex_tables.tex` - LaTeX tables
- [ ] `figures/*.pdf` - All figures in vector format
- [ ] `requirements.txt` - All dependencies
- [ ] `README.md` - How to run code
