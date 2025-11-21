# Model Comparison

```json
{
  "econometric_models": {
    "VAR": {
      "aic": -Infinity,
      "bic": -Infinity,
      "lag_order": 2
    },
    "OLS_Regressions": {
      "num_models": 4,
      "avg_rsquared": 0.24417356222019235
    }
  },
  "ml_models": {
    "VAR_LSTM": {
      "rmse": 4.2124945747785007e-07,
      "training_samples": 6
    },
    "Reshoring_ML": {
      "random_forest": {
        "rmse": 0.33090482015225947,
        "mae": 0.30299999999999905,
        "r2": -9.949799999999996
      },
      "gradient_boosting": {
        "rmse": 0.3841010118453058,
        "mae": 0.33228799006854626,
        "r2": -13.75335873005888
      }
    }
  },
  "summary": {
    "total_econometric_models": 2,
    "total_ml_models": 2,
    "hybrid_approach": "VAR + LSTM for impulse responses, RF/GB for reshoring prediction",
    "recommendation": "Use VAR for interpretability, ML for prediction accuracy"
  }
}
```
