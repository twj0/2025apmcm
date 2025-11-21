# Model Comparison

```json
{
  "econometric_models": {
    "VAR": {
      "aic": null,
      "bic": null,
      "lag_order": null
    },
    "OLS_Regressions": {
      "num_models": 4,
      "avg_rsquared": 0.46532519302548503
    }
  },
  "ml_models": {
    "VAR_LSTM": {
      "rmse": 3.9952278117546266e-07,
      "training_samples": 5
    },
    "Reshoring_ML": {
      "random_forest": {
        "rmse": 0.15250573759697833,
        "mae": 0.14299999999999446,
        "r2": -8.30319999999943
      },
      "gradient_boosting": {
        "rmse": 0.1375118857702132,
        "mae": 0.1280996437468902,
        "r2": -6.56380749123212
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
