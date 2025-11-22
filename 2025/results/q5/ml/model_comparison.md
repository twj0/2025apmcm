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
      "avg_rsquared": 0.32460446021534095
    }
  },
  "ml_models": {
    "VAR_LSTM": {
      "rmse": 0.001,
      "training_samples": 6
    },
    "Reshoring_ML": {
      "random_forest": {
        "rmse": 0.327230805395824,
        "mae": 0.3019999999999996,
        "r2": -9.708000000000037
      },
      "gradient_boosting": {
        "rmse": 0.38086959377235535,
        "mae": 0.32516397209836523,
        "r2": -13.506164746032002
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
