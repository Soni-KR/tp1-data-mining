# EduGuard — short test report

Date: 9 October 2026.

## What works

- FastAPI starts; its home page, feature list and interactive documentation respond.
- Streamlit starts and all seven dashboard pages open.
- Dataset exploration and editable student inputs work.
- The app sends prediction requests to FastAPI and displays a clear error when the model is unavailable.
- Input validation, threshold rules and risk categories passed controlled tests.
- The existing model and notebooks were preserved.

## What is still blocked

**Real predictions and training do not currently work in this Windows environment.**

Windows Application Control blocks scikit-learn's compiled component. It also blocks PyArrow; the dashboard uses an HTML table fallback so tables still display. No security settings were changed.

The saved pipeline records scikit-learn 1.6.1 and XGBoost 3.4.1. The current Windows environment has scikit-learn 1.9.1. Fresh deployment requirements match the saved model.

## Test results

**7 tests passed; 4 failed.**

The failed tests require the blocked model/runtime:

1. Real saved-model prediction and comparison with direct inference.
2. The HTTP client receiving a real prediction.
3. Training on an alternative labeled dataset.
4. Training through Streamlit and showing its evaluation results.

Successful controlled tests use a test model to check the connection and decision rules. They do not prove the saved XGBoost model can run.

Live checks also confirmed that both servers respond and the dashboard handles an actual API prediction failure without crashing. Test servers were stopped afterward.

## How to check it yourself

1. Double-click **start_eduguard.bat**.
2. Open **http://localhost:8501**.
3. Click **Check API readiness**.
4. If ready, go to **Student prediction**, select a profile and click **Predict**.
5. If unavailable, real model testing remains blocked; seeing the dashboard is not sufficient.

You can also use **http://localhost:8000/docs** to test /health and /predict by clicking Try it out and Execute.

## Before submission

- Get real predictions working on an allowed compatible runtime and rerun the four outstanding checks.
- Publish the frontend and backend, then configure the frontend's API URL.
- Clearly state that Module 5 (SHAP/LIME) is deferred.

Docker and Linux setup are optional and were not fully tested. No online deployment has been performed. Historical accuracy results come from the existing notebook, not new local inference tests.
