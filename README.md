# EduGuard

Predict student dropout risk at the end of the first semester.

**3,630 students · 30 first-semester features · Saved XGBoost model.**

## Run locally

Install Python 3.12 or 3.13, then run once in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Double-click **start_eduguard.bat**, then open [localhost:8501](http://localhost:8501).

Test without writing code: **Check API readiness → Student prediction → select a profile → Predict**. No retraining is needed.

API documentation: [localhost:8000/docs](http://localhost:8000/docs). Use **Try it out** to test a prediction in the browser.

## Project

`app.py` (Streamlit), `api.py` (FastAPI), and `ml_utils.py` (training/evaluation). The saved model uses `preprocessing.pkl` + `xgboost_model.json`; `reference.csv` supplies example profiles.

Deliverables: original `prj1-datamining.ipynb`, [one-page report](ONE_PAGER.pdf), and screenshots below.

Training data loads from [UCI](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success), or upload a labeled CSV. Enrolled students and second-semester variables are excluded; Dropout=0, Graduate=1.

Optional training compares five classifiers. Choose thresholds on validation data before checking the held-out test.

Saved XGBoost results from the notebook: **ROC-AUC .9404 · Dropout F1 .8717 · Precision .9390 · Recall .8134**.

Use predictions to support student follow-up. Risk bands: Low <30%, Medium 30–<70%, High ≥70%.

## Screenshots

![Dashboard](screenshots/overview.png)

![Prediction](screenshots/prediction.png)

## Test and deploy

Tests: `.\.venv\Scripts\python.exe -m pytest -q`.

Deploy FastAPI on a backend server and Streamlit on Community Cloud. Set **EDUGUARD_API_URL** to the backend's HTTPS URL. Online deployment is pending.
