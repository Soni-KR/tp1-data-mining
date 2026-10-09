# EduGuard

ENSI Data Mining project: estimate student dropout risk **at the end of semester one**. Streamlit is the dashboard; FastAPI runs the existing trained model through HTTP.

## Try it locally

In this project folder, double-click **start_eduguard.bat** and keep the two server windows open.

1. Open [the dashboard](http://localhost:8501).
2. Click **Check API readiness**. It should say **Model is ready**.
3. Open **Student prediction**, select reference row **0**, and click **Predict**.
4. Expected example: **78.1% dropout probability**, **21.9% graduation probability**, **High risk**.
5. Try another profile or edit a fictional student.

No retraining is needed. Localhost means your own computer; it is not an online deployment. Avoid starting the launcher again while both servers are already running.

**Browser API test:** open [FastAPI docs](http://localhost:8000/docs), expand **POST /predict**, click **Try it out**, keep the supplied example, then **Execute**.

On a fresh clone, install Python 3.12 or 3.13 and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

To start manually, use two terminals:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --port 8000
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## What the dashboard does

| Page | Purpose |
|---|---|
| Overview | Objective, dataset summary and historical results. |
| Data exploration | Automatic data loading or CSV upload, statistics and interactive charts. |
| Preprocessing | Target/features, missing values, one-hot encoding, scaling and split settings. |
| Model comparison | Select algorithms and hyperparameters; train explicitly; optional GridSearchCV. |
| Evaluation | ROC/PR curves, confusion matrix, metrics and threshold controls. |
| Student prediction | Existing or fictional profile; real HTTP prediction through FastAPI. |
| About | Scope and limitations. |

Training only runs when you click **Train and compare**. Save preprocessing settings before training. Experimental models stay in session memory and do not replace the saved prediction model.

## Data and model

Source: [UCI dataset 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) / [Kaggle](https://www.kaggle.com/datasets/mattop/predict-students-dropout-and-academic-success).

- Original: 4,424 students, 36 features, three outcomes.
- Remove Enrolled because their final outcome is unknown: **3,630 students**, including 1,421 Dropout and 2,209 Graduate.
- Remove all six semester-two variables because they are unavailable at prediction time: **30 inputs**.
- Target mapping stays **Dropout=0, Graduate=1**.
- Stratified 80/20 split; StandardScaler, OneHotEncoder and binary indicators within pipelines. Preprocessing is fitted only on training folds.
- Five models: Logistic Regression, Decision Tree, KNN, Random Forest and XGBoost; five-fold GridSearchCV when requested.

Historical notebook results, not new measurements:

| Model | CV ROC-AUC | Test ROC-AUC | Dropout F1 |
|---|---:|---:|---:|
| Logistic Regression | .9348 | .9406 | .8587 |
| Decision Tree | .8924 | .9038 | .8302 |
| KNN | .8775 | .8762 | .6591 |
| Random Forest | .9303 | .9394 | .8387 |
| XGBoost | .9351 | .9404 | .8717 |

XGBoost was retained for its dropout F1. Historical dropout precision: .9390; recall: .8134.

## Understand predictions

A probability is an estimate, not a confirmed outcome. **Dropout** is predicted when dropout probability is at least the threshold (default **0.50**).

Risk categories: **Low <30%**, **Medium 30–<70%**, **High >=70%**. These application-defined bands are not statistically validated and do not change with the threshold.

Choose thresholds using training validation results before inspecting the held-out test. After GridSearchCV, displayed out-of-fold estimates are exploratory because parameters were selected on the same training data. Historical selection also consulted test results; external validation is still needed. Probabilities are not established as calibrated. Sex, age, nationality and family-status variables require fairness review. Use the score for supportive follow-up, never automatic exclusion or penalties.

## API

| Endpoint | Result |
|---|---|
| GET / | Service information. |
| GET /health | 200 only after a real sample prediction succeeds; 503 if unavailable. |
| GET /features | The 30 expected names. |
| POST /predict | Accepts data and optional threshold; returns both probabilities, prediction, risk level and threshold. |
| /docs | Interactive documentation with a valid student example. |

Missing/extra features or invalid thresholds return **422**. Model failure returns **503**. The frontend calls FastAPI through requests; it does not load the saved model for predictions.

## Files to know

- **app.py**, **api.py**, **api_client.py**, **ml_utils.py**: dashboard, service, HTTP connection and optional experiments.
- **prj1-datamining.ipynb** and **model.pkl**: original notebook and trained model, preserved.
- **preprocessing.pkl** and **xgboost_model.json**: portable copies of the existing fitted preprocessing and 200 trained trees, used by FastAPI; no retraining.
- **features.pkl** and **reference.csv**: feature names and example profiles. reference.csv has no labels and cannot measure accuracy.
- **data.csv**: labeled data. If absent, the dashboard downloads the public dataset automatically.
- **requirements.txt**, **start_eduguard.bat**, **Dockerfile**, **tests/**: installation, local launch, optional API container and verification.

Windows imports were fixed using compatible package versions. The original XGBoost runtime snapshot would not load; its stable trained-model section was exported to JSON. The portable copy matched the recovered pipeline exactly on 100 profiles.

## Assignment deliverables

| Required by the PDF | Current status |
|---|---|
| Deployed app covering six modules | Local app works. Online deployment and Module 5 remain unfinished. |
| Documented FastAPI called by the app | Prediction API works. SHAP/LIME interpretation endpoints are deferred. |
| GitHub: code, notebook, requirements, README, links and screenshots | Included in [this repository](https://github.com/Soni-KR/tp1-data-mining). |
| One-page PDF: problem, data, model, results, explanation, recommendations and limits | [ONE_PAGER.pdf](ONE_PAGER.pdf); missing XAI is explicitly disclosed. |

**SHAP/LIME is deliberately deferred. The original six-module assignment is not yet fully satisfied.** No causal explanations are fabricated.

## Screenshots

![Overview](screenshots/overview.png)

![Real student prediction](screenshots/prediction.png)

[API documentation screenshot](screenshots/api_docs.png)

## Tests and deployment

Run `.\.venv\Scripts\python.exe -m pytest -q`. Latest result: **11 passed**. See [TEST_RESULTS.md](TEST_RESULTS.md).

For deployment, host FastAPI on a compatible backend service and app.py on Streamlit Community Cloud. Set **EDUGUARD_API_URL** in Streamlit secrets or the environment to the backend's public HTTPS URL. A deployed frontend cannot use your personal computer's localhost. Include preprocessing.pkl, xgboost_model.json, features.pkl and reference.csv with the backend. Dockerfile optionally packages the API.

No public app/API URLs exist yet. Before submission, deploy both services, verify /health and a prediction online, add their URLs here, and disclose or complete Module 5.
