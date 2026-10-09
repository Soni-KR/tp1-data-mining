# EduGuard — simple guide

EduGuard estimates whether a student may drop out or graduate using information available at the **end of semester one**. It supports academic follow-up; predictions are not certain outcomes.

## 1. Open the application

Open this folder:

`C:\Users\moura\Desktop\ENSI\3eme\data mining`

Double-click **start_eduguard.bat**. It starts two programs:

- **FastAPI** runs the saved model.
- **Streamlit** shows the dashboard in your browser.

Keep their two windows open. Open [the dashboard](http://localhost:8501) if the browser does not open automatically. Close both windows when finished.

**Localhost means your own computer.** The app runs locally; it is not published online.

If you prefer terminals, run these commands in two separate PowerShell windows, from the project folder:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## 2. Test a student without writing code

1. Open [http://localhost:8501](http://localhost:8501).
2. Keep **FastAPI URL** set to `http://127.0.0.1:8000`.
3. Click **Check API readiness** in the sidebar.
4. If it says **Model is ready**, open **Student prediction**.
5. Choose **Existing reference profile** and select a row.
6. Keep the threshold at **0.50**, then click **Predict**.
7. Read the dropout probability, graduation probability, predicted outcome and risk category.
8. Try another row, or choose **Fictional / edited profile** to change the inputs.

No retraining is needed to use the saved model.

**Current blocker:** on this Windows environment, Application Control blocks scikit-learn. You can open and explore the dashboard, but real predictions and training currently fail. An unavailable-model message is honest reporting, not a successful prediction. Starting FastAPI does not fix this restriction.

Do not disable Windows security. To finish model testing, use an allowed environment with compatible dependencies; optional deployment files are included.

## 3. Understand the result

- **Dropout probability:** the model's estimated risk, not a confirmed future outcome.
- **Graduation probability:** the complementary estimate. Both probabilities sum to 100%.
- **Threshold:** predicts Dropout when its probability is at least this value. At 0.50, a 60% dropout estimate is classified as Dropout.
- Lowering the threshold flags more students; raising it flags fewer.
- **Risk category:** Low below 30%; Medium from 30% to below 70%; High at 70% or more.

Risk categories stay the same when you change the threshold. Their boundaries are application choices, not validated risk bands. Suggested follow-up includes counseling, tutoring, financial-support assessment and supportive outreach.

## 4. Test FastAPI directly in the browser

Open [http://localhost:8000/docs](http://localhost:8000/docs).

- Expand **GET /health**, click **Try it out**, then **Execute**.
- **200 + model_ready: true** means a sample model prediction succeeded.
- **503 + model_ready: false** means the model is unavailable.
- Expand **POST /predict**, click **Try it out**, keep the supplied student example, then **Execute**.
- A successful response shows probabilities, prediction, risk level and threshold.
- Change only `threshold` to test a different decision boundary.

The example uses real feature values from reference.csv; it does not contain the student's actual outcome.

Other endpoints: **GET /** (service information), **GET /features** (30 expected feature names). Missing or extra features return **422**. Model-loading failure returns **503**.

## 5. Use the other dashboard pages

| Page | What to do |
|---|---|
| Overview | Read the objective, dataset summary and historical model results. |
| Data exploration | Inspect the labeled dataset, missing values, charts and correlations. You can also upload a labeled CSV. |
| Preprocessing | Select features and processing settings; click Save preprocessing settings. |
| Model comparison | Select algorithms; click Train and compare. GridSearchCV is optional and slower. |
| Evaluation | After training, inspect metrics and curves; explore thresholds using training results before viewing held-out test results. |
| Student prediction | Use the existing saved model through FastAPI. |
| About | Read limitations and the project's scope. |

Training runs only when you click its button. Dashboard experiments do not replace model.pkl. Preprocessing is fitted inside training pipelines to avoid data leakage. Custom CSVs need a binary target and a selected class of interest.

Do not keep adjusting thresholds based on the final test set. Out-of-fold training results after hyperparameter tuning are exploratory estimates.

## 6. Project facts for your presentation

Dataset: [UCI](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) / [Kaggle](https://www.kaggle.com/datasets/mattop/predict-students-dropout-and-academic-success).

- Original data: 4,424 students, 36 features, three outcomes.
- Enrolled is removed because the final outcome is unknown.
- Remaining data: 3,630 students — 1,421 Dropout and 2,209 Graduate.
- All semester-two variables are removed because they are unavailable at prediction time.
- Model inputs: 30 features. Target mapping: **Dropout=0, Graduate=1**.
- Notebook processing: stratified 80/20 split, StandardScaler, OneHotEncoder, binary indicators preserved; five-fold GridSearchCV.
- Five algorithms: Logistic Regression, Decision Tree, KNN, Random Forest and XGBoost.

Historical notebook results, not new dashboard measurements:

| Model | CV ROC-AUC | Test ROC-AUC | Dropout F1 |
|---|---:|---:|---:|
| Logistic Regression | .9348 | .9406 | .8587 |
| Decision Tree | .8924 | .9038 | .8302 |
| KNN | .8775 | .8762 | .6591 |
| Random Forest | .9303 | .9394 | .8387 |
| XGBoost | .9351 | .9404 | .8717 |

Saved XGBoost dropout precision: .9390; recall: .8134. These results do not establish performance at another university or calibrated probabilities. Historical model selection considered test results, so an external untouched dataset is needed for an unbiased final assessment.

**Module 5 (SHAP/LIME) is deferred.** The project does not yet satisfy all six original modules. It provides no individual causal explanations and should not make automatic decisions about students.

## 7. Know the important files

| File | Purpose |
|---|---|
| start_eduguard.bat | Double-click to start the local app. |
| app.py | Streamlit dashboard. |
| api.py | FastAPI prediction service. |
| api_client.py | Sends dashboard requests to FastAPI. |
| ml_utils.py | Optional model-training and evaluation helpers. |
| model.pkl | Existing trained pipeline, including preprocessing. |
| features.pkl | Names of the 30 model inputs. |
| reference.csv | Example profiles without labels; not accuracy ground truth. |
| data.csv | Complete labeled dataset for exploration and experiments. |
| prj1-datamining.ipynb | Existing training notebook, preserved. |
| requirements.txt | Dependencies for a fresh compatible Python 3.12 environment. |
| TEST_RESULTS.md | Short test report and unresolved work. |

## 8. Deployment and remaining work

The simplest prepared arrangement is **FastAPI on a compatible server + Streamlit on Streamlit Community Cloud**. Set `EDUGUARD_API_URL` in the frontend's environment or Streamlit secrets to the backend's public URL. A cloud frontend cannot use your computer's localhost.

The saved model records scikit-learn **1.6.1** and XGBoost **3.4.1**; requirements match these versions. The current Windows environment has scikit-learn 1.9.1 and is also blocked by policy. Repeated reinstalling is not the proposed fix.

Dockerfile and compose.yaml are optional. If Docker Desktop's Linux engine is running, `docker compose up --build` starts both services. setup_linux.sh is another optional setup helper; neither path has been fully verified.

Before submission: successfully test real predictions, deploy the app, and disclose the deferred SHAP/LIME module. No external deployment or account creation has been done.

Optional automated checks: `.\.venv\Scripts\python.exe -m pytest -q`. You do not need this command for browser-based testing.
