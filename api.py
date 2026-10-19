"""EduGuard first-semester prediction service."""
from pathlib import Path
from functools import lru_cache
import math
import joblib
import pandas as pd
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

path = Path(__file__).resolve().parent
features = pd.read_csv(path / "reference.csv", nrows=0).columns.tolist()
app = FastAPI(title="EduGuard API", version="1.0", description="Student decision support: Dropout=0, Graduate=1.")

@lru_cache
def get_model():
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    # Existing trained weights, exported to XGBoost's portable model format.
    classifier = XGBClassifier()
    classifier.load_model(path / "xgboost_model.json")
    model = Pipeline([
        ('preprocessing', joblib.load(path / "preprocessing.pkl")),
        ('model', classifier),
    ])
    if list(model.feature_names_in_) != features or set(model.classes_) != {0, 1}:
        raise ValueError("Saved model does not match the features or target mapping")
    if any("Curricular units 2nd sem" in f for f in features):
        raise ValueError("Second-semester features are not allowed")
    return model

class Student(BaseModel):
    model_config = {"json_schema_extra": {"example": {
        "data": pd.read_csv(path / "reference.csv", nrows=1).iloc[0].to_dict(),
        "threshold": 0.5,
    }}}
    data: dict[str, float]
    threshold: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)

class Prediction(BaseModel):
    dropout_probability: float
    graduation_probability: float
    prediction: str
    risk_level: str
    threshold: float

class Cohort(BaseModel):
    model_config = {'json_schema_extra': {'example': {
        'data': pd.read_csv(path / 'reference.csv', nrows=2).to_dict('records'),
    }}}
    data: list[dict[str, float]] = Field(min_length=2, max_length=200)
    labels: list[Literal[0, 1]] | None = None


def validate_data(data):
    missing = sorted(set(features) - set(data))
    unexpected = sorted(set(data) - set(features))
    if missing or unexpected:
        raise HTTPException(422, detail={"missing_features": missing, "unexpected_features": unexpected})
    if not all(math.isfinite(v) for v in data.values()):
        raise HTTPException(422, "Feature values must be finite numbers")


def background():
    frame = pd.read_csv(path / 'reference.csv')
    return frame.sample(min(500, len(frame)), random_state=42)

@app.get("/")
def home():
    return {"message": "EduGuard API is running", "docs": "/docs"}

@app.get("/health")
def health():
    try:
        model = get_model()
        probabilities = model.predict_proba(pd.read_csv(path / "reference.csv", nrows=1)[features])[0]
        if not all(math.isfinite(float(p)) for p in probabilities) or abs(sum(probabilities) - 1) > 1e-5:
            raise ValueError("Invalid probabilities")
    except Exception as exc:
        raise HTTPException(503, detail={"status": "unavailable", "model_ready": False, "error": str(exc)})
    return {"status": "ready", "model_ready": True, "features": len(features)}

@app.get("/features")
def get_features():
    return {"features": features}

@app.post("/predict", response_model=Prediction)
def predict(student: Student):
    validate_data(student.data)
    try:
        model = get_model()
        probabilities = model.predict_proba(pd.DataFrame([student.data])[features])[0]
        prob = float(probabilities[list(model.classes_).index(0)])
        if not math.isfinite(prob) or not 0 <= prob <= 1:
            raise ValueError("Invalid model probability")
    except Exception as exc:
        raise HTTPException(503, detail=f"Model unavailable: {exc}")
    return {
        "dropout_probability": prob,
        "graduation_probability": 1 - prob,
        "prediction": "Dropout" if prob >= student.threshold else "Graduate",
        "risk_level": "Low" if prob < 0.3 else "Medium" if prob < 0.7 else "High",
        "threshold": student.threshold,
    }


@app.post('/explain', summary='Explain one student with SHAP and LIME')
def explain(student: Student):
    """Same prediction as /predict, plus class-0 SHAP, LIME and a summary."""
    prediction = predict(student)
    try:
        from xai import local_explanation
        explanation = local_explanation(get_model(), background(), pd.DataFrame([student.data])[features], class_label='dropout')
    except Exception as exc:
        raise HTTPException(503, detail=f'Explanation unavailable: {exc}')
    return {**prediction, 'explanation': explanation}


@app.post('/explain/global', summary='Global importance and SHAP for a cohort (2-200 rows)')
def explain_global(cohort: Cohort):
    """Returns model importance and SHAP. Add one true 0/1 label per row
    (both classes required) to compute ROC-AUC permutation importance.
    A supplied cohort is not assumed to be an independent test set.
    """
    for row in cohort.data:
        validate_data(row)
    if cohort.labels is not None and (len(cohort.labels) != len(cohort.data) or set(cohort.labels) != {0, 1}):
        raise HTTPException(422, 'Labels must match the rows and include both classes: Dropout=0, Graduate=1')
    try:
        from xai import global_explanation
        explanation = global_explanation(get_model(), background(), pd.DataFrame(cohort.data)[features], cohort.labels)
    except Exception as exc:
        raise HTTPException(503, detail=f'Global explanation unavailable: {exc}')
    return {**explanation, 'scope': 'Supplied cohort; may overlap model training data. Descriptive importance, not an independent performance estimate.'}
