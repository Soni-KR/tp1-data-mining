"""EduGuard first-semester prediction service."""
from pathlib import Path
from functools import lru_cache
import math
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

path = Path(__file__).resolve().parent
features = joblib.load(path / "features.pkl")
app = FastAPI(title="EduGuard API", version="1.0", description="Student decision support: Dropout=0, Graduate=1.")

@lru_cache
def get_model():
    model = joblib.load(path / "model.pkl")
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
    missing = sorted(set(features) - set(student.data))
    unexpected = sorted(set(student.data) - set(features))
    if missing or unexpected:
        raise HTTPException(422, detail={"missing_features": missing, "unexpected_features": unexpected})
    if not all(math.isfinite(v) for v in student.data.values()):
        raise HTTPException(422, "Feature values must be finite numbers")
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
