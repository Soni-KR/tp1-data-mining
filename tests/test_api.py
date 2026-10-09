import math
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
import api

client = TestClient(api.app)
student = pd.read_csv(api.path / 'reference.csv').iloc[0].to_dict()

def test_home_and_features():
    assert client.get('/').status_code == 200
    features = client.get('/features').json()['features']
    assert len(features) == 30
    assert not any('Curricular units 2nd sem' in f for f in features)
    assert client.get('/docs').status_code == 200

def test_missing_and_unexpected():
    missing = dict(student)
    feature = next(iter(missing))
    del missing[feature]
    response = client.post('/predict', json={'data': missing})
    assert response.status_code == 422
    assert feature in response.json()['detail']['missing_features']
    response = client.post('/predict', json={'data': {**student, 'unknown': 1}})
    assert response.status_code == 422
    assert 'unknown' in response.json()['detail']['unexpected_features']

def test_invalid_threshold():
    assert client.post('/predict', json={'data': student, 'threshold': 1.1}).status_code == 422

def test_probability_mapping_risk_and_threshold(monkeypatch):
    from types import SimpleNamespace
    for probability, risk in [(0.2,'Low'), (0.3,'Medium'), (0.7,'High')]:
        # Reverse class order to verify that the service looks up class 0.
        model = SimpleNamespace(classes_=[1,0], predict_proba=lambda frame: np.array([[1-probability,probability]]))
        monkeypatch.setattr(api, 'get_model', lambda: model)
        for threshold in [0, probability, 1]:
            result = client.post('/predict', json={'data': student, 'threshold':threshold}).json()
            assert result['dropout_probability'] == probability
            assert math.isclose(result['dropout_probability'] + result['graduation_probability'], 1)
            assert result['risk_level'] == risk
            assert result['prediction'] == ('Dropout' if probability >= threshold else 'Graduate')

def test_unavailable_health(monkeypatch):
    def unavailable():
        raise ImportError('Model blocked')
    monkeypatch.setattr(api, 'get_model', unavailable)
    assert client.get('/health').status_code == 503
    assert client.get('/health').json()['detail']['model_ready'] is False
    assert client.post('/predict', json={'data': student}).status_code == 503

def test_real_pipeline_predictions():
    # Intentionally fails, rather than skips, if the saved model cannot load.
    assert client.get('/health').status_code == 200
    model = api.get_model()
    direct = model.predict_proba(pd.DataFrame([student])[api.features])[0]
    for threshold in [0, 0.5, 1]:
        response = client.post('/predict', json={'data': student, 'threshold': threshold})
        assert response.status_code == 200
        result = response.json()
        assert math.isclose(result['dropout_probability'], direct[list(model.classes_).index(0)], abs_tol=1e-7)
        assert math.isclose(result['dropout_probability'] + result['graduation_probability'], 1)
        assert result['prediction'] == ('Dropout' if result['dropout_probability'] >= threshold else 'Graduate')
        assert result['threshold'] == threshold
        assert result['risk_level'] in ['Low', 'Medium', 'High']
