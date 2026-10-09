from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'app.py'

def test_dashboard_pages():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert len(app.metric) == 4
    for page in ['Data exploration', 'Preprocessing', 'Model comparison', 'Evaluation', 'Student prediction', 'About']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
    app.sidebar.radio[0].set_value('Student prediction').run()
    next(radio for radio in app.radio if radio.label == 'Student profile').set_value('Fictional / edited profile').run()
    assert not app.exception
    assert len(app.selectbox) + len(app.number_input) >= 30

def test_prediction_form_uses_http_client(monkeypatch):
    # A controlled model verifies the UI/HTTP contract independently of Windows DLL loading.
    from types import SimpleNamespace
    import numpy as np
    import api_client
    from fastapi.testclient import TestClient
    import api
    monkeypatch.setattr(api, 'get_model', lambda: SimpleNamespace(classes_=[0,1], predict_proba=lambda frame: np.array([[0.8,0.2]])))
    client = TestClient(api.app)
    calls = []
    def post(url, json, timeout):
        calls.append(json)
        return client.post('/predict', json=json)
    monkeypatch.setattr(api_client.requests, 'post', post)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.sidebar.radio[0].set_value('Student prediction').run()
    next(button for button in app.button if button.label == 'Predict').click().run()
    assert not app.exception
    assert len(calls) == 1
    assert len(calls[0]['data']) == 30
    assert len(app.metric) == 3

def test_training_and_evaluation_pages():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.sidebar.radio[0].set_value('Model comparison').run()
    next(widget for widget in app.multiselect if widget.label == 'Models').set_value(['Decision Tree']).run()
    next(button for button in app.button if button.label == 'Train and compare').click().run()
    assert not app.exception
    assert 'results' in app.session_state
    app.sidebar.radio[0].set_value('Evaluation').run()
    assert not app.exception
    app.slider[0].set_value(0.7).run()
    assert not app.exception
    app.checkbox[0].set_value(True).run()
    assert not app.exception
