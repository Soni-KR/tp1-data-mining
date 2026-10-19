from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'app.py'

def test_dashboard_pages():
    # A fresh clone downloads the UCI data on the first exploration visit.
    app = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not app.exception
    assert len(app.metric) == 4
    for page in ['Data exploration', 'Preprocessing', 'Model comparison', 'Evaluation', 'Explainability', 'Student prediction', 'About']:
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
    import requests
    from fastapi.testclient import TestClient
    import api
    monkeypatch.setattr(api, 'get_model', lambda: SimpleNamespace(classes_=[0,1], predict_proba=lambda frame: np.array([[0.8,0.2]])))
    client = TestClient(api.app)
    calls = []
    def post(url, json, timeout):
        calls.append(json)
        return client.post('/predict', json=json)
    monkeypatch.setattr(requests, 'post', post)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.sidebar.radio[0].set_value('Student prediction').run()
    next(button for button in app.button if button.label == 'Predict').click().run()
    assert not app.exception
    assert len(calls) == 1
    assert len(calls[0]['data']) == 30
    assert len(app.metric) == 3

def test_training_and_evaluation_pages():
    app = AppTest.from_file(str(APP), default_timeout=90).run()
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
    app.sidebar.radio[0].set_value('Explainability').run()
    next(widget for widget in app.selectbox if widget.label == 'Model to explain').set_value('Decision Tree').run()
    next(button for button in app.button if button.label == 'Compute global explanations').click().run()
    assert not app.exception and not app.error
    assert len(app.get('plotly_chart')) == 4
    next(button for button in app.button if button.label == 'Explain held-out individual').click().run()
    assert not app.exception and not app.error
    assert len(app.get('plotly_chart')) == 6


def test_real_explanation_form_calls_api(monkeypatch):
    import requests
    from fastapi.testclient import TestClient
    import api
    client = TestClient(api.app)
    urls = []
    def post(url, json, timeout):
        urls.append(url)
        return client.post('/explain', json=json)
    monkeypatch.setattr(requests, 'post', post)
    app = AppTest.from_file(str(APP), default_timeout=90).run()
    app.sidebar.radio[0].set_value('Student prediction').run()
    next(widget for widget in app.checkbox if widget.label == 'Explain with SHAP and LIME').set_value(True)
    next(button for button in app.button if button.label == 'Predict').click().run()
    assert not app.exception
    assert urls[0].endswith('/explain')
    assert not app.error
    assert any('Main model contributions' in item.value for item in app.info)


def test_global_explanations_use_api_and_keep_results(monkeypatch):
    import requests
    from fastapi.testclient import TestClient
    import api
    client = TestClient(api.app)
    calls = []
    def post(url, json, timeout):
        calls.append(url)
        return client.post('/explain/global', json=json)
    monkeypatch.setattr(requests, 'post', post)
    app = AppTest.from_file(str(APP), default_timeout=90).run()
    app.sidebar.radio[0].set_value('Explainability').run()
    next(button for button in app.button if button.label == 'Compute global explanations').click().run()
    assert not app.exception and not app.error
    assert calls == ['http://127.0.0.1:8000/explain/global']
    assert len(app.get('plotly_chart')) == 4
    next(widget for widget in app.selectbox if widget.label == 'SHAP dependence feature').select_index(1).run()
    assert not app.exception and not app.error
    assert len(calls) == 1  # Changing a plot must not recompute explanations.
