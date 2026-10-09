"""HTTP connection shared by Streamlit and its integration tests."""
import requests

def predict_student(api_url, data, threshold=0.5):
    response = requests.post(api_url.rstrip('/') + '/predict', json={'data': data, 'threshold': threshold}, timeout=30)
    response.raise_for_status()
    return response.json()
