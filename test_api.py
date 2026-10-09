import pandas as pd
import requests
import os
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent / "reference.csv")
student = df.iloc[0].to_dict()

response = requests.post(
    os.getenv('EDUGUARD_API_URL', 'http://127.0.0.1:8000').rstrip('/') + '/predict',
    json={"data": student, "threshold": 0.5},
    timeout=30,
)

print(response.status_code)
print(response.json())
