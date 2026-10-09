# Test report

**9 October 2026 — 11 tests passed on Windows Python 3.13.**

Verified:

- API starts; /docs works; /features returns 30 semester-one inputs.
- Health reports readiness accurately.
- Valid students produce real probabilities; invalid features and thresholds are rejected.
- Probabilities sum to one; thresholds and risk boundaries behave correctly.
- API predictions match direct pipeline inference.
- All seven dashboard pages and manual student inputs work.
- Training works with an alternative CSV and through the dashboard.
- Evaluation curves, confusion matrix and threshold controls render.
- The live Streamlit prediction form communicates with FastAPI over HTTP.

**Real example:** reference row 0 returns 78.1% dropout probability, 21.9% graduation probability and High risk. At threshold .50 it predicts Dropout; at .90 it predicts Graduate. reference.csv has no outcome labels, so this verifies inference, not accuracy.

The original model.pkl and notebook are unchanged. Portable model files matched the recovered original pipeline exactly on 100 profiles. Compatible scientific packages resolved Windows DLL failures without changing security settings.

The one-pager was checked to contain exactly one page and visually reviewed. Screenshots show the working dashboard, prediction and API docs.

Run: `.\.venv\Scripts\python.exe -m pytest -q`.

**Still to submit:** public deployment URLs and the deferred SHAP/LIME module. The optional API Docker image has not been built/tested locally. Historical performance figures come from the existing notebook.
