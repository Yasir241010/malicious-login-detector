# malicious-login-detector
# Malicious Login Detector

Streamlit demo using Logistic Regression to flag suspicious login attempts.

## Run locally
1. python -m venv venv
2. source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
3. pip install -r requirements.txt
4. streamlit run app.py

## Deploy
- Push this repo to GitHub.
- Go to https://share.streamlit.io → New app → select this repository, branch, and `app.py`.

## Notes
- This demo uses synthetic data for illustrative purposes.
- Next steps: replace with real login logs, add SHAP explainability, monitoring, and unit tests.
