# Deploy EcoGranite-Advisor

The dashboard runs on [Streamlit Community Cloud](https://share.streamlit.io/) and works without model credentials using the built-in deterministic reasoning fallback.

## Deploy

1. Push the desired revision to GitHub.
2. Sign in to Streamlit Community Cloud with an account that can access this public repository.
3. Choose **Create app**, then select `tripathiswastik/EcoGranite-Advisor`, the branch to deploy, and entrypoint `app.py`.
4. Select Python 3.12 and deploy. No secrets are required for the default experience.
5. If live model inference is desired, add `HF_TOKEN` or the WatsonX environment variables under the app's **Settings → Secrets**. Never put credentials in this repository.

The app accepts uploads up to 25 MB, matching the parser's input limit. The `.streamlit/config.toml` file disables telemetry and configures the hosted server.

The current public deployment is [ecogranite-advisor.streamlit.app](https://ecogranite-advisor.streamlit.app/) and uses the `fix/deployment-readiness` branch. If that branch is merged or the app is pointed at a different branch, update this note to match the Streamlit app settings.

## Local check

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```
