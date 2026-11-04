# Deploy EcoGranite-Advisor

The dashboard runs on [Streamlit Community Cloud](https://share.streamlit.io/) and works without model credentials using the built-in deterministic reasoning fallback.

## Deploy

1. Push or merge the desired revision to GitHub.
2. Sign in to Streamlit Community Cloud with an account that can access this public repository.
3. Choose **Create app**, then select `tripathiswastik/EcoGranite-Advisor`, branch `main`, and entrypoint `app.py`.
4. Keep the default Python version (3.12) and deploy. No secrets are required for the default experience.
5. If live model inference is desired, add `HF_TOKEN` or the WatsonX environment variables under the app's **Settings → Secrets**. Never put credentials in this repository.

The app accepts uploads up to 25 MB, matching the parser's input limit. The `.streamlit/config.toml` file disables telemetry and configures the hosted server.

## Local check

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```
