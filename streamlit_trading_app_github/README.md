# Forex Trading Dashboard (Streamlit)

## Overview
A machine learning-based forex trading dashboard using Streamlit, with live OANDA data, technical indicators, model predictions, backtesting, and alerts.

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Add your OANDA API credentials in `oanda.cfg`.

3. Place your trained model as `model.pkl` in the root directory.

4. Run the app:
```bash
streamlit run app.py
```

## Alerts
To enable Telegram or email alerts, fill in your bot credentials or SMTP login in `app.py` and uncomment the relevant code.

## Deployment
You can deploy this app to [Streamlit Cloud](https://streamlit.io/cloud) by uploading all files.

