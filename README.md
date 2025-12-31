Credit Card Fraud Detection (ML + FastAPI)

This project builds a credit card fraud detection system using machine learning and exposes it through a FastAPI-based REST API for real-time scoring.

The focus is on correct feature engineering, handling class imbalance, and avoiding target leakage, rather than just achieving high accuracy.

What this project does

Trains a Random Forest classifier to detect fraudulent transactions

Uses target encoding + frequency encoding for high-cardinality merchant categories

Evaluates performance using PR-AUC, which is suitable for imbalanced fraud data

Provides a real-time API that returns a fraud risk score and decision

Model Performance

PR-AUC: ~0.93

ROC-AUC: ~0.99

(PR-AUC is emphasized due to heavy class imbalance.)

API Overview
Endpoint
POST /score

Input (example)
{
  "amount": 1200,
  "transaction_hour": 2,
  "merchant_category": "Electronics",
  "foreign_transaction": 1,
  "location_mismatch": 1,
  "device_trust_score": 30,
  "velocity_last_24h": 6,
  "cardholder_age": 28
}

Output
{
  "risk_score": 0.87,
  "decision": "ESCALATE"
}

How to run
pip install -r requirements.txt
uvicorn app:app --reload

Open: http://127.0.0.1:8000/docs

Key Takeaways

PR-AUC is more meaningful than accuracy for fraud detection

Target leakage can inflate metrics if encoding is done incorrectly

Strong feature engineering can outperform complex models
