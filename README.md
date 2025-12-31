# Credit Card Fraud Detection (ML + FastAPI)

This project builds a **credit card fraud detection system** using machine learning and exposes it through a **FastAPI-based REST API** for real-time scoring.

The focus is on **correct feature engineering**, **handling class imbalance**, and **avoiding target leakage**, rather than just achieving high accuracy.

---

## What this project does

- Trains a **Random Forest classifier** to detect fraudulent transactions  
- Uses **target encoding + frequency encoding** for high-cardinality merchant categories  
- Evaluates performance using **PR-AUC**, suitable for imbalanced fraud data  
- Provides a **real-time API** that returns a fraud risk score and decision  

---

## Model Performance

- **PR-AUC:** ~0.93  
- **ROC-AUC:** ~0.99  

> PR-AUC is emphasized due to heavy class imbalance.

---

## API Overview

### Endpoint

POST /score


### Sample Input
```json
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
Sample Output
{
  "risk_score": 0.87,
  "decision": "ESCALATE"
}

Project Structure
.
├── eda_and_model.ipynb
├── app.py
├── random_forest_model.pkl
├── requirements.txt
└── README.md

How to Run
pip install -r requirements.txt
uvicorn app:app --reload


Open: http://127.0.0.1:8000/docs

Key Takeaways

PR-AUC is more reliable than accuracy for fraud detection

Target leakage can inflate metrics if encoding is done incorrectly

Strong feature engineering matters more than complex models


---

## 3️⃣ Why this version works

- Clear **visual hierarchy**
- Headings stand out
- Bullet points are readable
- Short enough for evaluators
- Does **not** look AI-generated

This is exactly how **good student projects** look on GitHub.

---

## 4️⃣ One last small polish (optional but recommended)

Rename:


eda_&_model.ipynb → eda_and_model.ipynb


And update the filename in README.

---

## Final check

Once you save this:
- Refresh the GitHub page
- You’ll immediately see **bold headings and clean sections**

You’re basically **done** now.

If you want:
- I can compress this to **5–6 lines only**
- Or write a **1-line repo description** for the sidebar
