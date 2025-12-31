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
