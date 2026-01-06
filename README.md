### Credit Card Fraud Detection
## ML + FastAPI + AI Review Pipeline
## Overview

 - This project implements a real-time credit card fraud detection system using a traditional machine learning model exposed through a FastAPI REST API.
Transactions with uncertain confidence are escalated to an AI-based review layer, simulating real-world fraud analyst decision-making.

 - The project focuses on robust feature engineering, correct handling of class imbalance, prevention of target leakage, and auditability, rather than only optimizing accuracy.

 - What This Project Does

 - Trains a Random Forest classifier for fraud detection

 - Applies frequency encoding and target-aware encoding for high-cardinality merchant categories

 - Evaluates performance using PR-AUC, suitable for imbalanced fraud datasets

 - Exposes a low-latency REST API using FastAPI

 - Introduces an AI review layer for medium-confidence transactions

 - Logs AI-reviewed cases for audit and monitoring

### Decision Pipeline

 - Incoming transaction is validated and feature-engineered

 - ML model outputs a fraud risk probability

### Decision logic:

 - Low risk → APPROVE

 - High risk → BLOCK

 - Medium risk → Escalated to AI review

 - AI review returns a verdict with reasoning

 - Final decision is returned and logged for audit

 - This hybrid approach reduces unnecessary manual review while safely handling edge cases.

### Model Performance

 - PR-AUC: ~0.93

 - ROC-AUC: ~0.99

 - PR-AUC is emphasized due to severe class imbalance in fraud detection problems.

### API Overview
 - Endpoint
 - POST /score

 - Sample Request
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

 - Sample Response
{
  "risk_score": 0.87,
  "decision": "BLOCK",
  "ai_verdict": "FRAUD",
  "reason": "High transaction velocity combined with low device trust and foreign transaction behavior."
}

### AI Review System

 - Triggered only for medium-confidence transactions

 - Uses an LLM as a decision-support agent, not a replacement for the ML model

 - Returns a structured verdict and explanation

 - All AI-reviewed cases are logged for human inspection

 - This mirrors real-world fraud operations where ambiguous cases are escalated for secondary review.

### How to Run
 - pip install -r requirements.txt
 - uvicorn app:app --reload


### API documentation:

 - http://127.0.0.1:8000/docs

### Key Takeaways

 - PR-AUC is more reliable than accuracy for fraud detection

 - Incorrect encoding can introduce target leakage and inflate metrics

 - Strong feature engineering often outperforms complex models

 - Hybrid ML + AI systems handle ambiguity effectively

 - Auditability is critical in financial systems
