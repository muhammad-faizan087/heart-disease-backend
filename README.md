# CardioPulse AI - Machine Learning Backend

FastAPI-powered machine learning inference service for cardiovascular risk and stroke prediction.

## Overview
This service loads pre-trained machine learning artifacts and serves high-performance predictive classification endpoints for cardiac risk factors.

- **Framework**: FastAPI + Uvicorn
- **ML Pipeline**: Scikit-Learn Logistic Regression with StandardScaler normalization
- **Artifacts**:
  - `model_heart_disease.pkl`: Pre-trained classifier
  - `scaler.pkl`: Feature normalization scaler
  - `columns.pkl`: One-hot encoded feature column schema

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <REPO_URL>
   cd <REPO_FOLDER>
   ```

2. **Create and activate a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## API Documentation

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc UI**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: `GET /`
- **Predict Endpoint**: `POST /predict`
