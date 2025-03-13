from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import DBSCAN

app = FastAPI(title="Disease Outbreak Detection API")

# Load and preprocess dataset at startup
df = pd.read_csv("disease_spread.csv")

# Preprocessing
df['disease'] = df['disease'].astype(str).str.strip().str.lower()
df['pincode'] = df['pincode'].astype(str).str.strip()

# Encoding
le_disease = LabelEncoder()
le_pincode = LabelEncoder()
df['disease_encoded'] = le_disease.fit_transform(df['disease'])
df['pincode_encoded'] = le_pincode.fit_transform(df['pincode'])

# Scaling
X = df[['disease_encoded', 'pincode_encoded']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['cluster'] = dbscan.fit_predict(X_scaled)

@app.get("/")
def home():
    return {"message": "Welcome to Disease Outbreak Detection API!"}

@app.get("/check_outbreak")
def check_outbreak(pincode: str = Query(..., description="Enter the pincode to check for outbreaks")):
    user_pincode = str(pincode).strip()
    df_pincode = df[df['pincode'] == user_pincode]

    if df_pincode.empty:
        return {"status": "No records found", "pincode": user_pincode}

    disease_counts = df_pincode['disease'].value_counts()
    outbreak_report = []
    outbreak_flag = False

    for disease, count in disease_counts.items():
        if count >= 70:
            alert = "RED ALERT - High spread detected!"
            outbreak_flag = True
        elif count >= 30:
            alert = "ORANGE ALERT - Moderate spread detected."
            outbreak_flag = True
        elif count >= 10:
            alert = "YELLOW ALERT - Early warning zone."
            outbreak_flag = True
        else:
            alert = "No outbreak for this disease."

        outbreak_report.append({
            "disease": disease.title(),
            "patients_affected": int(count),
            "alert_level": alert
        })

    return {
        "status": "Outbreak Report",
        "pincode": user_pincode,
        "alerts": outbreak_report,
        "outbreak_detected": outbreak_flag
    }
