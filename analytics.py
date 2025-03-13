import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import DBSCAN

df = pd.read_csv("disease_spread.csv")

df['disease'] = df['disease'].astype(str).str.strip().str.lower()
df['pincode'] = df['pincode'].astype(str).str.strip()

le_disease = LabelEncoder()
le_pincode = LabelEncoder()
df['disease_encoded'] = le_disease.fit_transform(df['disease'])
df['pincode_encoded'] = le_pincode.fit_transform(df['pincode'])

X = df[['disease_encoded', 'pincode_encoded']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.5, min_samples=5)
df['cluster'] = dbscan.fit_predict(X_scaled)

def alert_by_pincode(user_pincode):
    user_pincode = str(user_pincode).strip()

    df_pincode = df[df['pincode'] == user_pincode]

    if df_pincode.empty:
        return f"No records found for pincode {user_pincode}."

    disease_counts = df_pincode['disease'].value_counts()

    result = f"Outbreak Report for Pincode: {user_pincode}\n"
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

        result += f"\nDisease: {disease.title()}\nPatients Affected: {count}\nAlert Level: {alert}\n"

    if not outbreak_flag:
        result += "\nNo major outbreak detected in this pincode."

    return result

user_input = input("Enter a pincode to check for outbreak: ")
print(alert_by_pincode(user_input))
