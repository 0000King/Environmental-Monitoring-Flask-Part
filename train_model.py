import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Step 1: Load the CSV data
df = pd.read_csv("sensor_data.csv")

# Step 2: Feature names
features = {
    "temperature": df[['temperature']],
    "humidity": df[['humidity']],
    "aqi": df[['aqi']],
    "tds": df[['tds']]
}

# Step 3: Train and save a model for each feature
for feature_name, data in features.items():
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    model.fit(data)
    joblib.dump(model, f"{feature_name}_model.pkl")
    print(f"✅ {feature_name.capitalize()} model trained and saved as {feature_name}_model.pkl")