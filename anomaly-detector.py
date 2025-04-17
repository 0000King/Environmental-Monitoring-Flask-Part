import smtplib
from email.message import EmailMessage
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os

from generate_insights import generate_insight

app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'https://environmental-monitoring-spring-boot-and.onrender.com')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Load models for each feature
models = {
    "Temperature": joblib.load("temperature_model.pkl"),
    "Humidity": joblib.load("humidity_model.pkl"),
    "Air Quality": joblib.load("aqi_model.pkl"),
    "Water Quality": joblib.load("tds_model.pkl")
}

# Email configuration
EMAIL_SENDER = "amaninsightgeeks@gmail.com"
EMAIL_PASSWORD = os.getenv("E_PASS")

def send_email_alert(subject, body, email):
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_SENDER
    msg['To'] = email

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
        smtp.send_message(msg)

@app.route('/detect-anomaly', methods=['POST', 'OPTIONS'])
def detect_anomaly():
    if request.method == 'OPTIONS':
        # Preflight request — return necessary CORS headers
        response = jsonify({'message': 'CORS preflight'})
        response.headers.add('Access-Control-Allow-Origin', 'https://environmental-monitoring-spring-boot-and.onrender.com')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response

    data = request.json
    features = np.array(data['features'])  # Each element: [temp, humidity, aqi, tds]
    email = data['email']

    anomalies_report = []

    for i, row in enumerate(features):
        row = np.array(row).reshape(1, -1)  # Shape (1, 4)
        feature_values = {
            "Temperature": row[0][0],
            "Humidity": row[0][1],
            "Air Quality": row[0][2],
            "Water Quality": row[0][3]
        }

        detected_anomalies = []

        # Check each feature independently
        for feature_name, model in models.items():
            column_name = feature_name.lower().replace(" ", "_")  # Maps "Temperature" → "temperature"

            if column_name == "air_quality":
                column_name = "aqi"
            elif column_name == "water_quality":
                column_name = "tds"

            input_df = pd.DataFrame([[feature_values[feature_name]]], columns=[column_name])
            prediction = model.predict(input_df)
            if prediction[0] == -1:
                detected_anomalies.append(f"{feature_name}: {feature_values[feature_name]}")

        if detected_anomalies:
            anomalies_report.append({
                "reading": i,
                "anomalies": detected_anomalies
            })

    if anomalies_report:
        body_lines = ["Anomalies detected in the following readings:"]
        for report in anomalies_report:
            line = f"Reading {report['reading'] + 1}: " + ", ".join(report["anomalies"])
            body_lines.append(line)

        send_email_alert(
            subject="⚠️ Anomaly Detected in Sensor Data",
            body="\n".join(body_lines),
            email=email
        )

    response = jsonify({
        "anomaly_detected": bool(anomalies_report),
        "anomaly_details": anomalies_report
    })
    response.headers.add('Access-Control-Allow-Origin', 'https://environmental-monitoring-spring-boot-and.onrender.com')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

@app.route('/generate-insights', methods=['POST'])
def generate_location_insight():
    data = request.json
    location = data['location']
    temp = data['temperature']
    humidity = data['humidity']
    aqi = data['air']
    tds = data['water']
    print(data.get('prompt'))
    prompt = data.get('prompt', '')

    insight = generate_insight(location, temp, humidity, aqi, tds, prompt)
    return jsonify({"insight": insight})


if __name__ == '__main__':
    app.run(port=5001)