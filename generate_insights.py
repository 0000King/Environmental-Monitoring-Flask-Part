from huggingface_hub import InferenceClient
import os
from flask_cors import CORS
from flask import Flask

app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'https://environmental-monitoring-spring-boot-and.onrender.com')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Replace with your Hugging Face token
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

# Create the client for the hosted model
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=HUGGINGFACE_TOKEN
)

def generate_insight(location, temp, humidity, aqi, tds, prompts):
    sensor_context = f"""
    Location: {location}
    Temperature: {temp}°C
    Humidity: {humidity}%
    Air Quality Index (AQI): {aqi}
    Water TDS: {tds} ppm
    """

    # Combine user prompt (if any) with the sensor context
    if prompts:
        prompt = f"{sensor_context}\n\nUser Prompt: {prompts}"
    else:
        prompt = f"""
        {sensor_context}

        Provide a concise environmental health insight based on the above parameters.
        """

    response = client.text_generation(prompt, max_new_tokens=200)
    return response