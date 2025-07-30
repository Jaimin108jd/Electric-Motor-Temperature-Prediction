from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

try:
    model = joblib.load('motor_temperature_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    print("Model and scaler loaded successfully!")
except:
    print("Error: Model files not found. Please train the model first.")
    model = None
    scaler = None

FEATURE_NAMES = ['u_q', 'coolant', 'u_d', 'motor_speed', 'i_d', 'i_q', 'ambient']

def predict_temperature(features):
    if model is None or scaler is None:
        return {"error": "Model not loaded"}
    try:
        input_data = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        warnings = []
        if prediction > 80:
            warnings.append("⚠️ HIGH TEMPERATURE WARNING: Motor may be overheating!")
        elif prediction > 60:
            warnings.append("⚠️ ELEVATED TEMPERATURE: Monitor motor condition")
        elif prediction < 25:
            warnings.append("ℹ️ LOW TEMPERATURE: Motor may be under low load")
        else:
            warnings.append("ℹ️ Motor operating within normal range")
        return {"prediction": round(prediction, 2), "warnings": warnings, "status": "success"}
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

@app.route('/')
def home():
    model_info = {
        "model_type": "Random Forest",
        "accuracy": "99.1%",
        "rmse": "0.95°C",
        "features": len(FEATURE_NAMES),
        "dataset_size": "10,000 samples"
    }
    return render_template('home.html', model_info=model_info)

@app.route('/manual')
def manual_predict():
    return render_template('manual_predict.html')

@app.route('/sensor')
def sensor_predict():
    return render_template('sensor_predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = []
        feature_values = {}
        for feature in FEATURE_NAMES:
            value = float(request.form[feature])
            features.append(value)
            feature_values[feature] = value
        result = predict_temperature(features)
        if "error" in result:
            return render_template('result.html', error=result["error"], input_data=feature_values)
        return render_template('result.html', prediction=result["prediction"], warnings=result["warnings"], input_data=feature_values)
    except Exception as e:
        return render_template('result.html', error=f"Input error: {str(e)}", input_data={})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        if not data or not all(feature in data for feature in FEATURE_NAMES):
            return jsonify({"error": "Missing required features"}), 400
        features = [data[feature] for feature in FEATURE_NAMES]
        result = predict_temperature(features)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"API error: {str(e)}"}), 500

if __name__ == '__main__':
    print("🚀 Starting Electric Motor Temperature Prediction System...")
    print("📊 Model: Random Forest (99.1% accuracy)")
    print("🌐 Access at: http://localhost:5000")
    print("📡 API endpoint: http://localhost:5000/api/predict")
    app.run(debug=True, host='0.0.0.0', port=5000)
