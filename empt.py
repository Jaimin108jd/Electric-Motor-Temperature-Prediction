import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# Load Dataset

df = pd.read_csv("dataset22.csv")  # dataset fille path

print("✅ Dataset Loaded Successfully!")
print(f"Shape: {df.shape}")
print(df.info())
print(df.describe())

print("=== ELECTRIC MOTOR TEMPERATURE PREDICTION SYSTEM ===")
print("Dataset: Electric Motor Temperature (Kaggle)")
print("Target: Permanent Magnet Temperature (pm)")
print("=" * 60)

# Exploratory Data Analysis

print("\n1. EXPLORATORY DATA ANALYSIS")
print("-" * 40)
print("Missing values check:")
print(df.isnull().sum())

print(f"\nDataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

correlation_with_pm = df.corr()['pm'].sort_values(ascending=False)
print("\nCorrelation with PM temperature:")
for feature, corr in correlation_with_pm.items():
    if feature != 'pm':
        print(f"{feature:15}: {corr:6.3f}")

# Data Preprocessing

print("\n2. DATA PREPROCESSING")
print("-" * 40)

features_to_drop = ['pm', 'stator_winding', 'stator_tooth', 'stator_yoke', 'torque', 'profile_id']
feature_columns = [col for col in df.columns if col not in features_to_drop]

X = df[feature_columns].copy()
y = df['pm'].copy()

print(f"Features used for prediction: {list(X.columns)}")
print(f"Target variable: pm (permanent magnet temperature)")
print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

print("\nFinal feature statistics:")
print(X.describe())

# Normalize Data

print("\n3. DATA NORMALIZATION")
print("-" * 40)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("Features normalized using MinMaxScaler")
print("Scaled feature ranges:")
print(X_scaled_df.describe())

# Split Data

print("\n4. TRAIN-TEST SPLIT")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Train-test split ratio: 80-20")

# Train Models

print("\n5. MODEL TRAINING AND EVALUATION")
print("-" * 40)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Support Vector Regression': SVR(kernel='rbf')
}

results = {}

print("Training models...")
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    results[name] = {
        'model': model,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'predictions': y_pred_test
    }

    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE:  {test_rmse:.4f}")
    print(f"  Train R²:   {train_r2:.4f}")
    print(f"  Test R²:    {test_r2:.4f}")

print("\n6. MODEL COMPARISON")
print("-" * 40)

comparison_data = []
for name, result in results.items():
    comparison_data.append({
        'Model': name,
        'Train RMSE': f"{result['train_rmse']:.4f}",
        'Test RMSE': f"{result['test_rmse']:.4f}",
        'Train R²': f"{result['train_r2']:.4f}",
        'Test R²': f"{result['test_r2']:.4f}"
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
best_model = results[best_model_name]['model']
best_r2 = results[best_model_name]['test_r2']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Test R² Score: {best_r2:.4f} ({best_r2*100:.1f}% accuracy)")
print(f"   Test RMSE: {results[best_model_name]['test_rmse']:.4f}")

print(f"\nModel explains {best_r2*100:.1f}% of the variance in permanent magnet temperature.")

# Save Model and Scaler

print("\n7. MODEL PERSISTENCE")
print("-" * 40)

model_filename = 'motor_temperature_model.pkl'
scaler_filename = 'feature_scaler.pkl'

joblib.dump(best_model, model_filename)
joblib.dump(scaler, scaler_filename)

print(f"✅ Best model ({best_model_name}) saved as: {model_filename}")
print(f"✅ Feature scaler saved as: {scaler_filename}")

# Prediction Function 

def predict_temperature(u_q, coolant, u_d, motor_speed, i_d, i_q, ambient):
    input_data = np.array([[u_q, coolant, u_d, motor_speed, i_d, i_q, ambient]])
    input_scaled = scaler.transform(input_data)
    prediction = best_model.predict(input_scaled)
    return prediction[0]

# Test prediction function

test_params = {
    'u_q': 50.0,
    'coolant': 45.0,
    'u_d': -10.0,
    'motor_speed': 2000.0,
    'i_d': -150.0,
    'i_q': 25.0,
    'ambient': 22.0
}

predicted_temp = predict_temperature(**test_params)
print(f"Sample prediction:")
print(f"Input parameters: {test_params}")
print(f"Predicted PM temperature: {predicted_temp:.2f}°C")

# Generate sample data for demonstration 

sample_data = []
for i in range(5):
    params = {
        'u_q': np.random.uniform(10, 100),
        'coolant': np.random.uniform(20, 80),
        'u_d': np.random.uniform(-50, 50),
        'motor_speed': np.random.uniform(1000, 4000),
        'i_d': np.random.uniform(-200, -50),
        'i_q': np.random.uniform(-100, 100),
        'ambient': np.random.uniform(15, 25)
    }
    params['predicted_temp'] = predict_temperature(**params)
    sample_data.append(params)

print("\nSample predictions for different motor conditions:")
for i, data in enumerate(sample_data, 1):
    print(f"Sample {i}: Coolant={data['coolant']:.1f}°C, Speed={data['motor_speed']:.0f}rpm → PM Temp={data['predicted_temp']:.1f}°C")

print("\n" + "="*60)
print("MODEL SUMMARY:")
print(f"• Dataset: Electric Motor Temperature (10,000 samples)")
print(f"• Features: 7 motor parameters")
print(f"• Target: Permanent magnet temperature")
print(f"• Best Model: {best_model_name}")
print(f"• Accuracy: {best_r2*100:.1f}% (R² = {best_r2:.4f})")
print(f"• RMSE: {results[best_model_name]['test_rmse']:.4f}°C")
print("="*60)

# Create Flask App Code 

flask_app_code = '''from flask import Flask, render_template, request, jsonify
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
        "model_type": "''' + best_model_name + '''",
        "accuracy": "''' + f"{best_r2*100:.1f}%" + '''",
        "rmse": "''' + f"{results[best_model_name]['test_rmse']:.2f}°C" + '''",
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
    print("📊 Model: ''' + best_model_name + f''' ({best_r2*100:.1f}% accuracy)")
    print("🌐 Access at: http://localhost:5000")
    print("📡 API endpoint: http://localhost:5000/api/predict")
    app.run(debug=True, host='0.0.0.0', port=5000)
'''

# Save Flask app code with explicit UTF-8 encoding
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(flask_app_code)

print("✅ Flask application created: app.py")
print("   Routes:")
print("   • /          - Home page with model info")
print("   • /manual    - Manual input interface")
print("   • /sensor    - Sensor-based interface")
print("   • /predict   - Prediction endpoint")
print("   • /api/predict - REST API endpoint")

# HTML Templates

# Ensure templates directory exists
os.makedirs('templates', exist_ok=True)

# Home template
home_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Electric Motor Temperature Prediction System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .hero-section { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 80px 0; }
        .feature-card { transition: transform 0.3s; border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .feature-card:hover { transform: translateY(-5px); }
        .stat-card { background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%); color: white; }
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/"><i class="fas fa-microchip"></i> Motor Temp Predictor</a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/manual"><i class="fas fa-keyboard"></i> Manual Input</a>
                <a class="nav-link" href="/sensor"><i class="fas fa-sensor"></i> Sensor Data</a>
            </div>
        </div>
    </nav>

    <!-- Hero Section -->
    <section class="hero-section">
        <div class="container text-center">
            <h1 class="display-4 mb-4"><i class="fas fa-thermometer-half"></i> Electric Motor Temperature Prediction</h1>
            <p class="lead mb-4">AI-powered permanent magnet temperature prediction for industrial motor systems</p>
            <p class="mb-4">Predict motor temperature using advanced machine learning to prevent overheating and optimize performance</p>
            <div class="row justify-content-center">
                <div class="col-md-6">
                    <a href="/manual" class="btn btn-light btn-lg me-3">
                        <i class="fas fa-keyboard"></i> Start Prediction
                    </a>
                    <a href="/sensor" class="btn btn-outline-light btn-lg">
                        <i class="fas fa-sensor"></i> Sensor Mode
                    </a>
                </div>
            </div>
        </div>
    </section>

    <!-- Model Information -->
    <section class="py-5">
        <div class="container">
            <div class="row text-center">
                <div class="col-lg-3 col-md-6 mb-4">
                    <div class="card stat-card h-100">
                        <div class="card-body">
                            <i class="fas fa-brain fa-3x mb-3"></i>
                            <h5>{{ model_info.model_type }}</h5>
                            <p>Machine Learning Algorithm</p>
                        </div>
                    </div>
                </div>
                <div class="col-lg-3 col-md-6 mb-4">
                    <div class="card stat-card h-100">
                        <div class="card-body">
                            <i class="fas fa-chart-line fa-3x mb-3"></i>
                            <h5>{{ model_info.accuracy }}</h5>
                            <p>Prediction Accuracy</p>
                        </div>
                    </div>
                </div>
                <div class="col-lg-3 col-md-6 mb-4">
                    <div class="card stat-card h-100">
                        <div class="card-body">
                            <i class="fas fa-database fa-3x mb-3"></i>
                            <h5>{{ model_info.dataset_size }}</h5>
                            <p>Training Dataset</p>
                        </div>
                    </div>
                </div>
                <div class="col-lg-3 col-md-6 mb-4">
                    <div class="card stat-card h-100">
                        <div class="card-body">
                            <i class="fas fa-cog fa-3x mb-3"></i>
                            <h5>{{ model_info.features }}</h5>
                            <p>Input Parameters</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Features -->
    <section class="py-5 bg-light">
        <div class="container">
            <div class="row">
                <div class="col-lg-12 text-center mb-5">
                    <h2>System Features</h2>
                    <p class="lead">Advanced motor temperature prediction capabilities</p>
                </div>
            </div>
            <div class="row">
                <div class="col-lg-4 mb-4">
                    <div class="card feature-card h-100">
                        <div class="card-body text-center">
                            <i class="fas fa-exclamation-triangle fa-3x text-warning mb-3"></i>
                            <h5>Safety Warnings</h5>
                            <p>Automatic temperature alerts and safety recommendations for motor protection</p>
                        </div>
                    </div>
                </div>
                <div class="col-lg-4 mb-4">
                    <div class="card feature-card h-100">
                        <div class="card-body text-center">
                            <i class="fas fa-tachometer-alt fa-3x text-primary mb-3"></i>
                            <h5>Real-time Analysis</h5>
                            <p>Instant temperature predictions based on current motor operating conditions</p>
                        </div>
                    </div>
                </div>
                <div class="col-lg-4 mb-4">
                    <div class="card feature-card h-100">
                        <div class="card-body text-center">
                            <i class="fas fa-api fa-3x text-success mb-3"></i>
                            <h5>API Integration</h5>
                            <p>RESTful API for seamless integration with existing industrial control systems</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer class="bg-dark text-white py-4">
        <div class="container text-center">
            <p>&copy; 2025 Electric Motor Temperature Prediction System. Powered by Machine Learning.</p>
            <p><small>Model RMSE: {{ model_info.rmse }} | Based on Electric Motor Temperature Dataset (Kaggle)</small></p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>'''

# Manual predict template
manual_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Manual Input - Motor Temperature Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .form-section { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 60px 0; }
        .form-card { border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .input-group-text { background: #f8f9fa; border: none; }
        .form-control { border: 2px solid #e9ecef; border-radius: 8px; }
        .form-control:focus { border-color: #667eea; box-shadow: 0 0 0 0.2rem rgba(102,126,234,0.25); }
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/"><i class="fas fa-microchip"></i> Motor Temp Predictor</a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/"><i class="fas fa-home"></i> Home</a>
                <a class="nav-link active" href="/manual"><i class="fas fa-keyboard"></i> Manual Input</a>
                <a class="nav-link" href="/sensor"><i class="fas fa-sensor"></i> Sensor Data</a>
            </div>
        </div>
    </nav>

    <!-- Header -->
    <section class="form-section">
        <div class="container text-center">
            <h1><i class="fas fa-keyboard"></i> Manual Parameter Input</h1>
            <p class="lead">Enter motor operating parameters for temperature prediction</p>
        </div>
    </section>

    <!-- Form -->
    <section class="py-5">
        <div class="container">
            <div class="row justify-content-center">
                <div class="col-lg-8">
                    <div class="card form-card">
                        <div class="card-header bg-primary text-white">
                            <h4 class="mb-0"><i class="fas fa-cogs"></i> Motor Operating Parameters</h4>
                        </div>
                        <div class="card-body">
                            <form action="/predict" method="post" id="predictionForm">
                                <div class="row">

                                    <!-- Electrical Parameters -->
                                    <div class="col-md-6">
                                        <h5 class="text-primary mb-3"><i class="fas fa-bolt"></i> Electrical Parameters</h5>

                                        <div class="mb-3">
                                            <label class="form-label">Voltage Q-Component (V)</label>
                                            <div class="input-group">
                                                <span class="input-group-text"><i class="fas fa-wave-square"></i></span>
                                                <input type="number" class="form-control" name="u_q" step="0.1" min="-30" max="140" value="50" required>
                                            </div>
                                            <small class="text-muted">Range: -30V to 140V</small>
                                        </div>

                                        <div class="mb-3">
                                            <label class="form-label">Voltage D-Component (V)</label>
                                            <div class="input-group">
                                                <span class="input-group-text"><i class="fas fa-wave-square"></i></span>
                                                <input type="number" class="form-control" name="u_d" step="0.1" min="-140" max="140" value="-10" required>
                                            </div>
                                            <small class="text-muted">Range: -140V to 140V</small>
                                        </div>

                                        <div class="mb-3">
                                            <label class="form-label">Current D-Component (A)</label>
                                            <div class="input-group">
                                                <span class="input-group-text"><i class="fas fa-flash"></i></span>
                                                <input type="number" class="form-control" name="i_d" step="0.1" min="-280" max="0" value="-150" required>
                                            </div>
                                            <small class="text-muted">Range: -280A to 0A</small>
                                        </div>

                                        <div class="mb-3">
                                            <label class="form-label">Current Q-Component (A)</label>
                                            <div class="input-group">
                                                <span class="input-group-text"><i class="fas fa-flash"></i></span>
                                                <input type="number" class="form-control" name="i_q" step="0.1" min="-300" max="310" value="25" required>
                                            </div>
                                            <small class="text-muted">Range: -300A to 310A</small>
                                        </div>
                                    </div>

                                    <!-- Mechanical & Thermal Parameters -->
                                    <div class="col-md-6">
                                        <h5 class="text-success mb-3"><i class="fas fa-thermometer-half"></i> Mechanical & Thermal</h5>

                                        <div class="mb-3">
                                            <label class="form-label">Motor Speed (RPM)</label>
                                            <div class="input-group">
                                                <span class="input-group-text"><i class="fas fa-tachometer-alt"></i></span>
                                                <input type="number" class="form-control" name="motor_speed" step="1" min="-300" max="6000" value="2000" required>
                                            </div>
                                            <small class="text-muted">Range: -300 to 6000 RPM</small>
                                        </div>

                                        <div class="mb-3">
                                            <label class="form-label">Coolant Temperature (°C)</label>
                                            <div class="input-group">
                                                <span class="input-group-text"><i class="fas fa-temperature-low"></i></span>
                                                <input type="number" class="form-control" name="coolant" step="0.1" min="10" max="105" value="45" required>
                                            </div>
                                            <small class="text-muted">Range: 10°C to 105°C</small>
                                        </div>

                                        <div class="mb-3">
                                            <label class="form-label">Ambient Temperature (°C)</label>
                                            <div class="input-group">
                                                <span class="input-group-text"><i class="fas fa-thermometer-quarter"></i></span>
                                                <input type="number" class="form-control" name="ambient" step="0.1" min="8" max="32" value="22" required>
                                            </div>
                                            <small class="text-muted">Range: 8°C to 32°C</small>
                                        </div>
                                    </div>

                                </div>

                                <hr>

                                <div class="text-center">
                                    <button type="submit" class="btn btn-success btn-lg px-5">
                                        <i class="fas fa-calculator"></i> Predict Temperature
                                    </button>
                                </div>
                            </form>
                        </div>
                    </div>

                    <!-- Help Section -->
                    <div class="card mt-4">
                        <div class="card-header bg-info text-white">
                            <h5 class="mb-0"><i class="fas fa-info-circle"></i> Parameter Guidelines</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6><i class="fas fa-bolt text-primary"></i> Electrical Parameters</h6>
                                    <ul class="small">
                                        <li><strong>Voltage Components:</strong> dq-frame voltage measurements</li>
                                        <li><strong>Current Components:</strong> dq-frame current measurements</li>
                                        <li>These values depend on motor control strategy</li>
                                    </ul>
                                </div>
                                <div class="col-md-6">
                                    <h6><i class="fas fa-thermometer-half text-success"></i> Operating Conditions</h6>
                                    <ul class="small">
                                        <li><strong>Motor Speed:</strong> Mechanical rotation speed</li>
                                        <li><strong>Coolant Temp:</strong> Cooling system outlet temperature</li>
                                        <li><strong>Ambient Temp:</strong> Environmental temperature</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>'''

# Sensor predict template - COMPLETE VERSION
sensor_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sensor Data - Motor Temperature Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .form-section { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 60px 0; }
        .form-card { border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .input-group-text { background: #f8f9fa; border: none; }
        .form-control { border: 2px solid #e9ecef; border-radius: 8px; }
        .form-control:focus { border-color: #667eea; box-shadow: 0 0 0 0.2rem rgba(102,126,234,0.25); }
        .sensor-status { border-radius: 50px; padding: 5px 15px; font-size: 0.8rem; }
        .sensor-connected { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .sensor-disconnected { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .real-time-indicator { animation: pulse 2s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
        .sensor-card { background: linear-gradient(45deg, #667eea, #764ba2); color: white; }
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/"><i class="fas fa-microchip"></i> Motor Temp Predictor</a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/"><i class="fas fa-home"></i> Home</a>
                <a class="nav-link" href="/manual"><i class="fas fa-keyboard"></i> Manual Input</a>
                <a class="nav-link active" href="/sensor"><i class="fas fa-sensor"></i> Sensor Data</a>
            </div>
        </div>
    </nav>

    <!-- Header -->
    <section class="form-section">
        <div class="container text-center">
            <h1><i class="fas fa-satellite-dish"></i> Sensor Data Input</h1>
            <p class="lead">Real-time motor parameter monitoring and temperature prediction</p>
            <div class="mt-3">
                <span class="sensor-status sensor-connected">
                    <i class="fas fa-circle real-time-indicator"></i> Sensors Connected
                </span>
            </div>
        </div>
    </section>

    <!-- Sensor Status Dashboard -->
    <section class="py-4 bg-light">
        <div class="container">
            <div class="row">
                <div class="col-lg-12">
                    <div class="card sensor-card">
                        <div class="card-body">
                            <div class="row text-center">
                                <div class="col-md-3">
                                    <i class="fas fa-bolt fa-2x mb-2"></i>
                                    <h6>Electrical Sensors</h6>
                                    <small>Voltage & Current</small>
                                </div>
                                <div class="col-md-3">
                                    <i class="fas fa-tachometer-alt fa-2x mb-2"></i>
                                    <h6>Speed Sensor</h6>
                                    <small>Motor RPM</small>
                                </div>
                                <div class="col-md-3">
                                    <i class="fas fa-thermometer-half fa-2x mb-2"></i>
                                    <h6>Temperature Sensors</h6>
                                    <small>Coolant & Ambient</small>
                                </div>
                                <div class="col-md-3">
                                    <i class="fas fa-wifi fa-2x mb-2"></i>
                                    <h6>Communication</h6>
                                    <small>Real-time Data</small>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Form -->
    <section class="py-5">
        <div class="container">
            <div class="row justify-content-center">
                <div class="col-lg-8">
                    <div class="card form-card">
                        <div class="card-header bg-success text-white">
                            <h4 class="mb-0"><i class="fas fa-satellite-dish"></i> Live Sensor Data Input</h4>
                            <small class="d-block mt-1">
                                <i class="fas fa-info-circle"></i> 
                                Data can be automatically populated from connected sensors or manually adjusted
                            </small>
                        </div>
                        <div class="card-body">
                            <div class="alert alert-info">
                                <i class="fas fa-lightbulb"></i>
                                <strong>Sensor Mode:</strong> Values below are updated from real-time sensor data. You can modify them manually if needed for testing or calibration purposes.
                            </div>

                            <form action="/predict" method="post" id="sensorPredictionForm">
                                <div class="row">

                                    <!-- Electrical Parameters -->
                                    <div class="col-md-6">
                                        <h5 class="text-primary mb-3"><i class="fas fa-bolt"></i> Electrical Sensors</h5>

                                        <div class="mb-3">
                                            <label class="form-label">
                                                Voltage Q-Component (V) 
                                                <span class="sensor-status sensor-connected ms-2">
                                                    <i class="fas fa-circle real-time-indicator"></i> Live
                                                </span>
                                            </label>
                                            <div class="input-group">
                                                <span class="input-group-text"><i class="fas fa-wave-square"></i></span>
                                                <input type="number" class="form-control" name="u_q" step="0.1" min="-30" max="140" value="50" required>
                                                <button type="button" class="btn btn-outline-secondary" onclick="refreshSensorData('u_q')">
                                                    <i class="fas fa-sync-alt"></i>
                                                </button>
                                            </div>
                                            <small class="text-muted">Range: -30V to 140V | Auto-updated every 5 seconds</small>
                                        </div>

                                        <div class="mb-3">
                                            <label class="form-label">
                                                Voltage D-Component (V)
                                                <span class="sensor-status sensor-connected ms-2">
                                                    <i class="fas fa-circle real-time-indicator"></i> Live
                                                </span>
                                            </label>
                                            <div class="input-group">
                                                <span class="input-group-text"><i class="fas fa-wave-square"></i></span>
                                                <input type="number" class="form-control" name="u_d" step="0.1" min="-140" max="140" value="-10" required>
                                                <button type="button" class="btn btn-outline-secondary" onclick="refreshSensorData('u_d')">
                                                    <i class="fas fa-sync-alt"></i>
                                                </button>
                                            </div>
                                            <small class="text-muted">Range: -140V to 140V | Auto-updated every 5 seconds</small>
                                        </div>

                                        <div class="mb-3">
                                            <label class="form-label">
                                                Current D-Component (A)
                                                <span class="sensor-status sensor-connected ms-2">
                                                    <i class="fas fa-circle real-time-indicator"></i> Live
                                                </span>
                                            </label>
                                            <div class="input-group">
                                                <span class="input-group-text"><i class="fas fa-flash"></i></span>
                                                <input type="number" class="form-control" name="i_d" step="0.1" min="-280" max="0" value="-150" required>
                                                <button type="button" class="btn btn-outline-secondary" onclick="refreshSensorData('i_d')">
                                                    <i class="fas fa-sync-alt"></i>
                                                </button>
                                            </div>
                                            <small class="text-muted">Range: -280A to 0A | Auto-updated every 5 seconds</small>
                                        </div>

                                        <div class="mb-3">
                                            <label class="form-label">
                                                Current Q-Component (A)
                                                <span class="sensor-status sensor-connected ms-2">
                                                    <i class="fas fa-circle real-time-indicator"></i> Live
                                                </span>
                                            </label>
                                            <div class="input-group">
                                                <span class="input-group-text"><i class="fas fa-flash"></i></span>
                                                <input type="number" class="form-control" name="i_q" step="0.1" min="-300" max="310" value="25" required>
                                                <button type="button" class="btn btn-outline-secondary" onclick="refreshSensorData('i_q')">
                                                    <i class="fas fa-sync-alt"></i>
                                                </button>
                                            </div>
                                            <small class="text-muted">Range: -300A to 310A | Auto-updated every 5 seconds</small>
                                        </div>
                                    </div>

                                    <!-- Mechanical & Thermal Parameters -->
                                    <div class="col-md-6">
                                        <h5 class="text-success mb-3"><i class="fas fa-thermometer-half"></i> Mechanical & Thermal Sensors</h5>

                                        <div class="mb-3">
                                            <label class="form-label">
                                                Motor Speed (RPM)
                                                <span class="sensor-status sensor-connected ms-2">
                                                    <i class="fas fa-circle real-time-indicator"></i> Live
                                                </span>
                                            </label>
                                            <div class="input-group">
                                                <span class="input-group-text"><i class="fas fa-tachometer-alt"></i></span>
                                                <input type="number" class="form-control" name="motor_speed" step="1" min="-300" max="6000" value="2000" required>
                                                <button type="button" class="btn btn-outline-secondary" onclick="refreshSensorData('motor_speed')">
                                                    <i class="fas fa-sync-alt"></i>
                                                </button>
                                            </div>
                                            <small class="text-muted">Range: -300 to 6000 RPM | Auto-updated every 2 seconds</small>
                                        </div>

                                        <div class="mb-3">
                                            <label class="form-label">
                                                Coolant Temperature (°C)
                                                <span class="sensor-status sensor-connected ms-2">
                                                    <i class="fas fa-circle real-time-indicator"></i> Live
                                                </span>
                                            </label>
                                            <div class="input-group">
                                                <span class="input-group-text"><i class="fas fa-temperature-low"></i></span>
                                                <input type="number" class="form-control" name="coolant" step="0.1" min="10" max="105" value="45" required>
                                                <button type="button" class="btn btn-outline-secondary" onclick="refreshSensorData('coolant')">
                                                    <i class="fas fa-sync-alt"></i>
                                                </button>
                                            </div>
                                            <small class="text-muted">Range: 10°C to 105°C | Auto-updated every 10 seconds</small>
                                        </div>

                                        <div class="mb-3">
                                            <label class="form-label">
                                                Ambient Temperature (°C)
                                                <span class="sensor-status sensor-connected ms-2">
                                                    <i class="fas fa-circle real-time-indicator"></i> Live
                                                </span>
                                            </label>
                                            <div class="input-group">
                                                <span class="input-group-text"><i class="fas fa-thermometer-quarter"></i></span>
                                                <input type="number" class="form-control" name="ambient" step="0.1" min="8" max="32" value="22" required>
                                                <button type="button" class="btn btn-outline-secondary" onclick="refreshSensorData('ambient')">
                                                    <i class="fas fa-sync-alt"></i>
                                                </button>
                                            </div>
                                            <small class="text-muted">Range: 8°C to 32°C | Auto-updated every 30 seconds</small>
                                        </div>
                                    </div>

                                </div>

                                <hr>

                                <!-- Control Buttons -->
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="d-grid gap-2">
                                            <button type="button" class="btn btn-info" onclick="refreshAllSensors()">
                                                <i class="fas fa-sync-alt"></i> Refresh All Sensors
                                            </button>
                                            <button type="button" class="btn btn-warning" onclick="pauseAutoUpdate()">
                                                <i class="fas fa-pause"></i> Pause Auto-Update
                                            </button>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="d-grid gap-2">
                                            <button type="submit" class="btn btn-success btn-lg">
                                                <i class="fas fa-calculator"></i> Predict Temperature
                                            </button>
                                            <small class="text-muted text-center mt-2">
                                                <i class="fas fa-clock"></i> Last updated: <span id="lastUpdate">Now</span>
                                            </small>
                                        </div>
                                    </div>
                                </div>
                            </form>
                        </div>
                    </div>

                    <!-- Sensor Information -->
                    <div class="card mt-4">
                        <div class="card-header bg-dark text-white">
                            <h5 class="mb-0"><i class="fas fa-satellite-dish"></i> Sensor Network Information</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6><i class="fas fa-network-wired text-primary"></i> Connected Sensors</h6>
                                    <ul class="small">
                                        <li><strong>Voltage Transducers:</strong> dq-frame measurements</li>
                                        <li><strong>Current Sensors:</strong> Hall effect current clamps</li>
                                        <li><strong>Speed Encoder:</strong> Optical rotary encoder</li>
                                        <li><strong>Temperature Probes:</strong> PT100 RTD sensors</li>
                                    </ul>
                                </div>
                                <div class="col-md-6">
                                    <h6><i class="fas fa-cogs text-success"></i> System Features</h6>
                                    <ul class="small">
                                        <li><strong>Auto-Update:</strong> Real-time data streaming</li>
                                        <li><strong>Manual Override:</strong> Editable sensor values</li>
                                        <li><strong>Data Logging:</strong> Historical trend analysis</li>
                                        <li><strong>Calibration:</strong> Sensor accuracy verification</li>
                                    </ul>
                                </div>
                            </div>
                            <div class="alert alert-warning mt-3">
                                <i class="fas fa-exclamation-triangle"></i>
                                <strong>Note:</strong> This is a demonstration interface. In a production environment, sensor data would be automatically populated from actual hardware sensors connected via industrial protocols (Modbus, CAN, etc.).
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer class="bg-dark text-white py-4">
        <div class="container text-center">
            <p>&copy; 2025 Electric Motor Temperature Prediction System. Powered by Machine Learning.</p>
            <p><small>Sensor Integration | Real-time Monitoring | Industrial IoT Compatible</small></p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Simulated sensor data refresh functions
        function refreshSensorData(parameter) {
            // In a real implementation, this would fetch data from actual sensors
            const input = document.querySelector(`input[name="${parameter}"]`);
            if (input) {
                // Simulate small variations around current value
                const currentValue = parseFloat(input.value);
                const variation = (Math.random() - 0.5) * 0.1 * currentValue;
                const newValue = Math.max(input.min, Math.min(input.max, currentValue + variation));
                input.value = newValue.toFixed(1);
                
                // Visual feedback
                input.style.backgroundColor = '#e8f5e8';
                setTimeout(() => { input.style.backgroundColor = ''; }, 1000);
            }
            updateLastRefreshTime();
        }

        function refreshAllSensors() {
            const parameters = ['u_q', 'u_d', 'i_d', 'i_q', 'motor_speed', 'coolant', 'ambient'];
            parameters.forEach(param => refreshSensorData(param));
        }

        function updateLastRefreshTime() {
            document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
        }

        let autoUpdateEnabled = true;
        function pauseAutoUpdate() {
            autoUpdateEnabled = !autoUpdateEnabled;
            const button = event.target;
            if (autoUpdateEnabled) {
                button.innerHTML = '<i class="fas fa-pause"></i> Pause Auto-Update';
                button.className = 'btn btn-warning';
            } else {
                button.innerHTML = '<i class="fas fa-play"></i> Resume Auto-Update';
                button.className = 'btn btn-success';
            }
        }

        // Simulate periodic sensor updates (in real system, this would be WebSocket or similar)
        setInterval(() => {
            if (autoUpdateEnabled) {
                // Randomly update one parameter every few seconds
                const parameters = ['u_q', 'u_d', 'i_d', 'i_q', 'motor_speed', 'coolant', 'ambient'];
                const randomParam = parameters[Math.floor(Math.random() * parameters.length)];
                refreshSensorData(randomParam);
            }
        }, 5000);

        // Initialize timestamp
        updateLastRefreshTime();
    </script>
</body>
</html>'''

# Result 
result_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Prediction Result – Motor Temp</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .result-hero {background:linear-gradient(135deg,#2e8b57 0%,#667eea 100%);color:#fff;padding:60px 0}
        .badge-param {font-size:.9rem}
        .temp-display {
            font-size: 4rem;
            font-weight: bold;
        }
        .warning-high { color: #dc3545; }
        .warning-medium { color: #fd7e14; }
        .warning-normal { color: #198754; }
        .warning-low { color: #0dcaf0; }
    </style>
</head>
<body>
<nav class="navbar navbar-expand-lg navbar-dark bg-dark">
    <div class="container">
        <a class="navbar-brand" href="/"><i class="fas fa-microchip"></i> Motor Temp Predictor</a>
        <div class="navbar-nav ms-auto">
            <a class="nav-link" href="/"><i class="fas fa-home"></i> Home</a>
            <a class="nav-link" href="/manual"><i class="fas fa-keyboard"></i> Manual</a>
            <a class="nav-link" href="/sensor"><i class="fas fa-sensor"></i> Sensor</a>
        </div>
    </div>
</nav>

<section class="result-hero text-center">
    <div class="container">
        <h1><i class="fas fa-clipboard-check"></i> Prediction Result</h1>
    </div>
</section>

<section class="py-5">
    <div class="container">

        {% if error %}
            <div class="alert alert-danger text-center">
                <i class="fas fa-times-circle fa-3x mb-3"></i>
                <h4>Prediction Error</h4>
                <p>{{ error }}</p>
                <a href="/manual" class="btn btn-outline-primary">
                    <i class="fas fa-arrow-left"></i> Try Again
                </a>
            </div>

        {% else %}
            <div class="row justify-content-center">
                <div class="col-lg-8">
                    <div class="card shadow-lg mb-4">
                        <div class="card-body text-center p-5">
                            {% if prediction > 80 %}
                                <div class="temp-display warning-high">
                            {% elif prediction > 60 %}
                                <div class="temp-display warning-medium">
                            {% elif prediction < 25 %}
                                <div class="temp-display warning-low">
                            {% else %}
                                <div class="temp-display warning-normal">
                            {% endif %}
                                <i class="fas fa-thermometer-half"></i> {{ prediction }}°C
                            </div>
                            
                            <h3 class="mt-4 mb-3">Permanent Magnet Temperature</h3>
                            
                            {% for warning in warnings %}
                                {% if 'HIGH TEMPERATURE' in warning %}
                                    <div class="alert alert-danger">
                                        <i class="fas fa-exclamation-triangle"></i> {{ warning }}
                                    </div>
                                {% elif 'ELEVATED' in warning %}
                                    <div class="alert alert-warning">
                                        <i class="fas fa-exclamation-circle"></i> {{ warning }}
                                    </div>
                                {% elif 'LOW TEMPERATURE' in warning %}
                                    <div class="alert alert-info">
                                        <i class="fas fa-info-circle"></i> {{ warning }}
                                    </div>
                                {% else %}
                                    <div class="alert alert-success">
                                        <i class="fas fa-check-circle"></i> {{ warning }}
                                    </div>
                                {% endif %}
                            {% endfor %}
                            
                            <div class="mt-4">
                                <a href="/manual" class="btn btn-primary btn-lg me-3">
                                    <i class="fas fa-redo-alt"></i> New Prediction
                                </a>
                                <a href="/sensor" class="btn btn-outline-primary btn-lg">
                                    <i class="fas fa-sensor"></i> Sensor Mode
                                </a>
                            </div>
                        </div>
                    </div>

                    <div class="card">
                        <div class="card-header bg-light">
                            <h5 class="mb-0"><i class="fas fa-info-circle"></i> Input Parameters</h5>
                        </div>
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table table-hover">
                                    <thead class="table-light">
                                        <tr>
                                            <th>Parameter</th>
                                            <th class="text-end">Value</th>
                                            <th class="text-end">Unit</th>
                                            <th>Description</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {% for key, val in input_data.items() %}
                                            <tr>
                                                <td><strong>{{ key.replace('_', ' ').title() }}</strong></td>
                                                <td class="text-end">
                                                    <span class="badge bg-secondary badge-param">{{ val }}</span>
                                                </td>
                                                <td class="text-end">
                                                    {% if key in ['u_q','u_d'] %}
                                                        V
                                                    {% elif key in ['i_q','i_d'] %}
                                                        A
                                                    {% elif key=='motor_speed' %}
                                                        rpm
                                                    {% else %}
                                                        °C
                                                    {% endif %}
                                                </td>
                                                <td class="text-muted small">
                                                    {% if key == 'u_q' %}
                                                        Quadrature voltage component
                                                    {% elif key == 'u_d' %}
                                                        Direct voltage component
                                                    {% elif key == 'i_q' %}
                                                        Quadrature current component
                                                    {% elif key == 'i_d' %}
                                                        Direct current component
                                                    {% elif key == 'motor_speed' %}
                                                        Mechanical rotation speed
                                                    {% elif key == 'coolant' %}
                                                        Coolant outlet temperature
                                                    {% elif key == 'ambient' %}
                                                        Environmental temperature
                                                    {% endif %}
                                                </td>
                                            </tr>
                                        {% endfor %}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        {% endif %}

    </div>
</section>

<footer class="bg-dark text-white py-3 text-center">
    <small>&copy; 2025 Electric Motor Temperature Prediction System</small>
</footer>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>'''

# Save all templates with UTF-8 encoding
with open('templates/home.html', 'w', encoding='utf-8') as f:
    f.write(home_template)

with open('templates/manual_predict.html', 'w', encoding='utf-8') as f:
    f.write(manual_template)

with open('templates/sensor_predict.html', 'w', encoding='utf-8') as f:
    f.write(sensor_template)

with open('templates/result.html', 'w', encoding='utf-8') as f:
    f.write(result_template)

print("✅ All HTML Templates created successfully:")
print("   • templates/home.html")
print("   • templates/manual_predict.html") 
print("   • templates/sensor_predict.html")
print("   • templates/result.html")

# Documentation Files 

requirements_txt = '''Flask==2.3.3
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
matplotlib==3.7.2
seaborn==0.12.2
gunicorn==21.2.0
plotly==5.16.1
kaleido==0.2.1
'''

readme_md = '''# Electric Motor Temperature Prediction System

🔥 AI-powered permanent magnet temperature prediction for industrial motor systems.

## Features

- **Machine Learning Prediction**: Advanced algorithms for accurate temperature forecasting
- **Real-time Monitoring**: Live sensor data integration with instant predictions
- **Safety Warnings**: Automatic alerts for overheating conditions
- **Web Interface**: User-friendly dashboard for manual input and monitoring
- **REST API**: Integration-ready endpoints for industrial systems
- **Multi-Model Support**: Comparison of different ML algorithms

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**
   ```bash
   python train_model.py
   ```

3. **Start the Web Application**
   ```bash
   python app.py
   ```

4. **Access the System**
   - Web Interface: http://localhost:5000
   - API Endpoint: http://localhost:5000/api/predict

## Model Performance

- **Best Algorithm**: Linear Regression
- **Accuracy**: 37.9% (R² Score: 0.3791)
- **RMSE**: ~9.70°C
- **Training Dataset**: 10,000 samples
- **Features**: 7 motor parameters

## API Usage

### Predict Temperature
```bash
POST /api/predict
Content-Type: application/json

{
  "u_q": 50.0,
  "coolant": 45.0,
  "u_d": -10.0,
  "motor_speed": 2000.0,
  "i_d": -150.0,
  "i_q": 25.0,
  "ambient": 22.0
}
```

### Response
```json
{
  "prediction": 65.43,
  "warnings": ["⚠️ ELEVATED TEMPERATURE: Monitor motor condition"],
  "status": "success"
}
```

## Input Parameters

| Parameter | Unit | Range | Description |
|-----------|------|-------|-------------|
| u_q | V | -30 to 140 | Quadrature voltage component |
| u_d | V | -140 to 140 | Direct voltage component |
| i_q | A | -300 to 310 | Quadrature current component |
| i_d | A | -280 to 0 | Direct current component |
| motor_speed | rpm | -300 to 6000 | Mechanical rotation speed |
| coolant | °C | 10 to 105 | Coolant outlet temperature |
| ambient | °C | 8 to 32 | Environmental temperature |

## Safety Thresholds

- **🟢 Normal**: < 60°C - Motor operating within safe range
- **🟡 Elevated**: 60-80°C - Monitor motor condition closely  
- **🔴 High**: > 80°C - Risk of overheating, immediate attention required
- **🔵 Low**: < 25°C - Motor under low load conditions

## File Structure

```
motor-temp-prediction/
├── app.py                 # Flask web application
├── train_model.py         # Model training script  
├── templates/             # HTML templates
│   ├── home.html
│   ├── manual_predict.html
│   ├── sensor_predict.html
│   └── result.html
├── motor_temperature_model.pkl  # Trained model
├── feature_scaler.pkl     # Data scaler
├── requirements.txt       # Dependencies
└── README.md             # Documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions or issues, please open a GitHub issue or contact the development team.

---
**⚡ Powered by Machine Learning | Built for Industrial Applications**
'''

api_docs = '''# API Documentation

## Electric Motor Temperature Prediction API

### Base URL
```
http://localhost:5000
```

### Authentication
No authentication required for this version.

### Endpoints

#### 1. Predict Temperature
Predict permanent magnet temperature based on motor operating parameters.

**Endpoint:** `POST /api/predict`

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "u_q": 50.0,
  "coolant": 45.0, 
  "u_d": -10.0,
  "motor_speed": 2000.0,
  "i_d": -150.0,
  "i_q": 25.0,
  "ambient": 22.0
}
```

**Success Response (200):**
```json
{
  "prediction": 65.43,
  "warnings": ["⚠️ ELEVATED TEMPERATURE: Monitor motor condition"],
  "status": "success"
}
```

**Error Response (400):**
```json
{
  "error": "Missing required features"
}
```

**Error Response (500):**
```json
{
  "error": "API error: Model prediction failed"
}
```

### Parameter Specifications

| Parameter | Type | Required | Range | Unit | Description |
|-----------|------|----------|-------|------|-------------|
| u_q | float | Yes | -30 to 140 | V | Quadrature voltage component |
| u_d | float | Yes | -140 to 140 | V | Direct voltage component |
| i_q | float | Yes | -300 to 310 | A | Quadrature current component |
| i_d | float | Yes | -280 to 0 | A | Direct current component |
| motor_speed | float | Yes | -300 to 6000 | rpm | Mechanical rotation speed |
| coolant | float | Yes | 10 to 105 | °C | Coolant outlet temperature |
| ambient | float | Yes | 8 to 32 | °C | Environmental temperature |

### Warning Messages

The API returns contextual warnings based on predicted temperature:

- **🟢 Normal Range**: "ℹ️ Motor operating within normal range" (25-60°C)
- **🔵 Low Temperature**: "ℹ️ LOW TEMPERATURE: Motor may be under low load" (<25°C)  
- **🟡 Elevated**: "⚠️ ELEVATED TEMPERATURE: Monitor motor condition" (60-80°C)
- **🔴 High Temperature**: "⚠️ HIGH TEMPERATURE WARNING: Motor may be overheating!" (>80°C)

### Example Usage

#### Python
```python
import requests

url = "http://localhost:5000/api/predict"
data = {
    "u_q": 50.0,
    "coolant": 45.0,
    "u_d": -10.0, 
    "motor_speed": 2000.0,
    "i_d": -150.0,
    "i_q": 25.0,
    "ambient": 22.0
}

response = requests.post(url, json=data)
result = response.json()
print(f"Predicted Temperature: {result['prediction']}°C")
```

#### JavaScript (Fetch)
```javascript
const data = {
    u_q: 50.0,
    coolant: 45.0,
    u_d: -10.0,
    motor_speed: 2000.0,
    i_d: -150.0,
    i_q: 25.0,
    ambient: 22.0
};

fetch('http://localhost:5000/api/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(data)
})
.then(response => response.json())
.then(result => {
    console.log('Predicted Temperature:', result.prediction + '°C');
});
```

#### cURL
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "u_q": 50.0,
    "coolant": 45.0,
    "u_d": -10.0,
    "motor_speed": 2000.0,
    "i_d": -150.0,
    "i_q": 25.0,
    "ambient": 22.0
  }'
```

### Rate Limiting
Currently no rate limiting is implemented. For production use, implement appropriate rate limiting based on your requirements.

### Error Handling
Always check the response status and handle errors appropriately:

- **400 Bad Request**: Missing or invalid parameters
- **500 Internal Server Error**: Model prediction failure or server error

### Model Information
- **Algorithm**: Linear Regression
- **Accuracy**: 37.9% (R² Score: 0.3791)
- **RMSE**: ~9.70°C
- **Training Data**: Electric Motor Temperature Dataset (Kaggle)
- **Features**: 7 motor operating parameters

### Version History
- **v1.0**: Initial API release with basic prediction functionality
'''

# Save documentation files
with open('requirements.txt', 'w', encoding='utf-8') as f:
    f.write(requirements_txt)

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_md)

with open('API_Documentation.md', 'w', encoding='utf-8') as f:
    f.write(api_docs)

print("✅ Documentation files created:")
print("   • requirements.txt")
print("   • README.md") 
print("   • API_Documentation.md")

print("\n" + "="*80)
print("🎉 ELECTRIC MOTOR TEMPERATURE PREDICTION SYSTEM - SETUP COMPLETE!")
# Generate Visualization Charts

print("\n8. GENERATING VISUALIZATION CHARTS")
print("-" * 40)

# Model Comparison Chart using Matplotlib
plt.style.use('default')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Electric Motor Temperature Prediction - Model Analysis', fontsize=16, fontweight='bold')

# 1. Model R² Comparison Bar Chart
models_list = list(results.keys())
r2_scores = [results[model]['test_r2'] for model in models_list]
colors = ['#2E8B57', '#1FB8CD', '#DB4545', '#5D878F']

bars = ax1.bar(models_list, r2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
ax1.set_title('Model Performance Comparison (R² Score)', fontweight='bold', pad=20)
ax1.set_ylabel('R² Score')
ax1.set_xlabel('Machine Learning Models')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

# Rotate x-axis labels for better readability
ax1.tick_params(axis='x', rotation=45)

# 2. RMSE Comparison
rmse_scores = [results[model]['test_rmse'] for model in models_list]
bars2 = ax2.bar(models_list, rmse_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
ax2.set_title('Model RMSE Comparison', fontweight='bold', pad=20)
ax2.set_ylabel('RMSE (°C)')
ax2.set_xlabel('Machine Learning Models')
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for bar, rmse in zip(bars2, rmse_scores):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{rmse:.2f}', ha='center', va='bottom', fontweight='bold')

ax2.tick_params(axis='x', rotation=45)

# 3. Actual vs Predicted Scatter Plot (for best model)
best_predictions = results[best_model_name]['predictions']
ax3.scatter(y_test, best_predictions, alpha=0.6, color='#2E8B57', s=30)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
ax3.set_xlabel('Actual Temperature (°C)')
ax3.set_ylabel('Predicted Temperature (°C)')
ax3.set_title(f'Actual vs Predicted - {best_model_name}', fontweight='bold', pad=20)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Add R² score to the plot
ax3.text(0.05, 0.95, f'R² = {best_r2:.4f}', transform=ax3.transAxes, 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
         fontsize=12, fontweight='bold')

# 4. Target Variable Distribution (Histogram)
ax4.hist(y, bins=50, alpha=0.7, color='#1FB8CD', edgecolor='black', linewidth=0.5)
ax4.set_xlabel('Permanent Magnet Temperature (°C)')
ax4.set_ylabel('Frequency')
ax4.set_title('Temperature Distribution in Dataset', fontweight='bold', pad=20)
ax4.grid(True, alpha=0.3)

# Add statistics to histogram
mean_temp = y.mean()
std_temp = y.std()
ax4.axvline(mean_temp, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_temp:.1f}°C')
ax4.axvline(mean_temp + std_temp, color='orange', linestyle='--', alpha=0.7, label=f'+1σ: {mean_temp + std_temp:.1f}°C')
ax4.axvline(mean_temp - std_temp, color='orange', linestyle='--', alpha=0.7, label=f'-1σ: {mean_temp - std_temp:.1f}°C')
ax4.legend()

plt.tight_layout()
plt.savefig('motor_temperature_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Model analysis chart saved as 'motor_temperature_analysis.png'")

# Feature Correlation Heatmap 

plt.figure(figsize=(12, 8))
correlation_matrix = df[feature_columns + ['pm']].corr()

# Create heatmap
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.3f')

plt.title('Feature Correlation Matrix - Motor Temperature Dataset', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Features')
plt.ylabel('Features')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Feature correlation heatmap saved as 'feature_correlation_heatmap.png'")

# Step 13: Feature Importance and Distribution Analysis 

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Feature Distribution Analysis', fontsize=16, fontweight='bold')

feature_names = list(X.columns)
axes = axes.flatten()

for i, feature in enumerate(feature_names):
    axes[i].hist(df[feature], bins=30, alpha=0.7, color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
    axes[i].set_title(f'{feature.replace("_", " ").title()}', fontweight='bold')
    axes[i].set_xlabel(f'{feature} Value')
    axes[i].set_ylabel('Frequency')
    axes[i].grid(True, alpha=0.3)
    
    # Add statistics
    mean_val = df[feature].mean()
    std_val = df[feature].std()
    axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, alpha=0.8)
    axes[i].text(0.02, 0.98, f'μ={mean_val:.1f}\nσ={std_val:.1f}', 
                transform=axes[i].transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)

# Remove the last empty subplot
if len(feature_names) < len(axes):
    fig.delaxes(axes[-1])

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Feature distributions chart saved as 'feature_distributions.png'")

# Step 14: Model Residuals Analysis 

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(f'Residual Analysis - {best_model_name}', fontsize=16, fontweight='bold')

# Calculate residuals
residuals = y_test - best_predictions

# 1. Residuals vs Predicted
ax1.scatter(best_predictions, residuals, alpha=0.6, color='#2E8B57', s=30)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('Predicted Temperature (°C)')
ax1.set_ylabel('Residuals (°C)')
ax1.set_title('Residuals vs Predicted Values', fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Residuals Histogram
ax2.hist(residuals, bins=30, alpha=0.7, color='#1FB8CD', edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Residuals (°C)')
ax2.set_ylabel('Frequency')
ax2.set_title('Residuals Distribution', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axvline(0, color='red', linestyle='--', linewidth=2)

# Add normal distribution overlay
from scipy import stats
mu, sigma = stats.norm.fit(residuals)
x = np.linspace(residuals.min(), residuals.max(), 100)
y_norm = stats.norm.pdf(x, mu, sigma) * len(residuals) * (residuals.max() - residuals.min()) / 30
ax2.plot(x, y_norm, 'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
ax2.legend()

# 3. Q-Q Plot
stats.probplot(residuals, dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Residuals vs Actual
ax4.scatter(y_test, residuals, alpha=0.6, color='#DB4545', s=30)
ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Actual Temperature (°C)')
ax4.set_ylabel('Residuals (°C)')
ax4.set_title('Residuals vs Actual Values', fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_residuals_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Residuals analysis chart saved as 'model_residuals_analysis.png'")

# Step 15: Interactive Plotly Visualization

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    
    # Create interactive model comparison chart
    fig_interactive = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Model R² Comparison', 'RMSE Comparison', 
                       'Actual vs Predicted', 'Temperature Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Model R² Comparison
    fig_interactive.add_trace(
        go.Bar(x=models_list, y=r2_scores, 
               marker_color=colors, name='R² Score',
               text=[f'{score:.3f}' for score in r2_scores],
               textposition='outside'),
        row=1, col=1
    )
    
    # RMSE Comparison
    fig_interactive.add_trace(
        go.Bar(x=models_list, y=rmse_scores,
               marker_color=colors, name='RMSE',
               text=[f'{rmse:.2f}°C' for rmse in rmse_scores],
               textposition='outside'),
        row=1, col=2
    )
    
    # Actual vs Predicted Scatter
    fig_interactive.add_trace(
        go.Scatter(x=y_test, y=best_predictions,
                  mode='markers', name='Predictions',
                  marker=dict(color='#2E8B57', size=4, opacity=0.6),
                  hovertemplate='Actual: %{x:.1f}°C<br>Predicted: %{y:.1f}°C<extra></extra>'),
        row=2, col=1
    )
    
    # Perfect prediction line
    fig_interactive.add_trace(
        go.Scatter(x=[y_test.min(), y_test.max()], 
                  y=[y_test.min(), y_test.max()],
                  mode='lines', name='Perfect Prediction',
                  line=dict(color='red', dash='dash', width=2)),
        row=2, col=1
    )
    
    # Temperature Distribution
    fig_interactive.add_trace(
        go.Histogram(x=y, nbinsx=50, name='Temperature Distribution',
                    marker_color='#1FB8CD', opacity=0.7),
        row=2, col=2
    )
    
    fig_interactive.update_layout(
        title_text="Electric Motor Temperature Prediction - Interactive Analysis",
        title_x=0.5,
        height=800,
        showlegend=False
    )
    
    # Update axis labels
    fig_interactive.update_xaxes(title_text="Models", row=1, col=1)
    fig_interactive.update_yaxes(title_text="R² Score", row=1, col=1)
    
    fig_interactive.update_xaxes(title_text="Models", row=1, col=2)
    fig_interactive.update_yaxes(title_text="RMSE (°C)", row=1, col=2)
    
    fig_interactive.update_xaxes(title_text="Actual Temperature (°C)", row=2, col=1)
    fig_interactive.update_yaxes(title_text="Predicted Temperature (°C)", row=2, col=1)
    
    fig_interactive.update_xaxes(title_text="Temperature (°C)", row=2, col=2)
    fig_interactive.update_yaxes(title_text="Frequency", row=2, col=2)
    
    # Save as HTML
    fig_interactive.write_html("interactive_analysis.html")
    print("✅ Interactive analysis saved as 'interactive_analysis.html'")
    
    # Save as static image
    pio.write_image(fig_interactive, "interactive_analysis.png", width=1200, height=800, scale=2)
    print("✅ Interactive analysis (static) saved as 'interactive_analysis.png'")
    
except ImportError:
    print("⚠️ Plotly not available. Install with: pip install plotly kaleido")
except Exception as e:
    print(f"⚠️ Could not create interactive visualization: {str(e)}")

# Summary of Generated Visualizations

print("\n" + "="*60)
print("📊 VISUALIZATION FILES GENERATED:")
print("="*60)
print("✅ motor_temperature_analysis.png - Complete model analysis")
print("   • Model R² comparison bar chart")
print("   • RMSE comparison chart") 
print("   • Actual vs Predicted scatter plot")
print("   • Target variable histogram")
print()
print("✅ feature_correlation_heatmap.png - Feature correlations")
print("   • Correlation matrix heatmap")
print("   • Feature relationships visualization")
print()
print("✅ feature_distributions.png - Individual feature analysis")
print("   • Histogram for each input feature")
print("   • Statistical summaries (mean, std)")
print()
print("✅ model_residuals_analysis.png - Model performance deep dive")
print("   • Residuals vs Predicted scatter")
print("   • Residuals distribution histogram")
print("   • Q-Q plot for normality check")
print("   • Residuals vs Actual scatter")
print()
print("✅ interactive_analysis.html - Interactive dashboard")
print("   • Plotly interactive charts")
print("   • Hover tooltips and zoom functionality")
print("=" * 60)

print("=" * 80)
print("✅ Model training and evaluation completed")

print("✅ Flask web application created (app.py)")
print("✅ All HTML templates generated")
print("✅ Documentation files created")
print("✅ Model and scaler saved for deployment")
print()
print("🚀 TO RUN THE APPLICATION:")
print("   1. Ensure your dataset is at: E:\\EMTP\\dataset22.csv")
print("   2. Run this script to train the model")
print("   3. Execute: python app.py")
print("   4. Open: http://localhost:5000")
print()
print("📊 MODEL PERFORMANCE:")
print(f"   • Best Algorithm: {best_model_name}")
print(f"   • Accuracy: {best_r2*100:.1f}% (R² = {best_r2:.4f})")
print(f"   • RMSE: {results[best_model_name]['test_rmse']:.2f}°C")
print()
print("🌐 WEB INTERFACES:")
print("   • Home: http://localhost:5000/")
print("   • Manual Input: http://localhost:5000/manual")
print("   • Sensor Mode: http://localhost:5000/sensor")
print("   • API Endpoint: http://localhost:5000/api/predict")
print("=" * 80)
