# Electric Motor Temperature Prediction System

🔥 permanent magnet temperature prediction for industrial motor systems.

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
MAIL:jaimin108jd@gmail.com