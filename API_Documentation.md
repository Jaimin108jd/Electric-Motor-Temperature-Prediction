# API Documentation

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
curl -X POST http://localhost:5000/api/predict   -H "Content-Type: application/json"   -d '{
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
