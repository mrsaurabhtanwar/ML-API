# Student Performance Prediction API

A FastAPI application that predicts student performance based on quiz interaction data using machine learning.

## Render Deployment

This app is configured for easy deployment on Render.

### Deployment Steps

1. **Connect Repository**: Connect your GitHub repository to Render
2. **Configure Service**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3.13.3 (auto-detected from runtime.txt)

3. **Deploy**: Render will automatically build and deploy your service

### API Endpoints

- `GET /`: Welcome message
- `POST /predict`: Complete prediction with behavioral analysis  
- `POST /analyze`: Behavioral analysis without ML prediction
- `GET /health`: Health check endpoint
- `GET /docs`: Interactive API documentation

### Usage Example

```bash
curl -X POST "https://your-render-app.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "hint_count": 3,
       "bottom_hint": 1,
       "attempt_count": 2,
       "ms_first_response": 5000,
       "duration": 120,
       "avg_conf_frustrated": 0.3,
       "avg_conf_confused": 0.2,
       "avg_conf_concentrating": 0.7,
       "avg_conf_bored": 0.1
     }'
```

### Files Structure

```
ML-API/
├── app.py                # Main FastAPI application
├── student_model.pkl     # Trained ML model
├── requirements.txt      # Python dependencies
├── runtime.txt          # Python version for Render
├── build.sh             # Build script (optional)
└── README.md           # This file
```

### Support

- Health check: `GET /health`
- API documentation: `GET /docs`
- Alternative docs: `GET /redoc`
