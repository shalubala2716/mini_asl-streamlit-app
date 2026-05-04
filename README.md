# ASL Gesture Recognition App

This is a Streamlit frontend + FastAPI backend architecture for ASL gesture recognition.

## Architecture

- **Frontend**: Streamlit app (UI only, no ML dependencies)
- **Backend**: FastAPI service with TensorFlow model inference

## Deployment

### Frontend (Streamlit Cloud)

1. Deploy the Streamlit app to Streamlit Cloud
2. Files needed:
   - `app.py`
   - `requirements.txt`
   - `runtime.txt` (optional, may not work on Streamlit Cloud)

### Backend (Separate Service)

1. Deploy the backend to a platform that supports Python 3.11+ and TensorFlow:
   - Railway
   - Render
   - Heroku
   - DigitalOcean App Platform
   - AWS/GCP/Azure

2. Files needed:
   - `backend.py`
   - `backend_requirements.txt`
   - `best_asl_model.keras`

3. Set environment variables:
   - `PORT=8000` (or whatever your platform uses)

4. Update `app.py` with your backend URL:
   ```python
   BACKEND_URL = "https://your-deployed-backend-url.com/predict"
   ```

## Local Development

### Frontend
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Backend
```bash
pip install -r backend_requirements.txt
python backend.py
```

## API Endpoints

### POST /predict
- Input: `{"image": "base64_encoded_jpeg"}`
- Output: `{"gesture": "A", "confidence": 95.2, "probabilities": [0.952, 0.048]}`

### GET /health
- Check if backend and model are healthy

## Why This Architecture?

- Streamlit Cloud has Python version limitations
- TensorFlow requires specific Python versions
- Separating concerns allows each service to use optimal environments
- Frontend can deploy reliably without ML dependencies