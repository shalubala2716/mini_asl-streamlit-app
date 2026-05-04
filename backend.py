from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from PIL import Image
import base64
import io
from tensorflow.keras.models import load_model

app = FastAPI(title="ASL Gesture Recognition Backend")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Streamlit Cloud URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "best_asl_model.keras"
LABELS = ["A", "B"]

# Load model on startup
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

@app.post("/predict")
async def predict_gesture(data: dict):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        # Decode base64 image
        img_data = base64.b64decode(data["image"])
        img = Image.open(io.BytesIO(img_data))
        img_array = np.array(img)

        # Convert RGB to BGR if needed
        if img_array.shape[-1] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Preprocess
        img_resized = cv2.resize(img_array, (96, 96))
        img_norm = img_resized / 255.0
        img_input = np.expand_dims(img_norm, axis=0)

        # Predict
        pred = model.predict(img_input)
        class_id = np.argmax(pred)
        confidence = float(np.max(pred) * 100)
        gesture = LABELS[class_id]

        return {
            "gesture": gesture,
            "confidence": confidence,
            "probabilities": pred[0].tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)