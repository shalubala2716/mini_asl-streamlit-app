import streamlit as st
import numpy as np
import cv2
from PIL import Image

try:
    from tensorflow.keras.models import load_model
    tf_import_error = None
except Exception as e:
    load_model = None
    tf_import_error = e

MODEL_PATH = "best_asl_model.keras"
LABELS = ["A", "B"]

@st.cache_resource
def load_asl_model(path: str):
    if load_model is None:
        return None, "TensorFlow import failed. Please install tensorflow-cpu in requirements.txt."
    try:
        model = load_model(path)
        return model, None
    except FileNotFoundError:
        return None, f"Model file not found: {path}. Make sure the file is present in the repo."
    except Exception as e:
        return None, f"Failed to load model: {e}"

st.title("🧠 ASL Gesture Recognition System")
st.write("Upload an image to predict A or B")

if tf_import_error is not None:
    st.warning(
        "TensorFlow import failed on startup. The app UI is still available, but model prediction is disabled."
    )
    st.text(str(tf_import_error))

model, model_error = load_asl_model(MODEL_PATH)
if model_error is not None:
    st.warning(model_error)
    st.info("The app will still load the UI. Uploading images will show previews, but predictions are disabled until the model is fixed.")

uploaded_file = st.file_uploader("Upload Hand Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not decode the uploaded image. Please upload a valid JPG or PNG file.")
    else:
        st.image(img, channels="BGR", caption="Uploaded Image")

        if model is None:
            st.warning("Model is unavailable, so prediction cannot be performed.")
        else:
            img_resized = cv2.resize(img, (96, 96))
            img_norm = img_resized / 255.0
            img_input = np.expand_dims(img_norm, axis=0)

            try:
                pred = model.predict(img_input)
                class_id = np.argmax(pred)
                confidence = np.max(pred) * 100
                result = LABELS[class_id]

                st.markdown("## 🔍 Prediction Result")
                st.success(f"Gesture: {result}")
                st.info(f"Confidence: {confidence:.2f}%")
                st.bar_chart(pred[0])
            except Exception as err:
                st.error(f"Prediction failed: {err}")
else:
    st.info("Upload an image to preview it. If the model is available, prediction will appear below.")
