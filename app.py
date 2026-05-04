import streamlit as st
import numpy as np
import cv2
from PIL import Image

MODEL_PATH = "best_asl_model.keras"
LABELS = ["A", "B"]

@st.cache_resource
def load_asl_model(path: str):
    try:
        from tensorflow.keras.models import load_model
    except Exception as exc:
        raise RuntimeError(
            f"TensorFlow import failed: {exc}.\n" \
            "Use tensorflow-cpu in requirements.txt and a supported Python version."
        ) from exc

    try:
        return load_model(path)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Model file not found: {path}. Make sure the file is deployed with the app."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load model: {exc}") from exc

def get_model():
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.model_error = None
        st.session_state.model = None

    if not st.session_state.model_loaded:
        try:
            st.session_state.model = load_asl_model(MODEL_PATH)
            st.session_state.model_error = None
        except RuntimeError as exc:
            st.session_state.model = None
            st.session_state.model_error = str(exc)
        finally:
            st.session_state.model_loaded = True

    return st.session_state.model, st.session_state.model_error

st.set_page_config(page_title="ASL Gesture Recognition System", layout="wide")
st.title("🧠 ASL Gesture Recognition System")
st.write("Upload an image to predict A or B")

uploaded_file = st.file_uploader("Upload Hand Image", type=["jpg", "png", "jpeg"])

if uploaded_file is None:
    st.info("Upload an image to preview it. If the model is available, prediction will appear after upload.")
else:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not decode the uploaded image. Please upload a valid JPG or PNG file.")
    else:
        st.image(img, channels="BGR", caption="Uploaded Image")

        model, model_error = get_model()

        if model_error is not None:
            st.warning(model_error)
            st.info("The app UI is working, but prediction is disabled until the model is fixed.")
        elif model is None:
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
                st.info("The UI is still available; try again or fix the model/runtime.")

