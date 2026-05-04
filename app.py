import streamlit as st
import numpy as np
import cv2
from PIL import Image
import requests
import base64
import io

BACKEND_URL = "https://your-backend-url.com/predict"  # Replace with your actual backend URL

st.set_page_config(page_title="ASL Gesture Recognition System", layout="wide")
st.title("🧠 ASL Gesture Recognition System")
st.write("Upload an image to predict A or B")

uploaded_file = st.file_uploader("Upload Hand Image", type=["jpg", "png", "jpeg"])

if uploaded_file is None:
    st.info("Upload an image to preview it. If the backend is available, prediction will appear after upload.")
else:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not decode the uploaded image. Please upload a valid JPG or PNG file.")
    else:
        st.image(img, channels="BGR", caption="Uploaded Image")

        # Convert image to base64 for sending to backend
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        try:
            # Send to backend for prediction
            response = requests.post(
                BACKEND_URL,
                json={"image": img_base64},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                gesture = result.get("gesture", "Unknown")
                confidence = result.get("confidence", 0.0)

                st.markdown("## 🔍 Prediction Result")
                st.success(f"Gesture: {gesture}")
                st.info(f"Confidence: {confidence:.2f}%")

                if "probabilities" in result:
                    st.bar_chart(result["probabilities"])
            else:
                st.error(f"Backend error: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as err:
            st.error(f"Failed to connect to backend: {err}")
            st.info("The UI is working, but prediction requires a running backend service.")
        except Exception as err:
            st.error(f"Prediction failed: {err}")
            st.info("The UI is still available; try again or check the backend.")

