import streamlit as st
import requests
import base64
from PIL import Image
import io
import os

st.set_page_config(page_title="Face Recognition System", layout="centered")
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🔍 Face Recognition Control System</h1>",
    unsafe_allow_html=True,
)

# Updated with actual Flask endpoint
API_URL = os.getenv("API_URL", "https://ayanabil1--run-flask.modal.run/api/recognize")

uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_bytes = uploaded_file.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(
            Image.open(io.BytesIO(img_bytes)),
            caption="Uploaded Image",
            use_container_width=True,
        )

    if st.button("🚀 Recognize Face"):
        with st.spinner("Processing..."):
            try:
                response = requests.post(API_URL, json={"image": img_b64}, timeout=30)
                response.raise_for_status()  # Raise exception for non-200 status
            except requests.RequestException as e:
                st.error(f"Recognition Failed: {str(e)}")
                st.stop()

            if response.status_code == 200:
                result = response.json()
                st.success(
                    f"🎯 Match: {result['result']} (Confidence: {result['confidence']})"
                )

                st.markdown("<h3>📋 Top Matches</h3>", unsafe_allow_html=True)
                for match in result["top_matches"]:
                    st.markdown(
                        f"- **{match['name']}** — Confidence: {round(match['score'], 4)}"
                    )

                st.markdown("<h3>📸 Annotated Image</h3>", unsafe_allow_html=True)
                annotated_b64 = result["annotated_image"]
                annotated_image_data = base64.b64decode(annotated_b64)
                annotated_image = Image.open(io.BytesIO(annotated_image_data))

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(
                        annotated_image,
                        caption="Detected Faces",
                        use_container_width=True,
                    )
            else:
                st.error(
                    f"Recognition Failed: {response.json().get('error', 'Unknown error')}"
                )
