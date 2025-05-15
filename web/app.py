import streamlit as st
import requests
import base64
from PIL import Image
import io
from typing import Tuple, Dict, Optional

# Custom styling with pastel background
st.set_page_config(
    page_title="Face Recognition System",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #d6e4ff;  /* Pastel blue background */
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    .stButton>button {
        background-color: #27ae60;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #219653;
    }
    .stError {
        background-color: #e74c3c;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar for app info
with st.sidebar:
    st.title("Face Recognition System")
    st.write("Upload an image to recognize faces using our advanced API.")
    st.write("Developed by: Aya Nabil")  # Updated developer name
    st.write(
        "API Endpoint: https://ayanabil1--face-recognition-system-recognize-face.modal.run"
    )

# Session state to manage recognition state
if "recognition_result" not in st.session_state:
    st.session_state.recognition_result = None
    st.session_state.error = None

# API URL
API_URL = "https://ayanabil1--face-recognition-system-recognize-face.modal.run"


def process_image(file) -> Tuple[Optional[bytes], Optional[str]]:
    """Process the uploaded image and return bytes and base64 encoding."""
    try:
        img_bytes = file.read()
        return img_bytes, base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        return None, str(e)


def display_image(img_bytes: bytes, caption: str, width: int = 500):
    """Display an image with a specified width."""
    try:
        st.image(img_bytes, caption=caption, width=width)
    except Exception as e:
        st.error(f"Failed to display image: {str(e)}")


def call_recognition_api(image_b64: str) -> Dict:
    """Call the Modal API for face recognition."""
    try:
        response = requests.post(API_URL, json={"image": image_b64}, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}


def main():
    st.title("🔍 Face Recognition Control System")
    st.write("Upload an image to detect and recognize faces.")

    # File uploader
    uploaded_file = st.file_uploader(
        "📂 Upload an Image", type=["jpg", "jpeg", "png"], key="file_uploader"
    )

    if uploaded_file is not None:
        with st.spinner("Processing image..."):
            img_bytes, error = process_image(uploaded_file)
            if error:
                st.session_state.error = f"Image processing error: {error}"
            else:
                # Display uploaded image
                st.subheader("Uploaded Image")
                display_image(img_bytes, "Uploaded Image")

                # Recognize button
                if st.button("🚀 Recognize Face"):
                    with st.spinner("Recognizing faces..."):
                        result = call_recognition_api(
                            error or process_image(uploaded_file)[1]
                        )
                        if "error" in result:
                            st.session_state.error = (
                                f"Recognition Failed: {result['error']}"
                            )
                        else:
                            st.session_state.recognition_result = result
                            st.session_state.error = None

        # Display results or errors
        if st.session_state.error:
            st.error(st.session_state.error)
        elif st.session_state.recognition_result:
            st.success(
                f"🎯 Match: {st.session_state.recognition_result['result']} "
                f"(Confidence: {st.session_state.recognition_result['confidence']})"
            )

            st.subheader("📋 Top Matches")
            for match in st.session_state.recognition_result["top_matches"]:
                st.write(
                    f"- **{match['name']}** — Confidence: {round(match['score'], 4)}"
                )

            st.subheader("📸 Annotated Image")
            annotated_b64 = st.session_state.recognition_result["annotated_image"]
            annotated_image_data = base64.b64decode(annotated_b64)
            annotated_image = Image.open(io.BytesIO(annotated_image_data))
            display_image(annotated_image.tobytes(), "Detected Faces")


if __name__ == "__main__":
    main()
