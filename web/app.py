import streamlit as st
import requests
import base64
from PIL import Image
import io
from typing import Tuple, Dict, Optional

st.set_page_config(page_title="Face Recognition System", layout="centered")

# Dark Theme CSS
st.markdown(
    """
<style>
.stApp { background-color: #0e0e0e; color: #ffffff; }
.sidebar .sidebar-content { background-color: #1c1c1c; color: white; }
.stButton>button {
    background-color: #27ae60; color: white; border-radius: 5px; padding: 10px 20px;
}
.stButton>button:hover { background-color: #219653; }
.stError { background-color: #e74c3c; color: white; padding: 10px; border-radius: 5px; }
</style>
""",
    unsafe_allow_html=True,
)

API_URL = (
    "https://ayanabil1--face-recognition-system-recognize-face.modal.run/api/recognize"
)
API_ADD_URL = (
    "https://ayanabil1--face-recognition-system-recognize-face.modal.run/api/add-person"
)


def process_image(file) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        img_bytes = file.read()
        Image.open(io.BytesIO(img_bytes)).verify()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        return img_bytes, img_b64
    except Exception as e:
        return None, f"Invalid image file: {str(e)}"


def display_image(img_bytes: bytes, caption: str, width: int = 500):
    try:
        st.image(img_bytes, caption=caption, width=width)
    except Exception as e:
        st.error(f"Failed to display image: {str(e)}")


def call_api(image_b64: str) -> Dict:
    try:
        response = requests.post(API_URL, json={"image": image_b64}, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}


def add_person_api(name: str, image_b64: str) -> Dict:
    try:
        response = requests.post(
            API_ADD_URL, json={"name": name, "image": image_b64}, timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}


st.sidebar.title("Face Recognition System")
st.sidebar.write("Upload an image or add a new person to the system.")
st.sidebar.write("Developed by: Aya Nabil")

st.title("🔍 Face Recognition Control System")
uploaded_file = st.file_uploader(
    "📂 Upload an Image", type=["jpg", "jpeg", "png"], key="file_uploader"
)

if uploaded_file:
    img_bytes, img_b64_or_error = process_image(uploaded_file)
    if isinstance(img_b64_or_error, str):
        st.error(img_b64_or_error)
    else:
        img_b64 = img_b64_or_error
        st.subheader("Uploaded Image")
        display_image(img_bytes, "Uploaded Image")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Image Recognition"):
                with st.spinner("Recognizing faces..."):
                    result = call_api(img_b64)
                    if "error" in result:
                        st.error(f"Recognition Failed: {result['error']}")
                    else:
                        st.success(
                            f"🎯 Match: {result['result']} (Confidence: {result['confidence']})"
                        )
                        st.subheader("📋 Top Matches")
                        for match in result["top_matches"]:
                            st.write(
                                f"- **{match['name']}** — Confidence: {round(match['score'], 4)}"
                            )

                        st.subheader("📸 Annotated Image")
                        annotated_b64 = result["annotated_image"]
                        annotated_image = base64.b64decode(annotated_b64)
                        display_image(annotated_image, "Detected Faces")

        with col2:
            if st.button("Live Recognition"):
                st.warning("Live recognition is not yet implemented.")

        with col3:
            if st.button("Add New Person"):
                name = st.text_input("Enter Person's Name")
                add_image = st.file_uploader(
                    "Upload Image for New Person",
                    type=["jpg", "jpeg", "png"],
                    key="add_person",
                )
                if st.button("Submit"):
                    if name and add_image:
                        img_bytes_new, img_b64_or_error_new = process_image(add_image)
                        if isinstance(img_b64_or_error_new, str):
                            st.error(img_b64_or_error_new)
                        else:
                            response = add_person_api(name, img_b64_or_error_new)
                            if "error" in response:
                                st.error(response["error"])
                            else:
                                st.success(response["message"])
                    else:
                        st.warning("Please provide both a name and an image.")
