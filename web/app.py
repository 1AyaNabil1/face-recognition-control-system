import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Set the title of the Streamlit app
st.set_page_config(page_title="Face Recognition Control System", layout="wide")

# Define the base URL for the Flask API
API_URL = (
    "https://ayanabil1--face-recognition-system-recognize-face.modal.run/api/recognize"
)


# Function to capture image from webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    st.info("Press 'c' to capture an image or 'q' to quit.")
    img_captured = False
    img = None

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access the webcam.")
            break
        cv2.imshow("Press 'c' to capture or 'q' to quit", frame)
        key = cv2.waitKey(1)
        if key == ord("c"):
            img = frame
            img_captured = True
            break
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return img if img_captured else None


# Function to convert OpenCV image to bytes
def convert_image_to_bytes(img):
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


# Function to register a new user
def register_user(name, image_bytes):
    files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
    data = {"name": name}
    response = requests.post(f"{API_URL}/register", files=files, data=data)
    return response


# Function to recognize face from image
def recognize_face(image_bytes):
    files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
    response = requests.post(f"{API_URL}/recognize", files=files)
    return response


# Function to get list of registered users
def get_users():
    response = requests.get(f"{API_URL}/users")
    return response


# Function to get system logs
def get_logs():
    response = requests.get(f"{API_URL}/logs")
    return response


# Main application
def main():
    st.title("Face Recognition Control System")
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Home", "Live Recognition", "Register User", "View Users", "System Logs"],
    )

    if app_mode == "Home":
        st.write(
            "Welcome to the Face Recognition Control System. Use the sidebar to navigate."
        )

    elif app_mode == "Live Recognition":
        st.subheader("Live Face Recognition")
        img = capture_image()
        if img is not None:
            st.image(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                caption="Captured Image",
                use_column_width=True,
            )
            image_bytes = convert_image_to_bytes(img)
            with st.spinner("Recognizing..."):
                response = recognize_face(image_bytes)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Recognized: {result.get('name', 'Unknown')}")
                else:
                    st.error("Recognition failed.")

    elif app_mode == "Register User":
        st.subheader("Register a New User")
        name = st.text_input("Enter the name of the user")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if st.button("Register"):
            if name and uploaded_file:
                image_bytes = uploaded_file.read()
                with st.spinner("Registering..."):
                    response = register_user(name, image_bytes)
                    if response.status_code == 200:
                        st.success("User registered successfully.")
                    else:
                        st.error("Registration failed.")
            else:
                st.warning("Please provide both name and image.")

    elif app_mode == "View Users":
        st.subheader("Registered Users")
        response = get_users()
        if response.status_code == 200:
            users = response.json().get("users", [])
            if users:
                for user in users:
                    st.write(f"- {user}")
            else:
                st.info("No users registered yet.")
        else:
            st.error("Failed to fetch users.")

    elif app_mode == "System Logs":
        st.subheader("System Logs")
        response = get_logs()
        if response.status_code == 200:
            logs = response.json().get("logs", [])
            if logs:
                for log in logs:
                    st.write(f"{log}")
            else:
                st.info("No logs available.")
        else:
            st.error("Failed to fetch logs.")


if __name__ == "__main__":
    main()
