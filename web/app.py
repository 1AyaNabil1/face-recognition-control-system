"""Streamlit front-end for the Flask recognition API.

Run the API first (`python -m api.app`), then:
    FRCS_API_URL=http://localhost:5000 streamlit run web/app.py
"""

import base64
import os
from urllib.parse import quote

import requests
import streamlit as st

# Set the title of the Streamlit app
st.set_page_config(page_title="Face Recognition Control System", layout="wide")

# Base URL of the Flask API (api/app.py)
API_URL = os.getenv("FRCS_API_URL", "http://localhost:5000").rstrip("/")
API_KEY = os.getenv("FRCS_API_KEY", "")
TIMEOUT_SECONDS = 60


def _headers():
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _error_message(response):
    try:
        return response.json().get("error", response.text)
    except ValueError:
        return response.text


# Function to register a new user
def register_user(name, image_bytes):
    payload = {"name": name, "image": base64.b64encode(image_bytes).decode("ascii")}
    return requests.post(
        f"{API_URL}/api/add-person",
        json=payload,
        headers=_headers(),
        timeout=TIMEOUT_SECONDS,
    )


# Function to recognize face from image
def recognize_face(image_bytes):
    payload = {"image": base64.b64encode(image_bytes).decode("ascii")}
    return requests.post(
        f"{API_URL}/api/recognize",
        json=payload,
        headers=_headers(),
        timeout=TIMEOUT_SECONDS,
    )


# Function to get list of registered users
def get_users():
    return requests.get(
        f"{API_URL}/api/persons", headers=_headers(), timeout=TIMEOUT_SECONDS
    )


def delete_user(name):
    return requests.delete(
        f"{API_URL}/api/persons/{quote(name, safe='')}",
        headers=_headers(),
        timeout=TIMEOUT_SECONDS,
    )


def show_recognition(response):
    if response.status_code != 200:
        st.error(f"Recognition failed: {_error_message(response)}")
        return
    result = response.json()
    if result.get("faces_detected") == 0:
        st.warning("No face detected.")
        return
    st.success(f"Recognized: {result['result']} (score {result['confidence']})")
    if result.get("annotated_image"):
        st.image(base64.b64decode(result["annotated_image"]), caption="Detected faces")
    if result.get("top_matches"):
        st.write("Top matches:")
        for match in result["top_matches"]:
            st.write(f"- {match['name']}: {match['score']}")


# Main application
def main():
    st.title("Face Recognition Control System")
    st.sidebar.title("Navigation")
    st.sidebar.caption(f"API: {API_URL}")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Home", "Live Recognition", "Recognize Photo", "Register User", "View Users"],
    )

    try:
        if app_mode == "Home":
            st.write(
                "Welcome to the Face Recognition Control System. "
                "Use the sidebar to navigate."
            )

        elif app_mode == "Live Recognition":
            st.subheader("Live Face Recognition")
            snapshot = st.camera_input("Take a picture")
            if snapshot is not None:
                with st.spinner("Recognizing..."):
                    show_recognition(recognize_face(snapshot.getvalue()))

        elif app_mode == "Recognize Photo":
            st.subheader("Recognize a Photo")
            uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if uploaded is not None and st.button("Recognize"):
                with st.spinner("Recognizing..."):
                    show_recognition(recognize_face(uploaded.getvalue()))

        elif app_mode == "Register User":
            st.subheader("Register a New User")
            name = st.text_input("Enter the name of the user")
            uploaded_file = st.file_uploader(
                "Upload an image", type=["jpg", "jpeg", "png"]
            )
            if st.button("Register"):
                if name and uploaded_file:
                    with st.spinner("Registering..."):
                        response = register_user(name, uploaded_file.getvalue())
                        if response.status_code == 200:
                            st.success("User registered successfully.")
                        else:
                            st.error(f"Registration failed: {_error_message(response)}")
                else:
                    st.warning("Please provide both name and image.")

        elif app_mode == "View Users":
            st.subheader("Registered Users")
            response = get_users()
            if response.status_code != 200:
                st.error(f"Failed to fetch users: {_error_message(response)}")
                return
            users = response.json().get("persons", [])
            if not users:
                st.info("No users registered yet.")
            for user in users:
                col_name, col_delete = st.columns([4, 1])
                col_name.write(f"- {user['name']} ({user['embeddings']} embedding(s))")
                if col_delete.button("Delete", key=f"delete-{user['name']}"):
                    delete_response = delete_user(user["name"])
                    if delete_response.status_code == 200:
                        st.success(f"Deleted {user['name']}.")
                        st.rerun()
                    else:
                        st.error(f"Delete failed: {_error_message(delete_response)}")

    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_URL}: {exc}")


if __name__ == "__main__":
    main()
