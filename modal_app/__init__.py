import sys
import os
import modal

# Add root directory for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

app = modal.App("face-recognition-system")

image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "flask==2.3.3",
        "flask-cors==4.0.1",
        "onnxruntime==1.16.0",
        "insightface==0.7.3",
        "numpy==1.26.4",
        "pillow==10.4.0",
        "scikit-learn==1.5.1",
        "gunicorn==23.0.0",
        "ultralytics==8.2.58",
        "streamlit==1.36.0",
        "requests==2.31.0",
    )
)


@app.function(image=image)
def run_flask():
    import os

    os.system("gunicorn -w 4 -b 0.0.0.0:8000 api.app:app")


@app.function(image=image)
def run_streamlit():
    import os

    os.system("streamlit run web/app.py --server.port 8501 --server.address 0.0.0.0")
