# Face Recognition Control System

[![tests](https://github.com/1AyaNabil1/face-recognition-control-system/actions/workflows/test.yml/badge.svg)](https://github.com/1AyaNabil1/face-recognition-control-system/actions/workflows/test.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)

A face recognition service that enrolls people from photos and recognizes
them later. Faces are found with a YOLOv8 face detector, embedded with
InsightFace (ArcFace), and matched against per-person templates stored in
SQLite. It ships with:

- a **Flask HTTP API** (`api/`) to enroll, recognize, list and delete people,
- a **bulk-enrollment script** (`scripts/generate_embeddings.py`),
- a **Tkinter desktop GUI** (`app/gui/face_recognition_gui.py`),
- a **Streamlit web client** (`web/app.py`) that talks to the API.

The repository also holds an unfinished FastAPI/PostgreSQL/Redis
"production" prototype; it does not run yet and is described
[below](#unfinished-fastapi-prototype).

## How it works

```
image ──► YOLOv8n-face detector ──► largest face box (+30% context)
            (Haar cascade fallback)            │
                                               ▼
                         InsightFace SCRFD: finds the face and its
                         5 landmarks inside the padded crop, aligns it
                                               │
                                               ▼
                         ArcFace (buffalo_l) 512-d embedding, L2-normalised
                                               │
                     ┌─────────────────────────┴───────────────────────┐
           enroll    ▼                                     recognize    ▼
  store embedding in SQLite              cosine similarity to each person's
  (one row per photo)                    mean template ─► best match if score
                                         ≥ threshold, otherwise "Unknown"
```

- **Quality gates.** A crop is rejected if it is too dark/bright, has too
  little contrast, is blurry, or InsightFace's detection score is below
  `FRCS_MIN_FACE_QUALITY`. Rejections come back with a reason.
- **Templates.** Every stored embedding is normalised, the embeddings of each
  person are averaged and re-normalised, so adding more photos of a person
  makes their template more robust.
- **Ambiguity rule.** If the two best people are within 0.1 of each other, the
  required score is raised by 0.1, which avoids confidently picking one of two
  look-alikes.
- **One pipeline everywhere.** The API, GUI and script all go through
  `api/recognition_service.py`, so enrollment and recognition crop and embed
  faces the same way.

## Quick start

Requires Python 3.11+ and a C/C++ compiler (InsightFace 0.7.3 builds from
source; on macOS install the Xcode command line tools, on Debian/Ubuntu
`build-essential`).

```bash
git clone https://github.com/1AyaNabil1/face-recognition-control-system.git
cd face-recognition-control-system
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # optional, every setting has a default
```

The YOLOv8 face weights are in the repository. The InsightFace `buffalo_l`
model pack (~280 MB) is downloaded to `FRCS_INSIGHTFACE_ROOT`
(default `~/.insightface`) the first time a face is embedded.

**Enroll people** from a folder with one sub-folder per person:

```
dataset/
  Alice/  1.jpg  2.jpg
  Bob/    1.jpg
```

```bash
python scripts/generate_embeddings.py --dataset dataset
# --reset       start from an empty database
# --save-crops  write the detected face crops to debug_crops/ for inspection
```

**Run the API:**

```bash
python -m api.app                    # http://127.0.0.1:5000, for development
gunicorn --workers 2 --timeout 120 --bind 0.0.0.0:8000 "api.app:create_app()"
```

Models load on the first request that needs them, so that request takes a
few seconds; `GET /api/health` reports `models_loaded`.

**Optional clients:**

```bash
python -m app.gui.face_recognition_gui     # desktop GUI (needs Tkinter)

pip install -r web/requirements.txt        # Streamlit client
FRCS_API_URL=http://localhost:5000 streamlit run web/app.py
```

## API

All bodies are JSON; images are base64 strings (plain or `data:image/...;base64,`
URLs, JPEG or PNG). Errors are returned as `{"error": "<message>"}`. When
`FRCS_API_KEY` is set, every endpoint except `/api/health` requires the header
`X-API-Key: <key>`.

| Method | Path | Body | Success response |
| --- | --- | --- | --- |
| `GET` | `/api/health` | | `{"status": "OK", "models_loaded": false, ...}` |
| `POST` | `/api/add-person` | `{"name", "image"}` | `{"message", "name", "quality", "face_box"}` |
| `POST` | `/api/recognize` | `{"image", "return_annotated_image"?}` | `{"result", "confidence", "top_matches", "faces_detected", "face_box", "annotated_image"}` |
| `GET` | `/api/persons` | | `{"persons": [{"name", "embeddings"}]}` |
| `DELETE` | `/api/persons/<name>` | | `{"message", "name", "deleted"}` |

Status codes: `400` invalid JSON/name/image, `401` missing or wrong API key,
`404` unknown person, `413` image larger than `FRCS_MAX_IMAGE_BYTES`,
`422` no usable face (the message says why), `503` models could not be
loaded, `500` unexpected error (details are only written to the server log).

Example with `curl`:

```bash
printf '{"name": "Alice", "image": "%s"}' "$(base64 < alice.jpg | tr -d '\n')" > enroll.json
curl -s -X POST localhost:5000/api/add-person -H 'Content-Type: application/json' -d @enroll.json

printf '{"image": "%s", "return_annotated_image": false}' "$(base64 < probe.jpg | tr -d '\n')" > probe.json
curl -s -X POST localhost:5000/api/recognize -H 'Content-Type: application/json' -d @probe.json
# {"result": "Alice", "confidence": <best score>, "top_matches": [...],
#  "faces_detected": 1, "face_box": [x1, y1, x2, y2]}
```

`result` is `"Unknown"` when no enrolled person clears the threshold;
`confidence` is the cosine similarity of the best match.

## Configuration

Everything is read from environment variables (or a `.env` file, see
[`.env.example`](.env.example)). Nothing machine-specific is hard-coded.

| Variable | Default | Purpose |
| --- | --- | --- |
| `FRCS_DB_PATH` | `app/database/embeddings.db` | SQLite file with the embeddings |
| `FRCS_RECOGNITION_LOG_PATH` | `logs/recognition_log.csv` | CSV of successful recognitions |
| `FRCS_LOG_RECOGNITIONS` | `true` | Set `false` to disable that log |
| `FRCS_YOLO_MODEL_PATH` | `app/models/yolo/yolov8n-face-lindevs.pt` | Detector weights |
| `FRCS_YOLO_CONFIDENCE` | `0.2` | Detector confidence threshold |
| `FRCS_INSIGHTFACE_MODEL` | `buffalo_l` | InsightFace model pack |
| `FRCS_INSIGHTFACE_ROOT` | `~/.insightface` | Where model packs are downloaded |
| `FRCS_INSIGHTFACE_DET_SIZE` | `640` | InsightFace detector input size |
| `FRCS_MATCH_THRESHOLD` | `0.6` | Cosine similarity needed to accept a match |
| `FRCS_MIN_FACE_QUALITY` | `0.5` | Minimum InsightFace detection score |
| `FRCS_MAX_IMAGE_BYTES` | `10485760` | Largest accepted image (decoded) |
| `FRCS_API_KEY` | empty (no auth) | Shared key required in `X-API-Key` |
| `FRCS_CORS_ORIGINS` | empty (no CORS) | Comma-separated allowed origins |
| `FRCS_HOST` / `PORT` | `127.0.0.1` / `5000` | Bind address for `python -m api.app` |
| `FRCS_API_URL` | `http://localhost:5000` | API used by the Streamlit client |

## Tests and linting

The tests replace YOLOv8 and InsightFace with small deterministic fakes, so
they run offline in about a second and need only light dependencies:

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
python -m pytest
```

They cover the matching logic (normalisation, templates, threshold and
ambiguity rule), the quality gates and crop padding in the embedder, the
detector's box handling, the SQLite layer (validation, corrupt rows, deletion),
every API endpoint (input validation, size limits, API key, CORS, error
handling, model-loading failures) and the enrollment script.
[GitHub Actions](.github/workflows/test.yml) runs the same commands on
Python 3.11, 3.12 and 3.13 for every push and pull request.

## Privacy and security

Face embeddings are biometric data and are regulated in many places (for
example GDPR in the EU and BIPA in Illinois). If you deploy this:

- Only enroll people who have given informed consent, and use
  `DELETE /api/persons/<name>` (or the Streamlit "View Users" page) to erase
  someone's data when they withdraw it. Deletion works even if the models are
  not loaded.
- Protect `FRCS_DB_PATH` and the recognition log like any other personal
  data. Enrollment photos are not stored by the API; the bulk script only
  writes face crops when run with `--save-crops`.
- Set `FRCS_API_KEY` and keep CORS closed (the default) before exposing the
  API beyond localhost, and serve it behind HTTPS.
- Never combine `DEBUG=true` with `FRCS_HOST=0.0.0.0`: the Flask debugger
  allows remote code execution.

## Limitations

- **No accuracy figures.** The pipeline has not been benchmarked here (for
  example on LFW), and the default threshold of 0.6 is a conservative choice,
  not a calibrated one. Measure false accepts/rejects on your own data before
  relying on it. The test suite checks behaviour, not recognition quality.
- **No liveness or anti-spoofing.** A printed photo or a screen can be
  enrolled or recognized.
- **One face per request.** Only the largest detected face is used.
- **Linear search.** Every request compares against all templates loaded from
  SQLite. That is fine for small galleries but not for very large ones, and
  SQLite limits it to a single machine.
- **Memory per worker.** Each gunicorn worker loads its own copy of the models.
- **Not exercised in CI:** the GUI (needs a display) and the real models (the
  tests use fakes). The full pipeline was checked manually with the real
  YOLOv8 and InsightFace models.

## Project layout

```
api/                      Flask API, request validation, recognition service
app/settings.py           environment-based configuration
app/detection/            YOLOv8 face detector (+ Haar fallback)
app/embedding/            InsightFace embedder and quality gates
app/recognition/face_recognizer.py   template matching and thresholds
app/database/db_manager.py           SQLite storage
app/gui/                  Tkinter desktop app
scripts/                  bulk enrollment
web/                      Streamlit client
tests/                    offline test suite
```

## Unfinished FastAPI prototype

`app/main.py`, `app/core/`, `app/routers/`, `app/repositories/`,
`app/models/` (except the YOLO weights), `app/recognition/` (except
`face_recognizer.py`), `app/detection/face_detector.py`, `app/utils/`,
`modal_app.py`, `modal_app/`, `modal_deploy.py`, `deployment/`, the
`Dockerfile`, `docker-entrypoint.sh`, `postman/` and
`requirements-prototype.txt` belong to an earlier attempt at a FastAPI service
with PostgreSQL, Redis, MTCNN/FaceNet, DeepFace attributes, Prometheus and
Sentry, deployed on Modal. It is kept for reference but does not start:
among other things it imports a missing `app.core.routes` module, runs Alembic
migrations without an `alembic.ini`, needs PostgreSQL and Redis at startup,
and its anti-spoofing model file is not included. None of the
features listed only there (anti-spoofing, age/gender/emotion, rate limiting,
metrics) are available in the working service.

## Model weights and licenses

| File | Used | Source and license notes |
| --- | --- | --- |
| `app/models/yolo/yolov8n-face-lindevs.pt` | yes | [lindevs/yolov8-face](https://github.com/lindevs/yolov8-face) release 1.0.1 (repository MIT-licensed); trained with Ultralytics YOLOv8, which is AGPL-3.0 |
| InsightFace `buffalo_l` (downloaded at runtime) | yes | [deepinsight/insightface](https://github.com/deepinsight/insightface); its pretrained models are released for non-commercial research use only |
| `app/models/yolo/yolov8n.pt` | no | Ultralytics YOLOv8n COCO weights (AGPL-3.0) |
| `app/utils/shape_predictor_68_face_landmarks.dat` (95 MB) | no | dlib 68-point landmark model, trained on iBUG 300-W, whose license excludes commercial use |

Check these licenses before any commercial use.

## License

A LICENSE file has not been added yet, so no open-source license currently
applies to the code. Third-party model weights keep their own licenses (see
above).

## Contributing

Issues and pull requests are welcome. Please run `ruff check .`,
`ruff format --check .` and `python -m pytest` before opening a pull request.

## 👥 Authors

- **Aya Nabil** - *Initial work* - [1AyaNabil1](https://github.com/1AyaNabil1)

## 🙏 Acknowledgments

- InsightFace team for the face recognition models
- Modal team for the deployment platform
- Open source community for various tools and libraries
- [lindevs/yolov8-face](https://github.com/lindevs/yolov8-face) for the YOLOv8 face-detection weights and [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
