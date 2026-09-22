# Identity Verification System MVP (Python)

A decoupled, modular, and privacy-preserving Python MVP for multi-factor identity verification. The system matches a photo from an uploaded identity document against a live camera feed using **InsightFace (buffalo_l / SCRFD + ArcFace with ONNX Runtime)** for facial embeddings, **PaddleOCR** for document name extraction, **OpenCV** for camera capture, and **FastAPI** for an extensible REST API and web dashboard.

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Prerequisites & Python Version](#prerequisites--python-version)
3. [Installation & Setup](#installation--setup)
4. [Running the Application](#running-the-application)
5. [How It Works](#how-it-works)
   - [Facial Embeddings](#facial-embeddings)
   - [Cosine Similarity](#cosine-similarity)
   - [Threshold Calibration](#threshold-calibration)
   - [Verification Decision Logic](#verification-decision-logic)
6. [API Specification & Examples](#api-specification--examples)
7. [CLI Utilities & Testing](#cli-utilities--testing)
   - [Direct Image Comparison CLI](#direct-image-comparison-cli)
   - [Benchmark Evaluation Script](#benchmark-evaluation-script)
   - [Automated Pytest Suite](#automated-pytest-suite)
8. [Connecting with JavaFX (Future Extension)](#connecting-with-javafx-future-extension)
9. [Privacy & Biometrics Handling](#privacy--biometrics-handling)

---

## Architecture Overview

The system strictly decouples the face recognition, OCR, camera, and verification layers:

```
facePython/
├── app/
│   ├── main.py                     # FastAPI application, CORS, error handling, static UI
│   ├── api/
│   │   ├── routes_document.py      # POST /document/analyze
│   │   ├── routes_verification.py  # POST /verification/start, POST /verification/compare
│   │   └── routes_camera.py        # GET /cameras, POST /cameras/select, GET /camera/stream
│   ├── services/
│   │   ├── face_service.py         # InsightFace (SCRFD + ArcFace) embeddings & cosine similarity
│   │   ├── ocr_service.py          # PaddleOCR text and name extraction (independent of FaceService)
│   │   ├── camera_service.py       # OpenCV device enumeration, capture & MJPEG streaming
│   │   └── verification_service.py # Orchestrator for face similarity + name matching
│   ├── models/
│   │   └── schemas.py              # Pydantic models for structured requests and responses
│   ├── core/
│   │   ├── config.py               # Environment configuration (.env)
│   │   ├── exceptions.py           # Standard domain exceptions (DOCUMENT_FACE_NOT_FOUND, etc.)
│   │   └── logging.py              # Privacy-safe structured logging
│   ├── utils/
│   │   ├── image.py                # Image decoding, EXIF orientation, cropping, format checks
│   │   └── text.py                 # Text normalization, token & character similarity
│   └── static/
│       ├── index.html              # Responsive verification dashboard
│       ├── style.css               # Clean dark-mode aesthetic styling
│       └── app.js                  # Frontend workflow, stream handler, verification triggers
├── tests/                          # 13 pytest test scenarios covering edge cases
├── compare_faces.py                # CLI tool for comparing 2 static image files
├── evaluate.py                     # Benchmark tool for FAR / FRR / accuracy evaluation
├── requirements.txt                # Dependencies
├── Dockerfile & docker-compose.yml # Containerized deployment
└── README.md
```

---

## Prerequisites & Python Version

- **Python**: `3.10`, `3.11`, or `3.12` recommended (e.g. `/opt/homebrew/bin/python3.12` on macOS).
- **OS**: macOS (Apple Silicon / Intel), Linux, or Windows.
- **Webcam**: Any built-in or USB webcam supported by OpenCV.

---

## Installation & Setup

### 1. Clone & Enter Project Directory
```bash
cd /Users/nav/Building/facePython
```

### 2. Create Virtual Environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Key environment variables:
```env
HOST=0.0.0.0
PORT=8000
FACE_MODEL=buffalo_l
FACE_SIMILARITY_THRESHOLD=0.50
OCR_LANGUAGE=en
CAMERA_INDEX=0
SESSION_TTL_SECONDS=900
```

---

## Running the Application

### Option A: Local Dev Server
```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser and navigate to: **`http://localhost:8000`**

### Option B: Docker Compose
```bash
docker compose up --build
```

---

## How It Works

```
[ Upload Document ]  ---> [ Face Detection (SCRFD) ]  ---> Exactly 1 Face?
                                                                 │
                                                                 ├──> Crop Face & Generate 512-D Embedding
                                                                 │
[ Document Image ]   ---> [ PaddleOCR Engine ]        ---> Extract Candidate Name
                                                                 │
                                                      [ Create Temporary Session ]
                                                                 │
[ Live Camera Feed ] ---> [ Face Detection (SCRFD) ]  ---> Exactly 1 Face?
                                                                 │
                                                                 ├──> Generate 512-D Live Embedding
                                                                 │
                                                      [ Cosine Similarity Computation ]
                                                                 │
                                                      [ Name Match Comparison ]
                                                                 │
                                                      [ Final Verdict: NAME && FACE ]
```

### Facial Embeddings
InsightFace extracts a 512-dimensional vector embedding representing high-level facial features (invariant to minor age changes, lighting, pose, and glasses). The embedding is $L_2$-normalized such that $\|v\|_2 = 1.0$.

### Cosine Similarity
Cosine similarity is computed via the dot product of two normalized embeddings:
$$\text{Cosine Similarity} = \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{512} u_i v_i$$
- A score close to `1.0` indicates an identical identity.
- A score below `0.5` indicates distinct individuals.

### Threshold Calibration
The default production threshold is set to `0.50`:
- `similarity >= 0.50` $\rightarrow$ `face_match = True`
- `similarity < 0.50` $\rightarrow$ `face_match = False`

> [!NOTE]
> `0.50` provides robust age-invariant matching while maintaining complete separation from impostors. Run `evaluate.py` on your domain-specific dataset (national IDs, driver licenses, passports) to calibrate the optimal threshold balancing False Acceptance Rate (FAR) and False Rejection Rate (FRR).

### Verification Decision Logic
The final verification decision is strictly conjunctive:
$$\text{verified} = (\text{name\_match} == \text{True}) \land (\text{face\_match} == \text{True})$$

---

## API Specification & Examples

### 1. `POST /document/analyze`
Analyzes an uploaded identity document image.

**Request**: Multipart form data with `file` field.
```bash
curl -X POST http://localhost:8000/document/analyze \
  -F "file=@/path/to/driver_license.jpg"
```

**Response**:
```json
{
  "success": true,
  "document": {
    "face_detected": true,
    "face_count": 1,
    "name": "SUNAV SHARMA",
    "all_extracted_text": [
      "DRIVING LICENSE",
      "NAME: SUNAV SHARMA",
      "DOB: 1995-04-12"
    ],
    "session_id": "8f8b6631-15a9-4672-9cb7-28d15a98bf49"
  }
}
```

### 2. `POST /verification/compare`
Compares a live camera capture or uploaded selfie against an active document session.

**Request**:
```bash
curl -X POST http://localhost:8000/verification/compare \
  -F "session_id=8f8b6631-15a9-4672-9cb7-28d15a98bf49" \
  -F "expected_name=SUNAV SHARMA"
```

**Response**:
```json
{
  "success": true,
  "document": {
    "face_detected": true,
    "face_count": 1,
    "name": "SUNAV SHARMA"
  },
  "verification": {
    "document_valid": true,
    "document_face_detected": true,
    "document_face_count": 1,
    "ocr_success": true,
    "extracted_document_name": "SUNAV SHARMA",
    "expected_name": "SUNAV SHARMA",
    "name_match": true,
    "live_face_detected": true,
    "live_face_count": 1,
    "similarity": 0.8124,
    "threshold": 0.50,
    "face_match": true,
    "verified": true
  }
}
```

### 3. `GET /cameras`
Returns available camera devices on the host.
```bash
curl http://localhost:8000/cameras
```

### 4. `GET /health`
Returns system diagnostics, loaded models, and configuration.
```bash
curl http://localhost:8000/health
```

---

## CLI Utilities & Testing

### Direct Image Comparison CLI
Compare any two static images without running the web UI or using the camera:

```bash
python compare_faces.py ./sample_doc.jpg ./sample_selfie.jpg --threshold 0.50 --expected-name "JOHN DOE"
```

**Example Output:**
```text
Document face detected: YES
Live face detected:     YES
Document faces:         1
Live faces:             1
Extracted document name: JOHN DOE
Expected name:           JOHN DOE
Name match:              YES
Cosine similarity:       0.8241
Threshold:               0.5000
Result:                  MATCH
Final Verification:      VERIFIED
```

### Benchmark Evaluation Script
Evaluate a benchmark dataset containing positive (genuine) and negative (impostor) pairs:

```bash
python evaluate.py --dataset-dir ./data/benchmark --threshold 0.50
```

**Generates report with:**
- Number of genuine & impostor pairs
- Mean genuine similarity vs Mean impostor similarity
- Confusion matrix (TP, TN, FP, FN)
- Accuracy, Precision, Recall / TPR
- False Acceptance Rate (FAR) & False Rejection Rate (FRR)

### Automated Pytest Suite
Run the test suite covering all 13 test scenarios:
```bash
pytest tests/ -v
```

---

## Connecting with JavaFX (Future Extension)

The Python backend exposes a standard REST API and MJPEG stream. A JavaFX desktop client can interact with this service using standard Java HTTP clients (e.g. `java.net.http.HttpClient` or `OkHttp`):

1. **Document Analysis**: JavaFX sends `POST /document/analyze` via `MultipartBody` when the user selects a document file.
2. **Video Streaming**: JavaFX renders the MJPEG stream from `GET /camera/stream?session_id=<id>` directly inside an `ImageView`.
3. **Trigger Verification**: When the user clicks "Verify" in the JavaFX UI, it sends `POST /verification/compare` with the session ID.
4. **Display Breakdown**: JavaFX binds the structured JSON response fields to JavaFX labels and status indicators.

---

## Privacy & Biometrics Handling

- **Zero Permanent Biometric Storage**: No face embeddings, raw document images, or camera frames are persisted to disk or databases.
- **In-Memory Session TTL**: Verification sessions reside exclusively in volatile memory and expire automatically after `SESSION_TTL_SECONDS` (default: 15 minutes).
- **Sanitized Logging**: Biometric vectors, image arrays, and stack traces are excluded from application logs.
