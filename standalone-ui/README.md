# Standalone Face Verification UI

A lightweight, dedicated, and decoupled web user interface for **1:1 Face Verification** between an uploaded identity document and a live camera feed.

## Features
- **Takes User Document Input**: Drag & drop or browse identity document (JPG, PNG, WEBP).
- **Takes Live Camera Feed**: Directly streams camera in-browser with live framing guide (with an option to switch to server OpenCV stream).
- **Clear Match / Not Match Output**:
  - Distinct **MATCH** (emerald) or **NOT MATCH** (crimson) visual card.
  - **Similarity Score** displayed as percentage (e.g. `84.2%`) and raw cosine similarity (`0.8421`).
  - Visual similarity meter calibrated against the active threshold (default `0.40`).
  - Side-by-side face crops for instant visual inspection.
- **Completely Decoupled**: Kept in its own isolated folder (`standalone-ui/`) without touching or altering any existing frontend or backend logic.

---

## How to Run

### Step 1: Ensure the FastAPI Backend is Running
From the repository root `/Users/nav/Building/facePython`:
```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 2: Open this Standalone UI

You can open it in two ways:

#### Option A: Direct Browser File (No Extra Server Needed)
Simply open [`standalone-ui/index.html`](index.html) directly in Google Chrome, Edge, or Safari:
```bash
open standalone-ui/index.html
```

#### Option B: Simple Local Web Server
```bash
cd standalone-ui
python3 -m http.server 3000
```
Then navigate to: **`http://localhost:3000`** in your browser.

---

## Configuration
- **Backend URL**: Can be customized in the UI header (defaults to `http://localhost:8000`).
- **Threshold**: Can be adjusted dynamically via the settings gear icon (slider from 0.20 to 0.80, default 0.40).
- **Camera Source**: Choose between **Browser Webcam** (direct in-browser HTML5 video) or **Server OpenCV Feed** (`/cameras/stream`).
