# Standalone Document OCR Web Studio

A modern, standalone, zero-build web client for document text extraction, interactive bounding box inspection, and identity verification.

---

## 🚀 Features

- **Document Upload & Drag-and-Drop**: Easily browse or drop any document image (`PNG`, `JPG`, `WebP`, `BMP`, `TIFF`).
- **Interactive Region Visualizer**: Real-time canvas overlaying detected bounding boxes with confidence scores. Hover to inspect region text; click to jump directly to the line.
- **Full Text & Line-by-Line Extraction**: Tabbed view displaying the complete text and individual lines with confidence ratings.
- **Telemetry & Engine KPIs**: Display processing time in milliseconds, OCR engine (`PP-OCRv6`), lines count, and image preprocessing status (CLAHE contrast and deskew angle).
- **Flexible Backend Switching**: Built-in toggle to switch seamlessly between Cloud Remote (`http://103.144.195.105:8000`) and Local Python/Docker (`http://localhost:8000`).

---

## 💻 How to Run

### Method 1: Direct File Open
You can open `index.html` directly in any web browser (Chrome, Edge, Safari, Firefox):
```bash
open index.html
```

### Method 2: Lightweight Local HTTP Server (Recommended)
From within this directory, start any static HTTP server:

```bash
# Using Python 3
python3 -m http.server 3000

# Or using npx serve
npx serve .
```
Then visit: **[http://localhost:3000](http://localhost:3000)**
