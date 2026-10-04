/**
 * Document OCR Studio - Web Client Application
 * Vanilla ES6 JavaScript Implementation
 */

(function () {
  'use strict';

  // State Management
  const state = {
    serverUrl: 'http://103.144.195.105:8000',
    currentFile: null,
    currentImage: null,
    ocrResponse: null,
    selectedLineIndex: -1,
    hoveredLineIndex: -1,
    canvasTransform: { scale: 1, offsetX: 0, offsetY: 0 },
    isConnected: false,
  };

  // DOM Elements
  const el = {
    serverUrlInput: document.getElementById('server-url-input'),
    btnPresetRemote: document.getElementById('btn-preset-remote'),
    btnPresetLocal: document.getElementById('btn-preset-local'),
    btnCheckConnection: document.getElementById('btn-check-connection'),
    connectionStatusBadge: document.getElementById('connection-status-badge'),
    connectionStatusText: document.getElementById('connection-status-text'),

    fileInput: document.getElementById('file-input'),
    dropZone: document.getElementById('drop-zone'),
    btnBrowseFile: document.getElementById('btn-browse-file'),

    docFilename: document.getElementById('doc-filename'),
    docResolution: document.getElementById('doc-resolution'),
    docSize: document.getElementById('doc-size'),

    canvasContainer: document.getElementById('canvas-container'),
    canvas: document.getElementById('document-canvas'),
    canvasPlaceholder: document.getElementById('canvas-placeholder'),
    canvasTooltip: document.getElementById('canvas-tooltip'),
    toggleBoundingBoxes: document.getElementById('toggle-bounding-boxes'),
    toggleFastMode: document.getElementById('toggle-fast-mode'),

    btnRunOcr: document.getElementById('btn-run-ocr'),
    ocrSpinner: document.getElementById('ocr-spinner'),
    ocrButtonText: document.getElementById('ocr-button-text'),

    kpiEngine: document.getElementById('kpi-engine'),
    kpiTime: document.getElementById('kpi-time'),
    kpiRoundtrip: document.getElementById('kpi-roundtrip'),
    kpiLines: document.getElementById('kpi-lines'),
    kpiPreprocessing: document.getElementById('kpi-preprocessing'),

    tabButtons: document.querySelectorAll('.tab-btn'),
    tabContents: document.querySelectorAll('.tab-content'),
    tabLinesCount: document.getElementById('tab-lines-count'),

    linesList: document.getElementById('lines-list'),
    fullTextOutput: document.getElementById('full-text-output'),
    fulltextCharCount: document.getElementById('fulltext-char-count'),
    fulltextWordCount: document.getElementById('fulltext-word-count'),
    jsonOutput: document.getElementById('json-output'),

    btnCopyLines: document.getElementById('btn-copy-lines'),
    btnCopyFulltext: document.getElementById('btn-copy-fulltext'),
    btnCopyJson: document.getElementById('btn-copy-json'),

    toastContainer: document.getElementById('toast-container'),
  };

  const ctx = el.canvas.getContext('2d');

  // ==========================================================================
  // Initialization
  // ==========================================================================
  function init() {
    setupEventListeners();
    checkServerConnection();
    resizeCanvas();
  }

  // ==========================================================================
  // Event Listeners
  // ==========================================================================
  function setupEventListeners() {
    // Server Configuration
    el.serverUrlInput.addEventListener('change', () => {
      state.serverUrl = sanitizeUrl(el.serverUrlInput.value);
      checkServerConnection();
    });

    el.btnPresetRemote.addEventListener('click', () => {
      setServerPreset('http://103.144.195.105:8000', el.btnPresetRemote);
    });

    el.btnPresetLocal.addEventListener('click', () => {
      setServerPreset('http://localhost:8000', el.btnPresetLocal);
    });

    el.btnCheckConnection.addEventListener('click', checkServerConnection);

    // File Upload & Drag-and-Drop
    el.btnBrowseFile.addEventListener('click', () => el.fileInput.click());
    el.dropZone.addEventListener('click', (e) => {
      if (e.target !== el.btnBrowseFile) el.fileInput.click();
    });

    el.fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelect(e.target.files[0]);
      }
    });

    ['dragenter', 'dragover'].forEach((eventName) => {
      el.dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        el.dropZone.classList.add('drag-active');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      el.dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        el.dropZone.classList.remove('drag-active');
      });
    });

    el.dropZone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });

    // Canvas Interactions & Hover
    el.toggleBoundingBoxes.addEventListener('change', drawCanvas);
    el.canvas.addEventListener('mousemove', handleCanvasMouseMove);
    el.canvas.addEventListener('mouseleave', handleCanvasMouseLeave);
    el.canvas.addEventListener('click', handleCanvasClick);
    window.addEventListener('resize', () => {
      resizeCanvas();
      drawCanvas();
    });

    // OCR Action
    el.btnRunOcr.addEventListener('click', runOcrExtraction);

    // Tab Switching
    el.tabButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const targetTab = btn.getAttribute('data-tab');
        el.tabButtons.forEach((b) => b.classList.remove('active'));
        el.tabContents.forEach((c) => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(targetTab).classList.add('active');
      });
    });

    // Copy Buttons
    el.btnCopyFulltext.addEventListener('click', () => {
      if (el.fullTextOutput.value) copyToClipboard(el.fullTextOutput.value, 'Full document text copied!');
    });

    el.btnCopyLines.addEventListener('click', () => {
      if (state.ocrResponse && state.ocrResponse.lines && state.ocrResponse.lines.length) {
        const linesFormatted = state.ocrResponse.lines
          .map((l) => `[${(l.confidence * 100).toFixed(1)}%] ${l.text}`)
          .join('\n');
        copyToClipboard(linesFormatted, 'Recognized lines copied!');
      }
    });

    el.btnCopyJson.addEventListener('click', () => {
      if (state.ocrResponse) {
        copyToClipboard(JSON.stringify(state.ocrResponse, null, 2), 'Raw response JSON copied!');
      }
    });
  }

  // ==========================================================================
  // Server Management & Health Check
  // ==========================================================================
  function sanitizeUrl(url) {
    let clean = url.trim();
    if (clean.endsWith('/')) clean = clean.slice(0, -1);
    return clean;
  }

  function setServerPreset(url, activeBtn) {
    state.serverUrl = url;
    el.serverUrlInput.value = url;
    el.btnPresetRemote.classList.remove('active');
    el.btnPresetLocal.classList.remove('active');
    activeBtn.classList.add('active');
    checkServerConnection();
  }

  async function checkServerConnection() {
    el.connectionStatusBadge.className = 'connection-badge status-checking';
    el.connectionStatusText.textContent = 'Checking...';

    const healthUrl = `${state.serverUrl}/api/v1/health`;
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 6000);

      const resp = await fetch(healthUrl, {
        method: 'GET',
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (resp.ok) {
        const data = await resp.json();
        state.isConnected = true;
        el.connectionStatusBadge.className = 'connection-badge status-connected';
        el.connectionStatusText.textContent = 'Connected';
        showToast('Connected to server', 'success');
      } else {
        throw new Error(`HTTP ${resp.status}`);
      }
    } catch (err) {
      state.isConnected = false;
      el.connectionStatusBadge.className = 'connection-badge status-disconnected';
      el.connectionStatusText.textContent = 'Disconnected';
    }
  }

  // ==========================================================================
  // File & Document Loading
  // ==========================================================================
  function handleFileSelect(file) {
    if (!file) return;

    // Validate type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/bmp', 'image/tiff'];
    if (!validTypes.includes(file.type) && !/\.(png|jpe?g|webp|bmp|tiff?)$/i.test(file.name)) {
      showToast('Unsupported file type. Please upload a PNG, JPG, WebP, BMP, or TIFF image.', 'error');
      return;
    }

    state.currentFile = file;
    state.ocrResponse = null;
    state.selectedLineIndex = -1;
    state.hoveredLineIndex = -1;

    // Update Meta Bar
    el.docFilename.textContent = file.name;
    el.docSize.textContent = formatBytes(file.size);

    // Read image preview
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        state.currentImage = img;
        el.docResolution.textContent = `${img.width} × ${img.height} px`;
        el.canvasPlaceholder.style.display = 'none';
        el.btnRunOcr.disabled = false;
        resetResultsUI();
        resizeCanvas();
        drawCanvas();
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  // ==========================================================================
  // Canvas Rendering & Highlighting
  // ==========================================================================
  function resizeCanvas() {
    const container = el.canvasContainer;
    const rect = container.getBoundingClientRect();
    if (rect.width === 0) return;

    // High-DPI Canvas scaling
    const dpr = window.devicePixelRatio || 1;
    const targetWidth = rect.width;
    const targetHeight = Math.max(380, Math.min(520, window.innerHeight * 0.55));

    el.canvas.width = targetWidth * dpr;
    el.canvas.height = targetHeight * dpr;
    el.canvas.style.width = `${targetWidth}px`;
    el.canvas.style.height = `${targetHeight}px`;

    ctx.scale(dpr, dpr);
  }

  function drawCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const displayW = el.canvas.width / dpr;
    const displayH = el.canvas.height / dpr;

    ctx.clearRect(0, 0, displayW, displayH);

    if (!state.currentImage) {
      el.canvasPlaceholder.style.display = 'flex';
      return;
    }

    el.canvasPlaceholder.style.display = 'none';

    // Calculate aspect ratio fit
    const imgW = state.currentImage.width;
    const imgH = state.currentImage.height;
    const scale = Math.min(displayW / imgW, displayH / imgH);
    const drawW = imgW * scale;
    const drawH = imgH * scale;
    const offsetX = (displayW - drawW) / 2;
    const offsetY = (displayH - drawH) / 2;

    state.canvasTransform = { scale, offsetX, offsetY };

    // Render document image
    ctx.drawImage(state.currentImage, offsetX, offsetY, drawW, drawH);

    // Render bounding boxes if enabled
    if (
      el.toggleBoundingBoxes.checked &&
      state.ocrResponse &&
      Array.isArray(state.ocrResponse.lines)
    ) {
      state.ocrResponse.lines.forEach((line, idx) => {
        const bbox = line.bbox;
        if (!bbox || bbox.length < 4) return;

        const x1 = offsetX + bbox[0] * scale;
        const y1 = offsetY + bbox[1] * scale;
        const x2 = offsetX + bbox[2] * scale;
        const y2 = offsetY + bbox[3] * scale;
        const boxW = x2 - x1;
        const boxH = y2 - y1;

        const isSelected = idx === state.selectedLineIndex;
        const isHovered = idx === state.hoveredLineIndex;

        ctx.save();
        if (isSelected) {
          ctx.strokeStyle = '#000000';
          ctx.lineWidth = 2;
          ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
          ctx.fillRect(x1, y1, boxW, boxH);
          ctx.strokeRect(x1, y1, boxW, boxH);
        } else if (isHovered) {
          ctx.strokeStyle = '#3f3f46';
          ctx.lineWidth = 1.5;
          ctx.fillStyle = 'rgba(0, 0, 0, 0.08)';
          ctx.fillRect(x1, y1, boxW, boxH);
          ctx.strokeRect(x1, y1, boxW, boxH);
        } else {
          ctx.strokeStyle = '#71717a';
          ctx.lineWidth = 1;
          ctx.fillStyle = 'rgba(0, 0, 0, 0.03)';
          ctx.fillRect(x1, y1, boxW, boxH);
          ctx.strokeRect(x1, y1, boxW, boxH);
        }
        ctx.restore();
      });
    }
  }

  function getCanvasMouseCoords(event) {
    const rect = el.canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    return { mouseX, mouseY };
  }

  function findLineAtCoords(mouseX, mouseY) {
    if (!state.ocrResponse || !state.ocrResponse.lines) return -1;
    const { scale, offsetX, offsetY } = state.canvasTransform;

    for (let i = 0; i < state.ocrResponse.lines.length; i++) {
      const bbox = state.ocrResponse.lines[i].bbox;
      if (!bbox || bbox.length < 4) continue;

      const x1 = offsetX + bbox[0] * scale;
      const y1 = offsetY + bbox[1] * scale;
      const x2 = offsetX + bbox[2] * scale;
      const y2 = offsetY + bbox[3] * scale;

      if (mouseX >= x1 && mouseX <= x2 && mouseY >= y1 && mouseY <= y2) {
        return i;
      }
    }
    return -1;
  }

  function handleCanvasMouseMove(e) {
    if (!el.toggleBoundingBoxes.checked || !state.ocrResponse) return;
    const { mouseX, mouseY } = getCanvasMouseCoords(e);
    const hitIndex = findLineAtCoords(mouseX, mouseY);

    if (hitIndex !== state.hoveredLineIndex) {
      state.hoveredLineIndex = hitIndex;
      drawCanvas();

      if (hitIndex !== -1) {
        const line = state.ocrResponse.lines[hitIndex];
        el.canvasTooltip.classList.remove('hidden');
        el.canvasTooltip.innerHTML = `<strong>${escapeHtml(line.text)}</strong><br/><span style="color:#06b6d4">Confidence: ${(line.confidence * 100).toFixed(1)}%</span>`;
        el.canvasTooltip.style.left = `${mouseX}px`;
        el.canvasTooltip.style.top = `${mouseY}px`;
      } else {
        el.canvasTooltip.classList.add('hidden');
      }
    } else if (hitIndex !== -1) {
      el.canvasTooltip.style.left = `${mouseX}px`;
      el.canvasTooltip.style.top = `${mouseY}px`;
    }
  }

  function handleCanvasMouseLeave() {
    if (state.hoveredLineIndex !== -1) {
      state.hoveredLineIndex = -1;
      el.canvasTooltip.classList.add('hidden');
      drawCanvas();
    }
  }

  function handleCanvasClick(e) {
    if (!state.ocrResponse) return;
    const { mouseX, mouseY } = getCanvasMouseCoords(e);
    const hitIndex = findLineAtCoords(mouseX, mouseY);

    if (hitIndex !== -1) {
      selectLine(hitIndex);
    }
  }

  function selectLine(index) {
    state.selectedLineIndex = index;
    drawCanvas();

    // Update list view selection
    const items = el.linesList.querySelectorAll('.line-item');
    items.forEach((item, idx) => {
      if (idx === index) {
        item.classList.add('selected');
        item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      } else {
        item.classList.remove('selected');
      }
    });

    // Switch to Lines tab if not active
    const linesTabBtn = document.querySelector('.tab-btn[data-tab="tab-lines"]');
    if (!linesTabBtn.classList.contains('active')) {
      linesTabBtn.click();
    }
  }

  // ==========================================================================
  // Client-Side Image Pre-scaling & Optimization (Fast Mode)
  // ==========================================================================
  async function prepareOptimizedUpload(file) {
    if (!el.toggleFastMode || !el.toggleFastMode.checked || !state.currentImage) {
      return { blob: file, scaleFactor: 1.0, isOptimized: false };
    }

    const img = state.currentImage;
    const maxDim = 1600;
    const origW = img.naturalWidth || img.width;
    const origH = img.naturalHeight || img.height;

    // Skip optimization if already within optimal OCR dimensions
    if (origW <= maxDim && origH <= maxDim) {
      return { blob: file, scaleFactor: 1.0, isOptimized: false };
    }

    const scale = Math.min(maxDim / origW, maxDim / origH);
    const targetW = Math.round(origW * scale);
    const targetH = Math.round(origH * scale);
    const scaleFactor = origW / targetW;

    const offscreen = document.createElement('canvas');
    offscreen.width = targetW;
    offscreen.height = targetH;
    const offCtx = offscreen.getContext('2d');
    offCtx.imageSmoothingEnabled = true;
    offCtx.imageSmoothingQuality = 'high';
    offCtx.drawImage(img, 0, 0, targetW, targetH);

    const blob = await new Promise((resolve) => {
      offscreen.toBlob(resolve, 'image/jpeg', 0.92);
    });

    return {
      blob: blob || file,
      scaleFactor,
      isOptimized: true,
      originalDim: `${origW}×${origH}`,
      optimizedDim: `${targetW}×${targetH}`,
    };
  }

  // ==========================================================================
  // OCR Execution & API Calling
  // ==========================================================================
  async function runOcrExtraction() {
    if (!state.currentFile) return;

    // Loading State
    el.btnRunOcr.disabled = true;
    el.ocrSpinner.classList.remove('hidden');
    el.ocrButtonText.textContent = 'Processing...';

    const startTime = performance.now();

    try {
      const optimization = await prepareOptimizedUpload(state.currentFile);
      const uploadFile = optimization.isOptimized
        ? new File([optimization.blob], state.currentFile.name.replace(/\.[^.]+$/, '.jpg'), { type: 'image/jpeg' })
        : state.currentFile;

      const formData = new FormData();
      formData.append('file', uploadFile);

      const endpoint = `${state.serverUrl}/api/v1/ocr`;
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      const roundtripMs = performance.now() - startTime;
      el.kpiRoundtrip.textContent = `${roundtripMs.toFixed(1)} ms`;

      if (!response.ok || !data.success) {
        throw new Error(data.error || `Server returned error HTTP ${response.status}`);
      }

      // If pre-scaled, map coordinates back to original image coordinate space
      if (optimization.isOptimized && optimization.scaleFactor !== 1.0 && Array.isArray(data.lines)) {
        const factor = optimization.scaleFactor;
        data.lines.forEach((line) => {
          if (line.bbox && line.bbox.length === 4) {
            line.bbox = line.bbox.map((coord) => Math.round(coord * factor));
          }
          if (Array.isArray(line.polygon)) {
            line.polygon = line.polygon.map((pt) => [
              Math.round(pt[0] * factor),
              Math.round(pt[1] * factor),
            ]);
          }
        });
      }

      state.ocrResponse = data;

      // Populate UI with retrieved data
      renderOcrResults(data);

      const speedupNotice = optimization.isOptimized
        ? ` (Fast Mode: Pre-scaled ${optimization.originalDim} → ${optimization.optimizedDim})`
        : '';
      showToast('Information extracted successfully', 'success');
    } catch (err) {
      console.error('OCR Extraction error:', err);
      handleOcrError(err.message || 'Unknown network error');
    } finally {
      el.btnRunOcr.disabled = false;
      el.ocrSpinner.classList.add('hidden');
      el.ocrButtonText.textContent = 'Extract Information';
    }
  }

  function renderOcrResults(data) {
    // 1. KPI Telemetry
    el.kpiEngine.textContent = data.engine || 'paddleocr_ppocrv6';
    el.kpiTime.textContent = `${data.processing_time_ms ? data.processing_time_ms.toFixed(1) : '0'} ms`;
    el.kpiLines.textContent = `${(data.lines || []).length} lines`;

    if (data.preprocessing && data.preprocessing.applied) {
      const p = data.preprocessing;
      const deskew = p.deskew_angle != null ? `${p.deskew_angle.toFixed(1)}°` : 'None';
      const clahe = p.clahe_applied ? 'CLAHE: Yes' : 'CLAHE: No';
      el.kpiPreprocessing.textContent = `${clahe} | ${deskew}`;
    } else {
      el.kpiPreprocessing.textContent = 'None';
    }

    // 2. Extracted Full Text Tab
    el.fullTextOutput.value = data.text || '';
    const charCount = (data.text || '').length;
    const wordCount = (data.text || '').trim() ? data.text.trim().split(/\s+/).length : 0;
    el.fulltextCharCount.textContent = `${charCount} characters`;
    el.fulltextWordCount.textContent = `${wordCount} words`;

    // 3. Recognized Lines Tab
    renderLinesList(data.lines || []);

    // 4. Raw JSON Tab
    el.jsonOutput.innerHTML = `<code>${escapeHtml(JSON.stringify(data, null, 2))}</code>`;

    // 5. Redraw Canvas with text regions
    drawCanvas();
  }

  function renderLinesList(lines) {
    el.tabLinesCount.textContent = lines.length;
    el.linesList.innerHTML = '';

    if (lines.length === 0) {
      el.linesList.innerHTML = `
        <div class="empty-state">
          <p>No text lines recognized in this document.</p>
        </div>`;
      return;
    }

    lines.forEach((line, idx) => {
      const lineItem = document.createElement('div');
      lineItem.className = 'line-item';
      lineItem.setAttribute('data-index', idx);

      const conf = line.confidence * 100;
      let confClass = 'confidence-high';
      if (conf < 75) confClass = 'confidence-low';
      else if (conf < 90) confClass = 'confidence-medium';

      lineItem.innerHTML = `
        <span class="line-text">${escapeHtml(line.text)}</span>
        <span class="confidence-chip ${confClass}">${conf.toFixed(1)}%</span>
      `;

      lineItem.addEventListener('click', () => selectLine(idx));
      lineItem.addEventListener('mouseenter', () => {
        state.hoveredLineIndex = idx;
        drawCanvas();
      });
      lineItem.addEventListener('mouseleave', () => {
        state.hoveredLineIndex = -1;
        drawCanvas();
      });

      el.linesList.appendChild(lineItem);
    });
  }

  function handleOcrError(errorMsg) {
    el.fullTextOutput.value = `Error during OCR extraction:\n${errorMsg}`;
    el.linesList.innerHTML = `<div class="empty-state"><p style="color:#fb7185">Extraction failed: ${escapeHtml(errorMsg)}</p></div>`;
    el.tabLinesCount.textContent = '0';
    el.kpiEngine.textContent = '-';
    el.kpiTime.textContent = '0 ms';
    el.kpiRoundtrip.textContent = '-';
    el.kpiLines.textContent = '0';
    el.kpiPreprocessing.textContent = 'Error';
    showToast(errorMsg, 'error');
  }

  function resetResultsUI() {
    el.fullTextOutput.value = '';
    el.fulltextCharCount.textContent = '0 characters';
    el.fulltextWordCount.textContent = '0 words';
    el.linesList.innerHTML = `
      <div class="empty-state">
        <p>No OCR lines extracted yet. Click "Run OCR Extraction" to analyze.</p>
      </div>`;
    el.tabLinesCount.textContent = '0';
    el.jsonOutput.innerHTML = `<code>// Raw OCR response JSON will be rendered here...</code>`;
    el.kpiEngine.textContent = '-';
    el.kpiTime.textContent = '-';
    el.kpiRoundtrip.textContent = '-';
    el.kpiLines.textContent = '-';
    el.kpiPreprocessing.textContent = '-';
  }

  // ==========================================================================
  // Helper Utilities
  // ==========================================================================
  function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function copyToClipboard(text, toastMsg) {
    navigator.clipboard
      .writeText(text)
      .then(() => showToast(toastMsg || 'Copied to clipboard!', 'success'))
      .catch(() => showToast('Failed to copy to clipboard.', 'error'));
  }

  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const icon = type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ';
    toast.innerHTML = `<span style="font-weight:700;">${icon}</span><span>${escapeHtml(message)}</span>`;
    el.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.25s ease-out';
      setTimeout(() => toast.remove(), 250);
    }, 3200);
  }

  // Run initial setup
  init();
})();
