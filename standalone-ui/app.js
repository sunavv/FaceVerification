/**
 * Face Verification Standalone User Interface
 * 1:1 Identity Matching between Document Photo and Live Camera
 */

(function () {
  'use strict';

  // ==========================================================================
  // State Management
  // ==========================================================================
  const state = {
    apiUrl: detectDefaultApiUrl(),
    threshold: 0.40,
    cameraMode: 'browser', // 'browser' or 'server'
    isCameraRunning: false,
    mediaStream: null,
    facingMode: 'user',

    // Document State
    documentFile: null,
    sessionId: null,
    documentFaceCropBase64: null,
    documentFaceQuality: 'GOOD',

    // Verification In Progress
    isVerifying: false,
  };

  function detectDefaultApiUrl() {
    if (window.location.protocol.startsWith('http') && window.location.port === '8000') {
      return `${window.location.protocol}//${window.location.hostname}:8000`;
    }
    return 'http://localhost:8000';
  }

  // ==========================================================================
  // DOM Elements
  // ==========================================================================
  const el = {
    // Header & Settings
    apiStatusPill: document.getElementById('api-status-pill'),
    apiStatusDot: document.getElementById('api-status-dot'),
    apiStatusText: document.getElementById('api-status-text'),
    btnSettingsToggle: document.getElementById('btn-settings-toggle'),
    settingsDrawer: document.getElementById('settings-drawer'),
    inputApiUrl: document.getElementById('input-api-url'),
    inputThreshold: document.getElementById('input-threshold'),
    thresholdValText: document.getElementById('threshold-val-text'),
    cameraModeSelect: document.getElementById('camera-mode-select'),
    toastMessage: document.getElementById('toast-message'),

    // Document Panel
    docStatusTag: document.getElementById('doc-status-tag'),
    documentDropzone: document.getElementById('document-dropzone'),
    documentFileInput: document.getElementById('document-file-input'),
    dropzonePrompt: document.getElementById('dropzone-prompt'),
    btnBrowseDoc: document.getElementById('btn-browse-doc'),
    docPreviewContainer: document.getElementById('doc-preview-container'),
    docPreviewImg: document.getElementById('doc-preview-img'),
    detectedFaceBadge: document.getElementById('detected-face-badge'),
    docFaceCropImg: document.getElementById('doc-face-crop-img'),
    docFaceQualityText: document.getElementById('doc-face-quality-text'),
    btnChangeDoc: document.getElementById('btn-change-doc'),
    docLoaderOverlay: document.getElementById('doc-loader-overlay'),

    // Camera Panel
    cameraStatusTag: document.getElementById('camera-status-tag'),
    cameraViewport: document.getElementById('camera-viewport'),
    webcamVideo: document.getElementById('webcam-video'),
    serverStreamImg: document.getElementById('server-stream-img'),
    faceTargetGuide: document.getElementById('face-target-guide'),
    cameraPlaceholder: document.getElementById('camera-placeholder'),
    btnStartCameraPlaceholder: document.getElementById('btn-start-camera-placeholder'),
    btnToggleCamera: document.getElementById('btn-toggle-camera'),
    cameraToggleIcon: document.getElementById('camera-toggle-icon'),
    cameraToggleText: document.getElementById('camera-toggle-text'),
    btnSwitchCamera: document.getElementById('btn-switch-camera'),

    // Compare Action
    btnVerify: document.getElementById('btn-verify'),
    btnVerifyText: document.getElementById('btn-verify-text'),
    actionHint: document.getElementById('action-hint'),

    // Result Section
    resultCard: document.getElementById('result-card'),
    btnCloseResult: document.getElementById('btn-close-result'),
    decisionBanner: document.getElementById('decision-banner'),
    decisionIconWrap: document.getElementById('decision-icon-wrap'),
    decisionTitle: document.getElementById('decision-title'),
    decisionDesc: document.getElementById('decision-desc'),
    metricSimilarityScore: document.getElementById('metric-similarity-score'),
    metricSimilarityRaw: document.getElementById('metric-similarity-raw'),
    metricThresholdVal: document.getElementById('metric-threshold-val'),
    metricConfidencePill: document.getElementById('metric-confidence-pill'),
    meterThresholdLabel: document.getElementById('meter-threshold-label'),
    meterThresholdLine: document.getElementById('meter-threshold-line'),
    meterFill: document.getElementById('meter-fill'),
    resultDocCrop: document.getElementById('result-doc-crop'),
    resultDocQuality: document.getElementById('result-doc-quality'),
    resultLiveCrop: document.getElementById('result-live-crop'),
    resultLiveQuality: document.getElementById('result-live-quality'),
    btnReverify: document.getElementById('btn-reverify'),

    // Offscreen Canvas
    snapshotCanvas: document.getElementById('snapshot-canvas'),
  };

  // ==========================================================================
  // Initialization
  // ==========================================================================
  function init() {
    el.inputApiUrl.value = state.apiUrl;
    el.inputThreshold.value = state.threshold;
    el.thresholdValText.innerText = Number(state.threshold).toFixed(2);
    updateMeterThresholdVisual(state.threshold);

    setupEventListeners();
    checkBackendHealth();
    startCamera();
  }

  // ==========================================================================
  // Backend Health Check
  // ==========================================================================
  async function checkBackendHealth() {
    el.apiStatusDot.className = 'status-dot';
    el.apiStatusText.innerText = 'Connecting...';

    try {
      const res = await fetch(`${state.apiUrl}/health`, { method: 'GET' });
      if (res.ok) {
        const data = await res.json();
        el.apiStatusDot.className = 'status-dot connected';
        el.apiStatusText.innerText = `Online (${data.face_model || 'InsightFace'})`;
        if (typeof data.similarity_threshold === 'number' && !localStorage.getItem('custom_threshold')) {
          state.threshold = data.similarity_threshold;
          el.inputThreshold.value = state.threshold;
          el.thresholdValText.innerText = Number(state.threshold).toFixed(2);
          updateMeterThresholdVisual(state.threshold);
        }
      } else {
        throw new Error(`HTTP ${res.status}`);
      }
    } catch (e) {
      el.apiStatusDot.className = 'status-dot disconnected';
      el.apiStatusText.innerText = 'Backend Offline';
      console.warn('Backend health check error:', e);
    }
  }

  // Periodically check health every 15 seconds
  setInterval(checkBackendHealth, 15000);

  // ==========================================================================
  // Event Listeners
  // ==========================================================================
  function setupEventListeners() {
    // Settings Drawer Toggle
    el.btnSettingsToggle.addEventListener('click', () => {
      el.settingsDrawer.classList.toggle('hidden');
    });

    el.apiStatusPill.addEventListener('click', () => {
      el.settingsDrawer.classList.toggle('hidden');
      if (!el.settingsDrawer.classList.contains('hidden')) {
        el.inputApiUrl.focus();
      }
    });

    // API URL Change
    el.inputApiUrl.addEventListener('change', (e) => {
      let url = e.target.value.trim().replace(/\/+$/, '');
      if (url && !/^https?:\/\//i.test(url)) {
        url = `http://${url}`;
        el.inputApiUrl.value = url;
      }
      state.apiUrl = url;
      checkBackendHealth();
      if (state.cameraMode === 'server' && state.isCameraRunning) {
        restartServerStream();
      }
    });

    // Threshold Slider
    el.inputThreshold.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      state.threshold = val;
      el.thresholdValText.innerText = val.toFixed(2);
      localStorage.setItem('custom_threshold', 'true');
      updateMeterThresholdVisual(val);
    });

    // Camera Mode Select
    el.cameraModeSelect.addEventListener('change', (e) => {
      state.cameraMode = e.target.value;
      restartCamera();
    });

    // Document Upload Interactions
    el.btnBrowseDoc.addEventListener('click', (e) => {
      e.stopPropagation();
      el.documentFileInput.click();
    });

    el.documentDropzone.addEventListener('click', (e) => {
      if (!state.sessionId && !e.target.closest('#btn-change-doc')) {
        el.documentFileInput.click();
      }
    });

    el.documentFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleDocumentFile(e.target.files[0]);
      }
    });

    // Drag and drop
    el.documentDropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      el.documentDropzone.classList.add('dragover');
    });

    el.documentDropzone.addEventListener('dragleave', () => {
      el.documentDropzone.classList.remove('dragover');
    });

    el.documentDropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      el.documentDropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleDocumentFile(e.dataTransfer.files[0]);
      }
    });

    el.btnChangeDoc.addEventListener('click', (e) => {
      e.stopPropagation();
      resetDocumentState();
      el.documentFileInput.click();
    });

    // Camera Controls
    el.btnToggleCamera.addEventListener('click', toggleCamera);
    el.btnStartCameraPlaceholder.addEventListener('click', startCamera);
    el.btnSwitchCamera.addEventListener('click', switchCamera);

    // Verify Action
    el.btnVerify.addEventListener('click', handleVerifyClick);

    // Result Dismiss / Reverify
    el.btnCloseResult.addEventListener('click', () => {
      el.resultCard.classList.add('hidden');
    });

    el.btnReverify.addEventListener('click', () => {
      el.resultCard.classList.add('hidden');
      handleVerifyClick();
    });
  }

  // ==========================================================================
  // Document Processing
  // ==========================================================================
  async function handleDocumentFile(file) {
    if (!file) return;

    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type) && !/\.(jpe?g|png|webp)$/i.test(file.name)) {
      showToast('Please upload a valid image file (JPG, PNG, or WEBP).', 'error');
      return;
    }

    state.documentFile = file;

    // Display instant local preview
    const reader = new FileReader();
    reader.onload = (e) => {
      el.docPreviewImg.src = e.target.result;
      el.dropzonePrompt.classList.add('hidden');
      el.docPreviewContainer.classList.remove('hidden');
    };
    reader.readAsDataURL(file);

    // Show Analyzing State
    el.docLoaderOverlay.classList.remove('hidden');
    el.docStatusTag.innerText = 'Analyzing...';
    el.docStatusTag.className = 'status-tag analyzing';
    el.detectedFaceBadge.classList.add('hidden');
    updateVerifyButtonState();

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(`${state.apiUrl}/document/analyze`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      el.docLoaderOverlay.classList.add('hidden');

      if (!res.ok || !data.success || !data.document) {
        const errMsg = (data.error && data.error.message) || 'Failed to detect a valid face in document.';
        showToast(errMsg, 'error');
        resetDocumentState();
        return;
      }

      const doc = data.document;
      state.sessionId = doc.session_id;
      state.documentFaceCropBase64 = doc.face_image_base64 || doc.raw_face_image_base64;
      state.documentFaceQuality = doc.face_quality || 'GOOD';

      // Update UI with detected face crop thumbnail
      if (state.documentFaceCropBase64) {
        el.docFaceCropImg.src = state.documentFaceCropBase64;
        el.docFaceQualityText.innerText = `Quality: ${state.documentFaceQuality}`;
        el.detectedFaceBadge.classList.remove('hidden');
      }

      el.docStatusTag.innerText = '✓ Face Detected';
      el.docStatusTag.className = 'status-tag active';

      showToast('Document analyzed successfully! Face extracted.', 'success');
      updateVerifyButtonState();
    } catch (err) {
      console.error('Error analyzing document:', err);
      el.docLoaderOverlay.classList.add('hidden');
      showToast(`Document analysis error: ${err.message}. Is backend running on ${state.apiUrl}?`, 'error');
      resetDocumentState();
    }
  }

  function resetDocumentState() {
    state.documentFile = null;
    state.sessionId = null;
    state.documentFaceCropBase64 = null;

    el.documentFileInput.value = '';
    el.docPreviewImg.src = '';
    el.docFaceCropImg.src = '';
    el.docPreviewContainer.classList.add('hidden');
    el.detectedFaceBadge.classList.add('hidden');
    el.dropzonePrompt.classList.remove('hidden');

    el.docStatusTag.innerText = 'Waiting for upload';
    el.docStatusTag.className = 'status-tag';

    updateVerifyButtonState();
  }

  // ==========================================================================
  // Camera Management (Browser Webcam & Server Stream)
  // ==========================================================================
  async function startCamera() {
    if (state.cameraMode === 'browser') {
      await startBrowserCamera();
    } else {
      startServerStream();
    }
  }

  async function startBrowserCamera() {
    stopMediaTracks();

    el.serverStreamImg.classList.add('hidden');
    el.serverStreamImg.src = '';
    el.webcamVideo.classList.remove('hidden');
    el.cameraPlaceholder.classList.add('hidden');
    el.faceTargetGuide.classList.remove('hidden');

    try {
      const constraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: state.facingMode,
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      state.mediaStream = stream;
      el.webcamVideo.srcObject = stream;
      state.isCameraRunning = true;

      el.cameraStatusTag.innerText = 'Camera active';
      el.cameraStatusTag.className = 'status-tag active';
      el.cameraToggleIcon.innerText = '⏹';
      el.cameraToggleText.innerText = 'Pause Camera';

      updateVerifyButtonState();
    } catch (err) {
      console.warn('Browser getUserMedia failed, offering server camera fallback:', err);
      showToast('Browser camera access failed or denied. Falling back to server stream.', 'error');
      state.cameraMode = 'server';
      el.cameraModeSelect.value = 'server';
      startServerStream();
    }
  }

  function startServerStream() {
    stopMediaTracks();

    el.webcamVideo.classList.add('hidden');
    el.webcamVideo.srcObject = null;
    el.serverStreamImg.classList.remove('hidden');
    el.cameraPlaceholder.classList.add('hidden');
    el.faceTargetGuide.classList.remove('hidden');

    restartServerStream();

    state.isCameraRunning = true;
    el.cameraStatusTag.innerText = 'Server feed active';
    el.cameraStatusTag.className = 'status-tag active';
    el.cameraToggleIcon.innerText = '⏹';
    el.cameraToggleText.innerText = 'Pause Camera';

    updateVerifyButtonState();
  }

  function restartServerStream() {
    const sidParam = state.sessionId ? `&session_id=${encodeURIComponent(state.sessionId)}` : '';
    const threshParam = `&threshold=${state.threshold}`;
    el.serverStreamImg.src = `${state.apiUrl}/cameras/stream?t=${Date.now()}${sidParam}${threshParam}`;
  }

  function stopMediaTracks() {
    if (state.mediaStream) {
      state.mediaStream.getTracks().forEach((track) => track.stop());
      state.mediaStream = null;
    }
  }

  function stopCamera() {
    stopMediaTracks();
    if (state.cameraMode === 'server') {
      el.serverStreamImg.src = '';
    }

    state.isCameraRunning = false;
    el.webcamVideo.classList.add('hidden');
    el.serverStreamImg.classList.add('hidden');
    el.cameraPlaceholder.classList.remove('hidden');
    el.faceTargetGuide.classList.add('hidden');

    el.cameraStatusTag.innerText = 'Camera paused';
    el.cameraStatusTag.className = 'status-tag';
    el.cameraToggleIcon.innerText = '▶';
    el.cameraToggleText.innerText = 'Start Camera';

    updateVerifyButtonState();
  }

  function toggleCamera() {
    if (state.isCameraRunning) {
      stopCamera();
    } else {
      startCamera();
    }
  }

  function restartCamera() {
    stopCamera();
    startCamera();
  }

  function switchCamera() {
    if (state.cameraMode === 'browser') {
      state.facingMode = state.facingMode === 'user' ? 'environment' : 'user';
      startBrowserCamera();
    } else {
      // In server mode, ask backend to cycle camera index
      fetch(`${state.apiUrl}/cameras/select?index=0`, { method: 'POST' }).catch(() => {});
      restartServerStream();
    }
  }

  // ==========================================================================
  // Capture Snapshot from Browser Video
  // ==========================================================================
  function captureBrowserVideoBlob() {
    return new Promise((resolve, reject) => {
      const video = el.webcamVideo;
      if (!video || !video.videoWidth || !video.videoHeight) {
        reject(new Error('Video feed is not ready or has zero dimensions.'));
        return;
      }

      const canvas = el.snapshotCanvas;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');

      // Unmirror image when drawing to canvas
      ctx.save();
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      ctx.restore();

      canvas.toBlob((blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Failed to create snapshot blob from canvas.'));
        }
      }, 'image/jpeg', 0.92);
    });
  }

  // ==========================================================================
  // Verification Comparison Action
  // ==========================================================================
  async function handleVerifyClick() {
    if (!state.sessionId) {
      showToast('Please upload an identity document first.', 'error');
      return;
    }

    if (!state.isCameraRunning) {
      showToast('Please start the camera feed first.', 'error');
      return;
    }

    if (state.isVerifying) return;

    state.isVerifying = true;
    el.btnVerify.disabled = true;
    el.btnVerifyText.innerText = 'Comparing Faces...';
    hideToast();

    try {
      const formData = new FormData();
      formData.append('session_id', state.sessionId);
      formData.append('threshold', state.threshold);

      // If browser camera is running, capture the current frame and send as live_image
      if (state.cameraMode === 'browser') {
        const liveBlob = await captureBrowserVideoBlob();
        formData.append('live_image', liveBlob, 'camera_capture.jpg');
      }

      const response = await fetch(`${state.apiUrl}/verification/compare`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || !data.success || !data.verification) {
        const errMsg = (data.error && data.error.message) || 'Verification failed.';
        showToast(errMsg, 'error');
        return;
      }

      renderVerificationResult(data.verification);
    } catch (err) {
      console.error('Verification comparison error:', err);
      showToast(`Verification error: ${err.message}`, 'error');
    } finally {
      state.isVerifying = false;
      el.btnVerify.disabled = false;
      el.btnVerifyText.innerText = 'Compare Faces';
      updateVerifyButtonState();
    }
  }

  // ==========================================================================
  // Result Presentation: MATCH or NOT MATCH with Similarity Score
  // ==========================================================================
  function renderVerificationResult(v) {
    const isMatch = Boolean(v.face_match);
    const sim = typeof v.similarity === 'number' ? v.similarity : 0.0;
    const thresh = typeof v.threshold === 'number' ? v.threshold : state.threshold;
    const confidence = v.confidence || (isMatch ? 'HIGH' : 'REJECT');

    // 1. Decision Banner Styling
    el.decisionBanner.className = `decision-banner ${isMatch ? 'match' : 'not-match'}`;
    el.decisionTitle.innerText = isMatch ? 'MATCH' : 'NOT MATCH';
    el.decisionDesc.innerText = isMatch
      ? 'The person in the camera feed matches the identity in the document.'
      : 'Face similarity is below the required threshold. Identity does not match.';

    el.decisionIconWrap.innerHTML = isMatch
      ? '<svg viewBox="0 0 24 24" width="30" height="30" stroke="currentColor" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>'
      : '<svg viewBox="0 0 24 24" width="30" height="30" stroke="currentColor" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>';

    // 2. Metrics & Similarity Score
    const percentStr = `${(sim * 100).toFixed(1)}%`;
    el.metricSimilarityScore.innerText = percentStr;
    el.metricSimilarityRaw.innerText = `(${sim.toFixed(4)})`;

    el.metricThresholdVal.innerText = thresh.toFixed(2);

    // Confidence Pill
    el.metricConfidencePill.innerText = confidence;
    el.metricConfidencePill.className = `confidence-pill ${confidence.toLowerCase()}`;

    // 3. Meter Fill & Threshold Visual
    const clampedWidth = Math.max(0, Math.min(100, sim * 100));
    el.meterFill.style.width = `${clampedWidth}%`;
    el.meterFill.className = `meter-fill ${isMatch ? '' : 'fail'}`;

    // 4. Comparison Faces Side-by-side
    const docImg = v.reference_face_image_base64 || state.documentFaceCropBase64;
    if (docImg) {
      el.resultDocCrop.src = docImg;
    }
    el.resultDocQuality.innerText = `Quality: ${v.reference_face_quality || state.documentFaceQuality || 'Good'}`;

    const liveImg = v.live_face_image_base64 || v.live_raw_face_image_base64;
    if (liveImg) {
      el.resultLiveCrop.src = liveImg;
    } else {
      el.resultLiveCrop.src = docImg || '';
    }
    el.resultLiveQuality.innerText = `Quality: ${v.live_face_quality || 'Good'}`;

    // 5. Reveal Result Card & Scroll Into View
    el.resultCard.classList.remove('hidden');
    el.resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // ==========================================================================
  // Helper UI Utilities
  // ==========================================================================
  function updateMeterThresholdVisual(thresholdVal) {
    const leftPct = `${thresholdVal * 100}%`;
    el.meterThresholdLine.style.left = leftPct;
    el.meterThresholdLabel.innerText = `Threshold: ${Number(thresholdVal).toFixed(2)}`;
  }

  function updateVerifyButtonState() {
    const canVerify = Boolean(state.sessionId && state.isCameraRunning && !state.isVerifying);
    el.btnVerify.disabled = !canVerify;

    if (!state.sessionId) {
      el.actionHint.innerText = 'Upload a document photo above to enable verification';
    } else if (!state.isCameraRunning) {
      el.actionHint.innerText = 'Start camera feed above to compare faces';
    } else {
      el.actionHint.innerText = 'Ready! Click to compare document and live camera face';
    }
  }

  function showToast(message, type = 'error') {
    el.toastMessage.innerText = message;
    el.toastMessage.className = `toast ${type}`;
    el.toastMessage.classList.remove('hidden');

    setTimeout(() => {
      hideToast();
    }, 6000);
  }

  function hideToast() {
    el.toastMessage.classList.add('hidden');
  }

  // Initialize on DOM Ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
