document.addEventListener('DOMContentLoaded', () => {
    // State
    let activeSessionId = null;
    let selectedFile = null;
    let isCameraActive = false;
    let configuredThreshold = 0.65;

    // Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('document-input');
    const placeholder = document.getElementById('dropzone-placeholder');
    const previewContainer = document.getElementById('preview-container');
    const previewImg = document.getElementById('document-preview-img');
    const btnClearDoc = document.getElementById('btn-clear-doc');
    const btnAnalyze = document.getElementById('btn-analyze');
    
    const docAnalysisBox = document.getElementById('doc-analysis-box');
    const badgeDocFace = document.getElementById('badge-doc-face');
    const docFaceCount = document.getElementById('doc-face-count');
    const docOcrName = document.getElementById('doc-ocr-name');
    const expectedNameInput = document.getElementById('expected-name-input');
    
    const cameraSelect = document.getElementById('camera-select');
    const btnToggleCam = document.getElementById('btn-toggle-cam');
    const camToggleIcon = document.getElementById('cam-toggle-icon');
    const camToggleText = document.getElementById('cam-toggle-text');
    const cameraPlaceholder = document.getElementById('camera-placeholder');
    const liveStreamImg = document.getElementById('live-stream-img');
    const btnVerifyNow = document.getElementById('btn-verify-now');

    const valSimScore = document.getElementById('val-sim-score');
    const valThreshold = document.getElementById('val-threshold');
    const simProgressFill = document.getElementById('sim-progress-fill');

    // Decision Breakdown
    const boxDocFace = document.getElementById('box-doc-face');
    const resDocFace = document.getElementById('res-doc-face');
    const boxName = document.getElementById('box-name');
    const resNameMatch = document.getElementById('res-name-match');
    const boxFace = document.getElementById('box-face');
    const resFaceMatch = document.getElementById('res-face-match');
    const resSimScore = document.getElementById('res-sim-score');
    const boxFinal = document.getElementById('box-final');
    const resFinalVerdict = document.getElementById('res-final-verdict');
    const resultTimestamp = document.getElementById('result-timestamp');
    const errorBanner = document.getElementById('error-banner');
    const errorBannerText = document.getElementById('error-banner-text');

    // Steps
    const step1Badge = document.getElementById('step-1-badge');
    const step2Badge = document.getElementById('step-2-badge');
    const step3Badge = document.getElementById('step-3-badge');
    const step4Badge = document.getElementById('step-4-badge');

    // Fetch Health and Initial Cameras
    async function initSystem() {
        try {
            const healthRes = await fetch('/health');
            if (healthRes.ok) {
                const data = await healthRes.json();
                configuredThreshold = data.similarity_threshold || 0.65;
                valThreshold.innerText = configuredThreshold.toFixed(4);
            }
        } catch (e) {
            console.error("Health check error:", e);
        }

        try {
            const camRes = await fetch('/cameras');
            if (camRes.ok) {
                const data = await camRes.json();
                cameraSelect.innerHTML = '';
                if (data.cameras && data.cameras.length > 0) {
                    data.cameras.forEach(cam => {
                        const opt = document.createElement('option');
                        opt.value = cam.index;
                        opt.innerText = cam.name;
                        if (cam.index === data.selected_camera) opt.selected = true;
                        cameraSelect.appendChild(opt);
                    });
                } else {
                    cameraSelect.innerHTML = '<option value="0">Default Camera 0</option>';
                }
            }
        } catch (e) {
            console.error("Camera listing error:", e);
        }
    }

    initSystem();

    // Camera Selection change
    cameraSelect.addEventListener('change', async () => {
        const idx = cameraSelect.value;
        try {
            await fetch(`/cameras/select?index=${idx}`, { method: 'POST' });
            if (isCameraActive) {
                restartLiveStream();
            }
        } catch (e) {
            console.error("Error switching camera:", e);
        }
    });

    // Dropzone Events
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });
    fileInput.addEventListener('change', () => {
        if (fileInput.files && fileInput.files.length > 0) {
            handleFileSelection(fileInput.files[0]);
        }
    });

    function handleFileSelection(file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            placeholder.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            btnAnalyze.disabled = false;
            hideError();
            setStepActive(step1Badge);
        };
        reader.readAsDataURL(file);
    }

    btnClearDoc.addEventListener('click', (e) => {
        e.stopPropagation();
        resetDocumentState();
    });

    function resetDocumentState() {
        selectedFile = null;
        fileInput.value = '';
        previewImg.src = '';
        placeholder.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        btnAnalyze.disabled = true;
        docAnalysisBox.classList.add('hidden');
        activeSessionId = null;
        btnVerifyNow.disabled = true;
        expectedNameInput.value = '';
        resetVerdictDisplay();
        hideError();
    }

    // Step 2: Analyze Document
    btnAnalyze.addEventListener('click', async () => {
        if (!selectedFile) return;

        btnAnalyze.disabled = true;
        btnAnalyze.innerText = 'Analyzing...';
        hideError();

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const res = await fetch('/document/analyze', {
                method: 'POST',
                body: formData,
            });
            const data = await res.json();

            if (!data.success || !data.document) {
                const err = data.error || { message: 'Document analysis failed' };
                showError(err.code ? `[${err.code}] ${err.message}` : err.message);
                updateDocumentStatus(false, 0, null);
                btnAnalyze.disabled = false;
                btnAnalyze.innerText = '⚡ Analyze Document';
                return;
            }

            // Success
            const doc = data.document;
            activeSessionId = doc.session_id;
            updateDocumentStatus(doc.face_detected, doc.face_count, doc.name);
            
            if (doc.name) {
                expectedNameInput.value = doc.name;
            }

            btnVerifyNow.disabled = false;
            btnAnalyze.innerText = '✓ Analyzed';
            setStepActive(step2Badge);
            
            // If camera is already streaming, update stream to overlay reference comparison
            if (isCameraActive) {
                restartLiveStream();
            }
        } catch (e) {
            showError(`Network or server error: ${e.message}`);
            btnAnalyze.disabled = false;
            btnAnalyze.innerText = '⚡ Analyze Document';
        }
    });

    function updateDocumentStatus(faceDetected, faceCount, name) {
        docAnalysisBox.classList.remove('hidden');
        if (faceDetected && faceCount === 1) {
            badgeDocFace.innerText = 'PASS (1 FACE)';
            badgeDocFace.className = 'badge pass';
            resDocFace.innerText = 'PASS';
            resDocFace.className = 'decision-status pass';
        } else {
            badgeDocFace.innerText = faceCount > 1 ? `FAIL (${faceCount} FACES)` : 'FAIL (NO FACE)';
            badgeDocFace.className = 'badge fail';
            resDocFace.innerText = 'FAIL';
            resDocFace.className = 'decision-status fail';
        }

        docFaceCount.innerText = faceCount;
        docOcrName.innerText = name || '(No name extracted)';
    }

    // Camera Controls
    btnToggleCam.addEventListener('click', () => {
        if (!isCameraActive) {
            startCamera();
        } else {
            stopCamera();
        }
    });

    function startCamera() {
        isCameraActive = true;
        camToggleIcon.innerText = '⏹';
        camToggleText.innerText = 'Stop Camera';
        cameraPlaceholder.classList.add('hidden');
        liveStreamImg.classList.remove('hidden');
        restartLiveStream();
        setStepActive(step3Badge);
    }

    function stopCamera() {
        isCameraActive = false;
        camToggleIcon.innerText = '▶';
        camToggleText.innerText = 'Start Camera';
        liveStreamImg.src = '';
        liveStreamImg.classList.add('hidden');
        cameraPlaceholder.classList.remove('hidden');
    }

    function restartLiveStream() {
        const sidParam = activeSessionId ? `&session_id=${encodeURIComponent(activeSessionId)}` : '';
        const threshParam = `&threshold=${configuredThreshold}`;
        liveStreamImg.src = `/cameras/stream?t=${Date.now()}${sidParam}${threshParam}`;
    }

    // Step 4: Verify Match
    btnVerifyNow.addEventListener('click', async () => {
        if (!activeSessionId) {
            showError("Please analyze a document first.");
            return;
        }

        btnVerifyNow.disabled = true;
        btnVerifyNow.innerText = 'Verifying...';
        hideError();

        const formData = new FormData();
        formData.append('session_id', activeSessionId);
        if (expectedNameInput.value.trim()) {
            formData.append('expected_name', expectedNameInput.value.trim());
        }

        try {
            const res = await fetch('/verification/compare', {
                method: 'POST',
                body: formData,
            });
            const data = await res.json();

            btnVerifyNow.disabled = false;
            btnVerifyNow.innerText = '🎯 Verify Match';

            if (!data.success || !data.verification) {
                const err = data.error || { message: 'Verification failed' };
                showError(err.code ? `[${err.code}] ${err.message}` : err.message);
                return;
            }

            renderVerificationResult(data.verification);
            setStepActive(step4Badge);
        } catch (e) {
            btnVerifyNow.disabled = false;
            btnVerifyNow.innerText = '🎯 Verify Match';
            showError(`Verification error: ${e.message}`);
        }
    });

    function renderVerificationResult(v) {
        // Document Face
        resDocFace.innerText = v.document_face_detected && v.document_face_count === 1 ? 'PASS' : 'FAIL';
        resDocFace.className = `decision-status ${v.document_face_detected && v.document_face_count === 1 ? 'pass' : 'fail'}`;

        // Name Match
        resNameMatch.innerText = v.name_match ? 'PASS' : 'FAIL';
        resNameMatch.className = `decision-status ${v.name_match ? 'pass' : 'fail'}`;

        // Face Match
        resFaceMatch.innerText = v.face_match ? 'PASS' : 'FAIL';
        resFaceMatch.className = `decision-status ${v.face_match ? 'pass' : 'fail'}`;

        // Similarity Score
        const sim = v.similarity;
        const thresh = v.threshold;
        resSimScore.innerText = sim.toFixed(4);
        resSimScore.className = `decision-score ${v.face_match ? 'pass' : 'fail'}`;
        
        valSimScore.innerText = sim.toFixed(4);
        valThreshold.innerText = thresh.toFixed(4);

        // Progress bar percentage (0 to 1.0 clamped)
        const pct = Math.max(0, Math.min(100, Math.round(sim * 100)));
        simProgressFill.style.width = `${pct}%`;

        // Final Result
        if (v.verified) {
            resFinalVerdict.innerText = 'VERIFIED';
            boxFinal.className = 'decision-box final-box verified-badge';
            resFinalVerdict.className = 'decision-status final-status pass';
        } else {
            resFinalVerdict.innerText = 'NOT VERIFIED';
            boxFinal.className = 'decision-box final-box not-verified-badge';
            resFinalVerdict.className = 'decision-status final-status fail';
        }

        resultTimestamp.innerText = `Evaluated at ${new Date().toLocaleTimeString()}`;
    }

    function resetVerdictDisplay() {
        resDocFace.innerText = 'WAITING';
        resDocFace.className = 'decision-status';
        resNameMatch.innerText = 'WAITING';
        resNameMatch.className = 'decision-status';
        resFaceMatch.innerText = 'WAITING';
        resFaceMatch.className = 'decision-status';
        resSimScore.innerText = '----';
        resSimScore.className = 'decision-score';
        resFinalVerdict.innerText = 'NOT VERIFIED';
        boxFinal.className = 'decision-box final-box';
        resFinalVerdict.className = 'decision-status final-status';
        valSimScore.innerText = '0.0000';
        simProgressFill.style.width = '0%';
        resultTimestamp.innerText = 'Ready for analysis';
    }

    function setStepActive(stepElem) {
        [step1Badge, step2Badge, step3Badge, step4Badge].forEach(s => s.classList.remove('active'));
        stepElem.classList.add('active');
    }

    function showError(msg) {
        errorBannerText.innerText = msg;
        errorBanner.classList.remove('hidden');
    }

    function hideError() {
        errorBanner.classList.add('hidden');
    }
});
