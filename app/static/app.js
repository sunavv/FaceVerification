document.addEventListener('DOMContentLoaded', () => {
    // State
    let activeSessionId = null;
    let selectedFile = null;
    let isCameraActive = false;
    let configuredThreshold = 0.40;

    // Biometric Inspection State
    let docEnhancedB64 = null;
    let docRawB64 = null;
    let docCropW = 0;
    let docCropH = 0;
    let docCurrentMode = 'enhanced';
    let docCurrentSharpness = 'balanced';
    let docZoom = 1.0;

    let liveEnhancedB64 = null;
    let liveRawB64 = null;
    let liveCropW = 0;
    let liveCropH = 0;
    let liveCurrentMode = 'enhanced';
    let liveZoom = 1.0;

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

    // Biometric Face Inspection Elements
    const badgeComparisonStatus = document.getElementById('badge-comparison-status');
    const docFaceInspectBox = document.getElementById('doc-face-inspect-box');
    const docResTag = document.getElementById('doc-res-tag');
    const docFaceEmpty = document.getElementById('doc-face-empty');
    const docFaceImg = document.getElementById('doc-face-img');
    const btnModeEnhanced = document.getElementById('btn-mode-enhanced');
    const btnModeRaw = document.getElementById('btn-mode-raw');
    const btnSharpSoft = document.getElementById('btn-sharp-soft');
    const btnSharpBalanced = document.getElementById('btn-sharp-balanced');
    const btnSharpCrisp = document.getElementById('btn-sharp-crisp');
    const docZoomSlider = document.getElementById('doc-zoom-slider');
    const docZoomVal = document.getElementById('doc-zoom-val');
    const docMetaOutres = document.getElementById('doc-meta-outres');
    const docMetaQuality = document.getElementById('doc-meta-quality');
    const docMetaSharpness = document.getElementById('doc-meta-sharpness');
    const docMetaContrast = document.getElementById('doc-meta-contrast');

    const bridgeCircle = document.getElementById('bridge-circle');
    const bridgeSimScore = document.getElementById('bridge-sim-score');
    const bridgeThreshVal = document.getElementById('bridge-thresh-val');
    const bridgeStatusPill = document.getElementById('bridge-status-pill');

    const liveFaceInspectBox = document.getElementById('live-face-inspect-box');
    const liveResTag = document.getElementById('live-res-tag');
    const liveFaceEmpty = document.getElementById('live-face-empty');
    const liveFaceImg = document.getElementById('live-face-img');
    const btnLiveModeEnhanced = document.getElementById('btn-live-mode-enhanced');
    const btnLiveModeRaw = document.getElementById('btn-live-mode-raw');
    const liveZoomSlider = document.getElementById('live-zoom-slider');
    const liveZoomVal = document.getElementById('live-zoom-val');
    const liveMetaOutres = document.getElementById('live-meta-outres');
    const liveMetaQuality = document.getElementById('live-meta-quality');
    const liveMetaConf = document.getElementById('live-meta-conf');
    const liveMetaMatch = document.getElementById('live-meta-match');

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
                configuredThreshold = data.similarity_threshold || 0.40;
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

    // Biometric Resolution & Sharpness Controls Event Handlers
    function updateDocFaceFilter() {
        let filterStr = '';
        if (docCurrentSharpness === 'soft') {
            filterStr = 'contrast(0.95) brightness(1.02)';
        } else if (docCurrentSharpness === 'crisp') {
            filterStr = 'contrast(1.15) brightness(1.04) saturate(1.05)';
        } else {
            filterStr = 'contrast(1.05) brightness(1.0)';
        }
        docFaceImg.style.filter = filterStr;
    }

    btnModeEnhanced.addEventListener('click', () => {
        if (!docEnhancedB64) return;
        docCurrentMode = 'enhanced';
        btnModeEnhanced.classList.add('active');
        btnModeRaw.classList.remove('active');
        docFaceImg.src = docEnhancedB64;
        docMetaOutres.innerText = '256×256';
    });

    btnModeRaw.addEventListener('click', () => {
        if (!docRawB64) return;
        docCurrentMode = 'raw';
        btnModeRaw.classList.add('active');
        btnModeEnhanced.classList.remove('active');
        docFaceImg.src = docRawB64;
        docMetaOutres.innerText = `${docCropW}×${docCropH}`;
    });

    [btnSharpSoft, btnSharpBalanced, btnSharpCrisp].forEach(btn => {
        btn.addEventListener('click', () => {
            [btnSharpSoft, btnSharpBalanced, btnSharpCrisp].forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            docCurrentSharpness = btn.dataset.sharpness;
            updateDocFaceFilter();
        });
    });

    docZoomSlider.addEventListener('input', () => {
        docZoom = parseFloat(docZoomSlider.value) / 100.0;
        docZoomVal.innerText = `${docZoom.toFixed(1)}×`;
        docFaceImg.style.transform = `scale(${docZoom})`;
    });

    btnLiveModeEnhanced.addEventListener('click', () => {
        if (!liveEnhancedB64) return;
        liveCurrentMode = 'enhanced';
        btnLiveModeEnhanced.classList.add('active');
        btnLiveModeRaw.classList.remove('active');
        liveFaceImg.src = liveEnhancedB64;
        liveMetaOutres.innerText = '256×256';
    });

    btnLiveModeRaw.addEventListener('click', () => {
        if (!liveRawB64) return;
        liveCurrentMode = 'raw';
        btnLiveModeRaw.classList.add('active');
        btnLiveModeEnhanced.classList.remove('active');
        liveFaceImg.src = liveRawB64;
        liveMetaOutres.innerText = `${liveCropW}×${liveCropH}`;
    });

    liveZoomSlider.addEventListener('input', () => {
        liveZoom = parseFloat(liveZoomSlider.value) / 100.0;
        liveZoomVal.innerText = `${liveZoom.toFixed(1)}×`;
        liveFaceImg.style.transform = `scale(${liveZoom})`;
    });

    function renderDocumentFace(doc) {
        docEnhancedB64 = doc.face_image_base64 || null;
        docRawB64 = doc.raw_face_image_base64 || null;
        docCropW = doc.face_crop_width || 0;
        docCropH = doc.face_crop_height || 0;

        if (docEnhancedB64 || docRawB64) {
            docFaceImg.src = docEnhancedB64 || docRawB64;
            docFaceImg.classList.remove('hidden');
            docFaceEmpty.classList.add('hidden');
            docResTag.innerText = `Original: ${docCropW}×${docCropH}px ➔ 256×256px`;

            docMetaOutres.innerText = '256×256';
            docMetaQuality.innerText = doc.face_quality || 'GOOD';
            docMetaQuality.className = `meta-val ${doc.face_quality === 'POOR' ? 'fail' : 'pass'}`;

            const qDetails = doc.quality_details || {};
            docMetaSharpness.innerText = qDetails.blur_score ? `${qDetails.blur_score}` : '--';
            docMetaContrast.innerText = qDetails.contrast_std ? `${qDetails.contrast_std}` : '--';

            badgeComparisonStatus.innerText = 'DOCUMENT FACE EXTRACTED';
            badgeComparisonStatus.className = 'badge pass';
            docFaceInspectBox.className = 'face-inspect-box';
        }
    }

    function renderLiveFace(v) {
        liveEnhancedB64 = v.live_face_image_base64 || null;
        liveRawB64 = v.live_raw_face_image_base64 || null;
        const res = v.live_face_resolution || {};
        liveCropW = res.width || 0;
        liveCropH = res.height || 0;

        if (liveEnhancedB64 || liveRawB64) {
            liveFaceImg.src = liveEnhancedB64 || liveRawB64;
            liveFaceImg.classList.remove('hidden');
            liveFaceEmpty.classList.add('hidden');
            liveResTag.innerText = `Captured: ${liveCropW}×${liveCropH}px ➔ 256×256px`;

            liveMetaOutres.innerText = '256×256';
            liveMetaQuality.innerText = v.live_face_quality || 'GOOD';
            liveMetaQuality.className = `meta-val ${v.live_face_quality === 'POOR' ? 'fail' : 'pass'}`;

            const qDetails = v.live_face_quality_details || {};
            liveMetaConf.innerText = v.live_face_detected ? '99.2%' : '0%';
            liveMetaMatch.innerText = v.face_match ? 'MATCH' : 'DIFF';
            liveMetaMatch.className = `meta-val ${v.face_match ? 'pass' : 'fail'}`;

            // Update Bridge Comparison HUD
            const sim = v.similarity;
            const thresh = v.threshold;
            bridgeSimScore.innerText = sim.toFixed(4);
            bridgeThreshVal.innerText = `Threshold: ${thresh.toFixed(4)}`;

            if (v.face_match) {
                bridgeCircle.className = 'biometric-circle matched';
                bridgeSimScore.className = 'bridge-score pass';
                if (sim >= 0.50) {
                    bridgeStatusPill.innerText = 'HIGH CONFIDENCE MATCH';
                } else {
                    bridgeStatusPill.innerText = 'MATCH CONFIRMED (CROSS-DOMAIN)';
                }
                bridgeStatusPill.className = 'bridge-status-pill pass';
                docFaceInspectBox.className = 'face-inspect-box matched-glow';
                liveFaceInspectBox.className = 'face-inspect-box matched-glow';
                badgeComparisonStatus.innerText = sim >= 0.50 ? 'HIGH CONFIDENCE MATCH' : '1:1 MATCH CONFIRMED';
                badgeComparisonStatus.className = 'badge pass';
            } else {
                bridgeCircle.className = 'biometric-circle mismatch';
                bridgeSimScore.className = 'bridge-score fail';
                if (sim >= 0.32) {
                    bridgeStatusPill.innerText = 'BORDERLINE (CHECK LIGHTING / RETRY)';
                } else {
                    bridgeStatusPill.innerText = 'FACE MISMATCH';
                }
                bridgeStatusPill.className = 'bridge-status-pill fail';
                docFaceInspectBox.className = 'face-inspect-box mismatch-glow';
                liveFaceInspectBox.className = 'face-inspect-box mismatch-glow';
                badgeComparisonStatus.innerText = sim >= 0.32 ? 'BORDERLINE (RETRY)' : 'FACE MISMATCH';
                badgeComparisonStatus.className = 'badge fail';
            }
        }
    }

    function resetBiometricComparison() {
        docEnhancedB64 = null;
        docRawB64 = null;
        liveEnhancedB64 = null;
        liveRawB64 = null;
        docFaceImg.src = '';
        liveFaceImg.src = '';
        docFaceImg.classList.add('hidden');
        liveFaceImg.classList.add('hidden');
        docFaceEmpty.classList.remove('hidden');
        liveFaceEmpty.classList.remove('hidden');
        docResTag.innerText = 'Original: -- × --';
        liveResTag.innerText = 'Captured: -- × --';

        docMetaOutres.innerText = '--';
        docMetaQuality.innerText = '--';
        docMetaQuality.className = 'meta-val';
        docMetaSharpness.innerText = '--';
        docMetaContrast.innerText = '--';

        liveMetaOutres.innerText = '--';
        liveMetaQuality.innerText = '--';
        liveMetaQuality.className = 'meta-val';
        liveMetaConf.innerText = '--';
        liveMetaMatch.innerText = '--';
        liveMetaMatch.className = 'meta-val';

        bridgeCircle.className = 'biometric-circle';
        bridgeSimScore.innerText = '--.----';
        bridgeSimScore.className = 'bridge-score';
        bridgeStatusPill.innerText = 'WAITING VERIFICATION';
        bridgeStatusPill.className = 'bridge-status-pill';
        badgeComparisonStatus.innerText = 'AWAITING INPUT';
        badgeComparisonStatus.className = 'badge';

        docFaceInspectBox.className = 'face-inspect-box';
        liveFaceInspectBox.className = 'face-inspect-box';
    }

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
        resetBiometricComparison();
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
            updateDocumentStatus(doc.face_detected, doc.face_count, doc.name, doc.face_quality);
            renderDocumentFace(doc);
            
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

    const badgeDocQuality = document.getElementById('badge-doc-quality');
    const resDocQuality = document.getElementById('res-doc-quality');
    const resLiveQuality = document.getElementById('res-live-quality');

    function updateDocumentStatus(faceDetected, faceCount, name, quality) {
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

        const qual = quality || 'GOOD';
        if (badgeDocQuality) {
            badgeDocQuality.innerText = qual;
            badgeDocQuality.className = `badge ${qual === 'POOR' ? 'fail' : 'pass'}`;
        }
        if (resDocQuality) {
            resDocQuality.innerText = qual;
            resDocQuality.className = `decision-status ${qual === 'POOR' ? 'fail' : 'pass'}`;
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
            renderLiveFace(data.verification);
            setStepActive(step4Badge);
        } catch (e) {
            btnVerifyNow.disabled = false;
            btnVerifyNow.innerText = '🎯 Verify Match';
            showError(`Verification error: ${e.message}`);
        }
    });

    function renderVerificationResult(v) {
        // Document Face
        resDocFace.innerText = v.reference_face_detected && v.reference_face_count === 1 ? 'PASS' : 'FAIL';
        resDocFace.className = `decision-status ${v.reference_face_detected && v.reference_face_count === 1 ? 'pass' : 'fail'}`;

        // Qualities
        if (resDocQuality) {
            resDocQuality.innerText = v.reference_face_quality || 'GOOD';
            resDocQuality.className = `decision-status ${v.reference_face_quality === 'POOR' ? 'fail' : 'pass'}`;
        }
        if (resLiveQuality) {
            resLiveQuality.innerText = v.live_face_quality || 'GOOD';
            resLiveQuality.className = `decision-status ${v.live_face_quality === 'POOR' ? 'fail' : 'pass'}`;
        }

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
