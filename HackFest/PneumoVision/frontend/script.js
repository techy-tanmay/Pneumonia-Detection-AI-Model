/**
 * PneumoVision Client Application
 * Clean, Hackathon-Ready AI Chest X-ray Analysis Dashboard
 * Dual-Engine: Faster R-CNN (Localization) + DenseNet-121 (Classification)
 */

document.addEventListener("DOMContentLoaded", () => {
    // --- State Variables ---
    let selectedFile = null;
    let currentAnalysis = null;
    let zoomLevel = 1.0;
    let isInverted = false;
    let activeHighlightIndex = -1;

    // --- DOM Elements ---
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const browseBtn = document.getElementById("browse-btn");
    const fileSummary = document.getElementById("file-summary");
    const fileNameText = document.getElementById("file-name-text");
    const fileSizeText = document.getElementById("file-size-text");
    const fileBadgeFmt = document.getElementById("file-badge-fmt");
    const removeFileBtn = document.getElementById("remove-file-btn");

    const analyzeBtn = document.getElementById("analyze-btn");
    const progressContainer = document.getElementById("progress-container");
    const progressStatusTitle = document.getElementById("progress-status-title");

    const viewerPlaceholder = document.getElementById("viewer-placeholder");
    const imageViewport = document.getElementById("image-viewport");
    const transformWrapper = document.getElementById("transform-wrapper");
    const xrayImage = document.getElementById("xray-image");
    const overlayCanvas = document.getElementById("overlay-canvas");

    const findingsContainer = document.getElementById("findings-container");
    const findingBanner = document.getElementById("finding-banner");
    const topConfidenceBadge = document.getElementById("top-confidence-badge");
    const findingText = document.getElementById("finding-text");
    const findingDesc = document.getElementById("finding-desc");

    const metricConfidence = document.getElementById("metric-confidence");
    const confidenceMeterFill = document.getElementById("confidence-meter-fill");
    const metricRegions = document.getElementById("metric-regions");
    const metricLocalization = document.getElementById("metric-localization");
    const metricLatency = document.getElementById("metric-latency");

    const secondaryClfCard = document.getElementById("secondary-clf-card");
    const clfModelName = document.getElementById("clf-model-name");
    const clfProbVal = document.getElementById("clf-prob-val");
    const clfProbBar = document.getElementById("clf-prob-bar");
    const clfStatusBadge = document.getElementById("clf-status-badge");
    const clfLatencyText = document.getElementById("clf-latency-text");

    const regionsTableBody = document.getElementById("regions-table-body");
    const regionsCountPill = document.getElementById("regions-count-pill");

    const zoomInBtn = document.getElementById("zoom-in-btn");
    const zoomOutBtn = document.getElementById("zoom-out-btn");
    const zoomResetBtn = document.getElementById("zoom-reset-btn");
    const invertViewBtn = document.getElementById("invert-view-btn");
    const fullscreenBtn = document.getElementById("fullscreen-btn");

    const downloadHtmlReportBtn = document.getElementById("download-html-report-btn");
    const downloadTextReportBtn = document.getElementById("download-text-report-btn");

    const historyBtn = document.getElementById("open-history-btn");
    const historyModal = document.getElementById("history-modal");
    const closeHistoryBtn = document.getElementById("close-history-btn");
    const dismissHistoryBtn = document.getElementById("dismiss-history-btn");
    const clearHistoryBtn = document.getElementById("clear-history-btn");
    const historyList = document.getElementById("history-list");
    const historyCounter = document.getElementById("history-counter");

    const fullscreenModal = document.getElementById("fullscreen-modal");
    const closeFullscreenBtn = document.getElementById("close-fullscreen-btn");
    const fsXrayImage = document.getElementById("fs-xray-image");
    const fsOverlayCanvas = document.getElementById("fs-overlay-canvas");

    const statusDot = document.getElementById("status-dot");
    const statusText = document.getElementById("status-text");
    const devicePill = document.getElementById("device-pill");

    // ==========================================================================
    // 1. Initial Health Check
    // ==========================================================================
    async function checkBackendHealth() {
        try {
            const res = await fetch("/health");
            if (res.ok) {
                const data = await res.json();
                statusDot.className = "status-indicator-dot online";
                statusText.textContent = "AI Pipeline Ready (Faster R-CNN + DenseNet-121)";
                devicePill.textContent = (data.device || "cpu").toUpperCase();
            } else {
                setBackendOffline();
            }
        } catch (e) {
            setBackendOffline();
        }
    }

    function setBackendOffline() {
        statusDot.className = "status-indicator-dot offline";
        statusText.textContent = "Backend Offline";
        devicePill.textContent = "OFFLINE";
    }

    checkBackendHealth();
    updateHistoryCounter();

    // ==========================================================================
    // 2. File Selection & Drag & Drop
    // ==========================================================================
    browseBtn.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("click", (e) => {
        if (e.target !== browseBtn) fileInput.click();
    });

    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) handleSelectedFile(files[0]);
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    function handleSelectedFile(file) {
        selectedFile = file;
        const name = file.name;
        const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
        const ext = name.split(".").pop().toUpperCase();

        fileNameText.textContent = name;
        fileSizeText.textContent = `${sizeMb} MB`;
        fileBadgeFmt.textContent = ext.substring(0, 4);

        fileSummary.classList.remove("hidden");
        dropzone.classList.add("hidden");
        analyzeBtn.disabled = false;
    }

    removeFileBtn.addEventListener("click", () => {
        selectedFile = null;
        fileInput.value = "";
        fileSummary.classList.add("hidden");
        dropzone.classList.remove("hidden");
        analyzeBtn.disabled = true;
    });

    // Sample Radiograph Quick Loader
    document.querySelectorAll(".sample-chip").forEach(chip => {
        chip.addEventListener("click", async () => {
            const sampleName = chip.getAttribute("data-sample");
            chip.style.opacity = "0.6";
            try {
                const res = await fetch(`/sample_images/${sampleName}`);
                if (!res.ok) throw new Error("Sample file not found");
                const blob = await res.blob();
                const file = new File([blob], sampleName, { type: blob.type || "application/octet-stream" });
                handleSelectedFile(file);
            } catch (err) {
                alert(`Error loading sample radiograph: ${err.message}`);
            } finally {
                chip.style.opacity = "1";
            }
        });
    });

    // ==========================================================================
    // 3. Automated Dual-Engine Analysis
    // ==========================================================================
    analyzeBtn.addEventListener("click", runAnalysis);

    async function runAnalysis() {
        if (!selectedFile) return;

        analyzeBtn.disabled = true;
        progressContainer.classList.remove("hidden");
        resetStepper();

        // 1. Uploaded Step
        setStepActive("step-upload");
        progressStatusTitle.textContent = "Uploading radiograph...";

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            // 2. Preprocessing Step
            setStepActive("step-preprocess");
            progressStatusTitle.textContent = "Preprocessing radiograph (1st-99th percentile normalization)...";

            // 3. Detection Step
            setStepActive("step-detect");
            progressStatusTitle.textContent = "Running Faster R-CNN & DenseNet-121 dual-engine analysis...";

            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({ detail: "Network error" }));
                throw new Error(errData.detail || "Inference failed on server.");
            }

            const data = await response.json();

            // 4. Localization Step
            setStepActive("step-localize");
            progressStatusTitle.textContent = "Synthesizing bounding coordinates & radiological zones...";

            // 5. Complete Step
            setStepActive("step-complete");
            progressStatusTitle.textContent = "Analysis complete!";

            currentAnalysis = data;
            renderAnalysisResults(data);
            saveToHistory(data);

        } catch (err) {
            alert(`Analysis Error: ${err.message}`);
            progressStatusTitle.textContent = "Analysis halted due to an error.";
        } finally {
            analyzeBtn.disabled = false;
            setTimeout(() => {
                progressContainer.classList.add("hidden");
            }, 1200);
        }
    }

    function resetStepper() {
        ["step-upload", "step-preprocess", "step-detect", "step-localize", "step-complete"].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.className = "step-item";
        });
    }

    function setStepActive(stepId) {
        const steps = ["step-upload", "step-preprocess", "step-detect", "step-localize", "step-complete"];
        const curIdx = steps.indexOf(stepId);
        steps.forEach((id, idx) => {
            const el = document.getElementById(id);
            if (!el) return;
            if (idx < curIdx) {
                el.className = "step-item complete";
            } else if (idx === curIdx) {
                el.className = "step-item active";
            } else {
                el.className = "step-item";
            }
        });
    }

    // ==========================================================================
    // 4. Render Results & Interactive Bounding Boxes
    // ==========================================================================
    function renderAnalysisResults(data) {
        viewerPlaceholder.classList.add("hidden");
        imageViewport.classList.remove("hidden");
        findingsContainer.classList.remove("hidden");

        const isPositive = data.is_positive !== undefined ? data.is_positive : (data.num_detections > 0);

        // Model Finding Card
        findingBanner.className = isPositive ? "finding-banner positive" : "finding-banner negative";
        topConfidenceBadge.textContent = isPositive ? `${data.confidence_percent} Confidence` : "Negative (Normal)";
        topConfidenceBadge.className = isPositive ? "confidence-badge" : "confidence-badge neg";
        findingText.textContent = data.finding;
        findingDesc.textContent = data.finding_description;

        // Key Metrics
        metricConfidence.textContent = isPositive ? data.confidence_percent : "0.0%";
        confidenceMeterFill.style.width = isPositive ? `${Math.min(100, (data.confidence || 0) * 100)}%` : "0%";
        metricRegions.textContent = `${data.num_detections} Region${data.num_detections === 1 ? '' : 's'}`;
        metricLocalization.textContent = data.localization || "None detected";
        metricLatency.textContent = `${data.inference_time_ms} ms`;

        // Classification Card
        if (data.classification && data.classification.status === "success") {
            secondaryClfCard.classList.remove("hidden");
            clfModelName.textContent = data.classification.model_name || "DenseNet-121";
            clfProbVal.textContent = data.classification.probability_percent;
            clfProbBar.style.width = data.classification.probability_percent;

            const isClfPos = data.classification.is_positive;
            if (clfStatusBadge) {
                clfStatusBadge.textContent = isClfPos ? "Pneumonia Confirmed (≥85%)" : "Within Normal Limits (<85%)";
                clfStatusBadge.className = isClfPos ? "clf-verdict-pill pos" : "clf-verdict-pill neg";
            }
            if (clfLatencyText && data.classification.latency_ms) {
                clfLatencyText.textContent = `Latency: ${data.classification.latency_ms} ms`;
            }
        }

        // Multiple Regions Table
        regionsCountPill.textContent = `${data.num_detections} Box${data.num_detections === 1 ? '' : 'es'}`;
        renderRegionsList(data.detections || []);

        // Load Radiograph Image
        xrayImage.src = data.display_image;
        fsXrayImage.src = data.display_image;

        xrayImage.onload = () => {
            resetZoom();
            drawBoundingBoxes(overlayCanvas, xrayImage, data.detections || []);
            drawBoundingBoxes(fsOverlayCanvas, fsXrayImage, data.detections || []);
        };
    }

    function renderRegionsList(detections) {
        regionsTableBody.innerHTML = "";
        if (!detections || detections.length === 0) {
            regionsTableBody.innerHTML = `
                <div style="padding: 14px; text-align: center; color: var(--text-muted); font-size: 0.85rem; font-style: italic;">
                    No localized opacity regions exceeded the 0.75 confidence threshold.
                </div>
            `;
            return;
        }

        detections.forEach((det, idx) => {
            const row = document.createElement("div");
            row.className = "region-item-row";
            row.setAttribute("data-index", idx);

            const b = det.box;
            row.innerHTML = `
                <div class="region-badge-name">#${idx + 1} Opacity</div>
                <div class="region-zone">${det.anatomical_zone}</div>
                <div class="region-conf-pill">${det.confidence_percent}</div>
                <div class="region-coords">[${b.x}, ${b.y}, ${b.width}, ${b.height}]</div>
            `;

            row.addEventListener("mouseenter", () => {
                activeHighlightIndex = idx;
                drawBoundingBoxes(overlayCanvas, xrayImage, currentAnalysis.detections || [], activeHighlightIndex);
                row.classList.add("active");
            });

            row.addEventListener("mouseleave", () => {
                activeHighlightIndex = -1;
                drawBoundingBoxes(overlayCanvas, xrayImage, currentAnalysis.detections || [], -1);
                row.classList.remove("active");
            });

            regionsTableBody.appendChild(row);
        });
    }

    function drawBoundingBoxes(canvas, imgElement, detections, highlightIdx = -1) {
        if (!canvas || !imgElement) return;

        const displayedWidth = imgElement.clientWidth;
        const displayedHeight = imgElement.clientHeight;

        canvas.width = displayedWidth;
        canvas.height = displayedHeight;

        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!currentAnalysis || !currentAnalysis.image) return;

        const origW = currentAnalysis.image.width || displayedWidth;
        const origH = currentAnalysis.image.height || displayedHeight;

        const scaleX = displayedWidth / origW;
        const scaleY = displayedHeight / origH;

        detections.forEach((det, idx) => {
            const b = det.box;
            const x = b.x * scaleX;
            const y = b.y * scaleY;
            const w = b.width * scaleX;
            const h = b.height * scaleY;

            const isHighlighted = (idx === highlightIdx);

            // Bounding box fill
            ctx.fillStyle = isHighlighted ? "rgba(244, 63, 94, 0.28)" : "rgba(13, 148, 136, 0.22)";
            ctx.fillRect(x, y, w, h);

            // Bounding box stroke
            ctx.strokeStyle = isHighlighted ? "#f43f5e" : "#0d9488";
            ctx.lineWidth = isHighlighted ? 3 : 2;
            ctx.setLineDash(isHighlighted ? [4, 2] : []);
            ctx.strokeRect(x, y, w, h);
            ctx.setLineDash([]);

            // Label tag on top of box
            const labelText = `Opacity #${idx + 1} (${det.confidence_percent})`;
            ctx.font = "bold 11px 'Plus Jakarta Sans', sans-serif";
            const textWidth = ctx.measureText(labelText).width;
            const tagHeight = 18;
            const tagWidth = textWidth + 14;

            const tagY = Math.max(0, y - tagHeight);
            ctx.fillStyle = isHighlighted ? "#f43f5e" : "#0d9488";
            ctx.fillRect(x, tagY, tagWidth, tagHeight);

            ctx.fillStyle = "#ffffff";
            ctx.fillText(labelText, x + 7, tagY + 13);
        });
    }

    window.addEventListener("resize", () => {
        if (currentAnalysis) {
            drawBoundingBoxes(overlayCanvas, xrayImage, currentAnalysis.detections || []);
            drawBoundingBoxes(fsOverlayCanvas, fsXrayImage, currentAnalysis.detections || []);
        }
    });

    // ==========================================================================
    // 5. Image Controls (Zoom, Pan, Invert, Fullscreen)
    // ==========================================================================
    zoomInBtn.addEventListener("click", () => setZoom(zoomLevel + 0.25));
    zoomOutBtn.addEventListener("click", () => setZoom(zoomLevel - 0.25));
    zoomResetBtn.addEventListener("click", resetZoom);

    invertViewBtn.addEventListener("click", () => {
        isInverted = !isInverted;
        xrayImage.classList.toggle("inverted", isInverted);
        fsXrayImage.classList.toggle("inverted", isInverted);
        invertViewBtn.textContent = isInverted ? "Normal" : "Invert";
    });

    fullscreenBtn.addEventListener("click", () => {
        if (!currentAnalysis) return;
        fullscreenModal.classList.remove("hidden");
        setTimeout(() => {
            drawBoundingBoxes(fsOverlayCanvas, fsXrayImage, currentAnalysis.detections || []);
        }, 100);
    });

    closeFullscreenBtn.addEventListener("click", () => {
        fullscreenModal.classList.add("hidden");
    });

    function setZoom(lvl) {
        zoomLevel = Math.max(0.5, Math.min(3.0, lvl));
        transformWrapper.style.transform = `scale(${zoomLevel})`;
    }

    function resetZoom() {
        zoomLevel = 1.0;
        transformWrapper.style.transform = "scale(1)";
    }

    // ==========================================================================
    // 6. Report Downloads
    // ==========================================================================
    downloadHtmlReportBtn.addEventListener("click", () => requestReportDownload("html"));
    downloadTextReportBtn.addEventListener("click", () => requestReportDownload("text"));

    async function requestReportDownload(format) {
        if (!currentAnalysis) return;

        const payload = {
            ...currentAnalysis,
            format: format
        };

        try {
            const res = await fetch("/report/download", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!res.ok) throw new Error("Failed to generate report.");

            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `PneumoVision_Report_${currentAnalysis.filename}.${format === 'text' ? 'txt' : 'html'}`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
        } catch (e) {
            alert(`Report download error: ${e.message}`);
        }
    }

    // ==========================================================================
    // 7. Local History Management (Browser localStorage)
    // ==========================================================================
    const HISTORY_KEY = "pneumovision_analysis_history";

    function getHistory() {
        try {
            return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
        } catch (e) {
            return [];
        }
    }

    function saveToHistory(record) {
        const history = getHistory();
        const isPos = record.is_positive !== undefined ? record.is_positive : (record.num_detections > 0);
        const item = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            filename: record.filename,
            finding: record.finding,
            is_positive: isPos,
            confidence_percent: record.confidence_percent,
            num_detections: record.num_detections,
            localization: record.localization,
            latency: record.inference_time_ms
        };
        history.unshift(item);
        if (history.length > 30) history.pop();
        localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
        updateHistoryCounter();
    }

    function updateHistoryCounter() {
        const count = getHistory().length;
        if (historyCounter) historyCounter.textContent = count;
    }

    historyBtn.addEventListener("click", openHistoryModal);
    closeHistoryBtn.addEventListener("click", () => historyModal.classList.add("hidden"));
    dismissHistoryBtn.addEventListener("click", () => historyModal.classList.add("hidden"));

    clearHistoryBtn.addEventListener("click", () => {
        if (confirm("Clear local analysis history?")) {
            localStorage.removeItem(HISTORY_KEY);
            openHistoryModal();
            updateHistoryCounter();
        }
    });

    function openHistoryModal() {
        const history = getHistory();
        historyList.innerHTML = "";

        if (history.length === 0) {
            historyList.innerHTML = `<div class="empty-history" style="text-align: center; padding: 20px; color: var(--text-muted);">No recorded analyses in browser storage.</div>`;
        } else {
            history.forEach(item => {
                const isPos = item.is_positive;
                const dateStr = new Date(item.timestamp).toLocaleString();
                const div = document.createElement("div");
                div.className = "history-item";
                div.innerHTML = `
                    <div class="history-item-left">
                        <span class="history-filename">${item.filename}</span>
                        <span class="history-meta">${dateStr} • ${item.latency} ms • ${item.localization}</span>
                    </div>
                    <div class="history-item-right">
                        <span class="history-badge ${isPos ? 'pos' : 'neg'}">${isPos ? item.confidence_percent : 'Negative'}</span>
                        <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 4px;">${item.num_detections} Region(s)</div>
                    </div>
                `;
                historyList.appendChild(div);
            });
        }
        historyModal.classList.remove("hidden");
    }
});
