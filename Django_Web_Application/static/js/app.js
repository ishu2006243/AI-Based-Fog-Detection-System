/* ==========================================================================
   Interactive Frontend Logic for Glass Fog Detection WebApp
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");
  
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  
  const cameraVideo = document.getElementById("cameraVideo");
  const cameraPlaceholder = document.getElementById("cameraPlaceholder");
  const startCameraBtn = document.getElementById("startCameraBtn");
  const stopCameraBtn = document.getElementById("stopCameraBtn");
  const captureBtn = document.getElementById("captureBtn");
  const cameraCanvas = document.getElementById("cameraCanvas");
  
  const loaderOverlay = document.getElementById("loaderOverlay");
  const resultsCard = document.getElementById("resultsCard");
  const resultsPlaceholder = document.getElementById("resultsPlaceholder");
  
  const statusBanner = document.getElementById("statusBanner");
  const statusBadge = document.getElementById("statusBadge");
  const statusDesc = document.getElementById("statusDesc");
  const statusIcon = document.getElementById("statusIcon");
  
  const previewImg = document.getElementById("previewImg");
  const viewAnnotatedBtn = document.getElementById("viewAnnotatedBtn");
  const viewOriginalBtn = document.getElementById("viewOriginalBtn");
  
  const reasonsList = document.getElementById("reasonsList");
  
  // Metric values
  const valBgSharp = document.getElementById("valBgSharp");
  const valResRatio = document.getElementById("valResRatio");
  const valDynRange = document.getElementById("valDynRange");
  const valDroplets = document.getElementById("valDroplets");
  const valMeniscus = document.getElementById("valMeniscus");
  const valHaze = document.getElementById("valHaze");

  // State
  let cameraStream = null;
  let currentResult = null;

  // --------------------------------------------------------------------------
  // Tab Switching
  // --------------------------------------------------------------------------
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));
      
      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      document.getElementById(targetId).classList.add("active");

      // Auto start/stop camera on tab switch
      if (targetId === "cameraTab") {
        initCamera();
      } else {
        stopCamera();
      }
    });
  });

  // --------------------------------------------------------------------------
  // File Upload Handlers
  // --------------------------------------------------------------------------
  dropzone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      uploadFile(e.target.files[0]);
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });

  function uploadFile(file) {
    showLoader("Analyzing uploaded image...");
    const formData = new FormData();
    formData.append("image", file);

    fetch("/api/detect/", {
      method: "POST",
      body: formData,
    })
      .then(res => res.json())
      .then(data => handleAnalysisResult(data))
      .catch(err => {
        hideLoader();
        alert("Error analyzing image: " + err.message);
      });
  }

  // --------------------------------------------------------------------------
  // Camera Functions
  // --------------------------------------------------------------------------
  async function initCamera() {
    if (cameraStream) return;
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user"
        },
        audio: false,
      });
      cameraVideo.srcObject = cameraStream;
      cameraVideo.style.display = "block";
      cameraPlaceholder.style.display = "none";
      captureBtn.style.display = "inline-flex";
      stopCameraBtn.style.display = "inline-flex";
      startCameraBtn.style.display = "none";
    } catch (err) {
      console.warn("Camera access failed:", err);
      cameraPlaceholder.innerHTML = `
        <div style="color: #f87171; font-weight:600;">Camera Access Denied or Unavailable</div>
        <p style="font-size:0.85rem; color:#94a3b8;">Please allow camera permission in your browser or use the Upload tab.</p>
      `;
    }
  }

  function stopCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      cameraStream = null;
      cameraVideo.srcObject = null;
      cameraVideo.style.display = "none";
      cameraPlaceholder.style.display = "flex";
      captureBtn.style.display = "none";
      stopCameraBtn.style.display = "none";
      startCameraBtn.style.display = "inline-flex";
    }
  }

  startCameraBtn.addEventListener("click", initCamera);
  stopCameraBtn.addEventListener("click", stopCamera);

  captureBtn.addEventListener("click", () => {
    if (!cameraStream) return;
    
    // Draw current frame to hidden canvas
    const width = cameraVideo.videoWidth || 640;
    const height = cameraVideo.videoHeight || 480;
    cameraCanvas.width = width;
    cameraCanvas.height = height;
    
    const ctx = cameraCanvas.getContext("2d");
    // Un-mirror image if captured from mirrored preview
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(cameraVideo, 0, 0, width, height);

    const base64Data = cameraCanvas.toDataURL("image/jpeg", 0.9);

    showLoader("Capturing and analyzing frame...");
    fetch("/api/detect/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_data: base64Data }),
    })
      .then(res => res.json())
      .then(data => handleAnalysisResult(data))
      .catch(err => {
        hideLoader();
        alert("Camera processing failed: " + err.message);
      });
  });

  // --------------------------------------------------------------------------
  // Sample Chips Click
  // --------------------------------------------------------------------------
  document.querySelectorAll(".sample-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const sampleName = chip.getAttribute("data-sample");
      showLoader(`Evaluating sample '${sampleName}'...`);
      fetch("/api/detect/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_name: sampleName }),
      })
        .then(res => res.json())
        .then(data => handleAnalysisResult(data))
        .catch(err => {
          hideLoader();
          alert("Sample evaluation failed: " + err.message);
        });
    });
  });

  // --------------------------------------------------------------------------
  // Handle & Display Results
  // --------------------------------------------------------------------------
  function handleAnalysisResult(data) {
    hideLoader();
    if (!data.success) {
      alert("Error: " + (data.error || "Analysis failed"));
      return;
    }

    currentResult = data;
    resultsPlaceholder.style.display = "none";
    resultsCard.classList.add("active");

    const isFogged = data.is_fogged;

    // Status Banner
    statusBanner.className = `status-banner ${isFogged ? "fogged" : "clear"}`;
    statusBadge.textContent = isFogged ? "FOGGED" : "CLEAR";
    statusDesc.textContent = isFogged
      ? "Water condensation, dew beading, or steamy fog detected on the glass."
      : "The glass surface is clean. The scene behind the glass is sharp and unobstructed.";

    statusIcon.innerHTML = isFogged
      ? `<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/><path d="M7 22h10"/><path d="M9 16h6"/></svg>`
      : `<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>`;

    // Serial Status Notification
    const statusSerial = document.getElementById("statusSerial");
    if (statusSerial) {
      if (data.serial) {
        statusSerial.style.display = "block";
        if (data.serial.sent) {
          statusSerial.innerHTML = `📡 Serial: Sent "*1" to hardware`;
          statusSerial.style.color = "#86efac";
        } else {
          statusSerial.innerHTML = `⚠️ Serial: ${data.serial.message}`;
          statusSerial.style.color = "#fca5a5";
        }
      } else {
        statusSerial.style.display = "none";
      }
    }

    // Set Default Image (Annotated)
    previewImg.src = data.annotated_preview;
    viewAnnotatedBtn.classList.add("active");
    viewOriginalBtn.classList.remove("active");

    // Reasons List
    reasonsList.innerHTML = "";
    if (data.reasons && data.reasons.length > 0) {
      data.reasons.forEach(r => {
        const li = document.createElement("li");
        li.textContent = r;
        reasonsList.appendChild(li);
      });
    } else {
      const li = document.createElement("li");
      li.textContent = "Surface is clear (scene behind glass is sharp and unobstructed)";
      reasonsList.appendChild(li);
    }

    // Metrics (if present in DOM)
    const m = data.metrics;
    if (m) {
      if (valBgSharp) valBgSharp.textContent = m.bg_sharpness;
      if (valResRatio) valResRatio.textContent = m.res_ratio;
      if (valDynRange) valDynRange.textContent = m.dyn_range;
      if (valDroplets) valDroplets.textContent = `${m.blob_count} (${m.blob_density}/10k)`;
      if (valMeniscus) valMeniscus.textContent = `${m.meniscus_ratio}%`;
      if (valHaze) valHaze.textContent = m.haze_index;
    }

    // Smooth scroll to results on mobile
    if (window.innerWidth <= 960) {
      resultsCard.scrollIntoView({ behavior: "smooth" });
    }
  }

  // Toggle View
  viewAnnotatedBtn.addEventListener("click", () => {
    if (currentResult) {
      previewImg.src = currentResult.annotated_preview;
      viewAnnotatedBtn.classList.add("active");
      viewOriginalBtn.classList.remove("active");
    }
  });

  viewOriginalBtn.addEventListener("click", () => {
    if (currentResult) {
      previewImg.src = currentResult.original_preview;
      viewOriginalBtn.classList.add("active");
      viewAnnotatedBtn.classList.remove("active");
    }
  });

  function showLoader(text = "Processing...") {
    document.getElementById("loaderText").textContent = text;
    loaderOverlay.classList.add("active");
  }

  function hideLoader() {
    loaderOverlay.classList.remove("active");
  }
});
