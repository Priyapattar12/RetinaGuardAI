// Retina Guard AI - frontend logic (plain JS, no frameworks)

const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const dropZone = document.getElementById("dropZone");
const uploadPrompt = document.getElementById("uploadPrompt");
const previewImg = document.getElementById("previewImg");
const analyzeBtn = document.getElementById("analyzeBtn");
const analyzeBtnText = document.getElementById("analyzeBtnText");
const analyzeSpinner = document.getElementById("analyzeSpinner");
const resetBtn = document.getElementById("resetBtn");
const errorBox = document.getElementById("errorBox");
const sampleGallery = document.getElementById("sampleGallery");

const resultPlaceholder = document.getElementById("resultPlaceholder");
const resultContent = document.getElementById("resultContent");
const heatmapImg = document.getElementById("heatmapImg");
const gradeBadge = document.getElementById("gradeBadge");
const confidenceText = document.getElementById("confidenceText");
const confidenceBar = document.getElementById("confidenceBar");
const lesionCount = document.getElementById("lesionCount");
const lesionDensity = document.getElementById("lesionDensity");
const analysisMode = document.getElementById("analysisMode");
const adviceBox = document.getElementById("adviceBox");

let selectedFile = null;      // File object, when the user uploads their own image
let selectedSampleUrl = null; // URL string, when the user picks a sample

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove("d-none");
}

function clearError() {
  errorBox.classList.add("d-none");
  errorBox.textContent = "";
}

function resetAll() {
  selectedFile = null;
  selectedSampleUrl = null;
  fileInput.value = "";
  previewImg.src = "";
  previewImg.classList.add("d-none");
  uploadPrompt.classList.remove("d-none");
  analyzeBtn.disabled = true;
  clearError();
  resultContent.classList.add("d-none");
  resultPlaceholder.classList.remove("d-none");
  document.querySelectorAll(".sample-thumb").forEach(el => el.classList.remove("selected"));
}

function showPreview(src) {
  previewImg.src = src;
  previewImg.classList.remove("d-none");
  uploadPrompt.classList.add("d-none");
  analyzeBtn.disabled = false;
  clearError();
}

// --- File selection (browse / drag-drop) ---

browseBtn.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("click", (e) => {
  if (e.target === dropZone || e.target === uploadPrompt) fileInput.click();
});

fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files[0]) {
    selectedFile = fileInput.files[0];
    selectedSampleUrl = null;
    const reader = new FileReader();
    reader.onload = (e) => showPreview(e.target.result);
    reader.readAsDataURL(selectedFile);
  }
});

["dragenter", "dragover"].forEach(evt => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach(evt => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
  });
});

dropZone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) {
    selectedFile = file;
    selectedSampleUrl = null;
    const reader = new FileReader();
    reader.onload = (ev) => showPreview(ev.target.result);
    reader.readAsDataURL(file);
  }
});

// --- Sample gallery ---

async function loadSamples() {
  try {
    const res = await fetch("/api/samples");
    const data = await res.json();
    sampleGallery.innerHTML = "";
    data.samples.forEach(url => {
      const col = document.createElement("div");
      col.className = "col-4 col-md-2 col-lg-4";
      const img = document.createElement("img");
      img.src = url;
      img.className = "sample-thumb";
      img.alt = "Sample fundus image";
      img.addEventListener("click", () => {
        selectedFile = null;
        selectedSampleUrl = url;
        showPreview(url);
        document.querySelectorAll(".sample-thumb").forEach(el => el.classList.remove("selected"));
        img.classList.add("selected");
      });
      col.appendChild(img);
      sampleGallery.appendChild(col);
    });
  } catch (err) {
    console.error("Could not load sample images", err);
  }
}

// --- Analyze ---

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile && !selectedSampleUrl) return;

  clearError();
  analyzeBtn.disabled = true;
  analyzeBtnText.textContent = "Analyzing...";
  analyzeSpinner.classList.remove("d-none");

  try {
    const formData = await buildFormData();
    const res = await fetch("/api/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || "Something went wrong while analyzing the image.");
      return;
    }

    renderResult(data);
  } catch (err) {
    showError("Could not reach the server. Is the backend running?");
    console.error(err);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtnText.textContent = "Analyze Image";
    analyzeSpinner.classList.add("d-none");
  }
});

// Builds the multipart form body for whichever image is currently
// selected (an uploaded File, or a sample fetched as a blob), so both
// /api/predict and /api/report send the exact same image.
async function buildFormData() {
  const formData = new FormData();
  if (selectedFile) {
    formData.append("file", selectedFile);
  } else {
    const resp = await fetch(selectedSampleUrl);
    const blob = await resp.blob();
    const name = selectedSampleUrl.split("/").pop();
    formData.append("file", blob, name);
  }
  return formData;
}

function renderResult(data) {
  resultPlaceholder.classList.add("d-none");
  resultContent.classList.remove("d-none");

  heatmapImg.src = data.heatmap_image_base64;

  gradeBadge.textContent = data.grade;
  gradeBadge.style.backgroundColor = data.color;

  confidenceText.textContent = `${data.confidence}%`;
  confidenceBar.style.width = `${data.confidence}%`;
  confidenceBar.style.backgroundColor = data.color;

  lesionCount.textContent = data.lesion_count !== null ? data.lesion_count : "N/A";
  lesionDensity.textContent = data.lesion_density_pct !== null ? `${data.lesion_density_pct}%` : "N/A";
  analysisMode.textContent = data.mode === "trained-model"
    ? "Trained EfficientNetB0 model"
    : "Heuristic demo (OpenCV)";

  adviceBox.textContent = data.advice;

  if (!data.fundus_like) {
    showError("This doesn't look like a typical fundus (retina) photo - results may be unreliable.");
  }
}

resetBtn.addEventListener("click", resetAll);

const downloadReportBtn = document.getElementById("downloadReportBtn");
const reportSpinner = document.getElementById("reportSpinner");

downloadReportBtn.addEventListener("click", async () => {
  if (!selectedFile && !selectedSampleUrl) return;

  downloadReportBtn.disabled = true;
  reportSpinner.classList.remove("d-none");

  try {
    const formData = await buildFormData();
    const res = await fetch("/api/report", { method: "POST", body: formData });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      showError(data.error || "Could not generate the PDF report.");
      return;
    }

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "retina_guard_report.pdf";
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  } catch (err) {
    showError("Could not reach the server while generating the report.");
    console.error(err);
  } finally {
    downloadReportBtn.disabled = false;
    reportSpinner.classList.add("d-none");
  }
});

loadSamples();
