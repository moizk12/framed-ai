import { AnalysisError, requestAnalysis } from "./analysis-client.js";
import { renderResult } from "./result-renderer.js";
import { setupFeedback } from "./feedback.js";

const MAX_BYTES = 16 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const app = document.querySelector("[data-analysis-app]");

if (app) {
  const form = app.querySelector("#analysis-form");
  const input = app.querySelector("#image-input");
  const dropZone = app.querySelector("[data-drop-zone]");
  const empty = app.querySelector("[data-drop-empty]");
  const selectedView = app.querySelector("[data-drop-selected]");
  const preview = app.querySelector("[data-preview]");
  const submit = app.querySelector("[data-submit]");
  const cancel = app.querySelector("[data-cancel]");
  const errorText = app.querySelector("[data-file-error]");
  const progress = app.querySelector("[data-progress]");
  const errorState = app.querySelector("[data-error-state]");
  const result = app.querySelector("[data-result]");
  const liveStatus = app.querySelector("[data-live-status]");
  let file = null;
  let previewURL = "";
  let controller = null;
  let submitting = false;
  let currentAnalysisId = "";
  let stageTimers = [];

  setupFeedback(result, () => currentAnalysisId);

  const showOnly = (section) => {
    form.hidden = section !== form;
    progress.hidden = section !== progress;
    errorState.hidden = section !== errorState;
    result.hidden = section !== result;
  };

  const setFileError = (message = "") => {
    errorText.textContent = message;
    errorText.hidden = !message;
    input.setAttribute("aria-invalid", String(Boolean(message)));
  };

  const clearPreviewURL = () => {
    if (previewURL) URL.revokeObjectURL(previewURL);
    previewURL = "";
  };

  const validateFile = (candidate) => {
    if (!candidate) return "Choose a photograph before requesting a critique.";
    if (!ALLOWED_TYPES.has(candidate.type)) return "Use a JPEG, PNG, or WebP photograph.";
    if (candidate.size > MAX_BYTES) return "This photograph is larger than 16 MB. Export a smaller copy and try again.";
    return "";
  };

  const chooseFile = (candidate) => {
    const validation = validateFile(candidate);
    setFileError(validation);
    if (validation) { file = null; submit.disabled = true; return; }
    file = candidate;
    clearPreviewURL();
    previewURL = URL.createObjectURL(file);
    preview.src = previewURL;
    app.querySelector("[data-file-name]").textContent = file.name;
    app.querySelector("[data-file-size]").textContent = `${(file.size / 1024 / 1024).toFixed(file.size > 1024 * 1024 ? 1 : 2)} MB`;
    empty.hidden = true;
    selectedView.hidden = false;
    app.querySelector("[data-file-actions]").hidden = false;
    submit.disabled = false;
  };

  const resetFile = () => {
    file = null; input.value = ""; clearPreviewURL(); preview.removeAttribute("src");
    empty.hidden = false; selectedView.hidden = true; app.querySelector("[data-file-actions]").hidden = true;
    submit.disabled = true; setFileError();
  };

  const startPresentationProgress = () => {
    const labels = ["Reading the photograph", "Reviewing visual evidence", "Composing the critique"];
    const widths = [24, 58, 82];
    stageTimers.forEach(window.clearTimeout); stageTimers = [];
    labels.forEach((label, index) => {
      const timer = window.setTimeout(() => {
        app.querySelector("[data-progress-title]").textContent = label;
        liveStatus.textContent = label;
        app.querySelector("[data-progress-bar]").style.width = `${widths[index]}%`;
        app.querySelectorAll("[data-stage]").forEach((stage, stageIndex) => stage.toggleAttribute("aria-current", stageIndex === index));
      }, index === 0 ? 0 : index * 3500);
      stageTimers.push(timer);
    });
  };

  const stopProgress = () => { stageTimers.forEach(window.clearTimeout); stageTimers = []; };

  const showError = (error) => {
    showOnly(errorState);
    const messages = {
      invalid_image: ["This file is not a usable photograph.", "Choose a JPEG, PNG, or WebP image and try again."],
      oversized: ["This photograph is too large.", "Export a copy under 16 MB, then choose it here."],
      rate_limited: ["FRAMED is at capacity.", error.message], unavailable: ["The critique service is unavailable.", error.message],
      timeout: ["The critique took too long.", error.message], aborted: ["Analysis cancelled.", "Your photograph remains selected. Restart whenever you are ready."],
      network_error: ["FRAMED could not be reached.", error.message], malformed_response: ["The response was incomplete.", error.message],
      server_error: ["The service could not finish this critique.", error.message], request_failed: ["The critique could not be completed.", error.message],
    };
    const [title, message] = messages[error.code] || ["Something interrupted the analysis.", "Try again, or choose another photograph."];
    app.querySelector("[data-error-title]").textContent = title;
    app.querySelector("[data-error-message]").textContent = message;
    app.querySelector("[data-retry]").hidden = !error.retryable;
    errorState.focus();
  };

  const submitAnalysis = async () => {
    if (submitting) return;
    const validation = validateFile(file);
    if (validation) { showOnly(form); setFileError(validation); dropZone.focus(); return; }
    submitting = true; controller = new AbortController();
    showOnly(progress); startPresentationProgress();
    try {
      const payload = await requestAnalysis(file, controller.signal);
      currentAnalysisId = payload.analysis_id;
      renderResult(result, payload, previewURL);
      showOnly(result); result.focus();
    } catch (error) { showError(error instanceof AnalysisError ? error : new AnalysisError("network_error", "The critique request failed.")); }
    finally { stopProgress(); controller = null; submitting = false; }
  };

  form.addEventListener("submit", (event) => { event.preventDefault(); submitAnalysis(); });
  input.addEventListener("change", () => chooseFile(input.files?.[0]));
  dropZone.addEventListener("click", (event) => { if (!event.target.closest("button")) input.click(); });
  dropZone.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); input.click(); } });
  ["dragenter", "dragover"].forEach((type) => dropZone.addEventListener(type, (event) => { event.preventDefault(); dropZone.classList.add("is-dragging"); }));
  ["dragleave", "drop"].forEach((type) => dropZone.addEventListener(type, (event) => { event.preventDefault(); dropZone.classList.remove("is-dragging"); }));
  dropZone.addEventListener("drop", (event) => chooseFile(event.dataTransfer?.files?.[0]));
  app.querySelector("[data-replace-image]").addEventListener("click", () => input.click());
  app.querySelector("[data-remove-image]").addEventListener("click", resetFile);
  cancel.addEventListener("click", () => controller?.abort());
  app.querySelector("[data-retry]").addEventListener("click", submitAnalysis);
  app.querySelector("[data-error-reset]").addEventListener("click", () => { resetFile(); showOnly(form); dropZone.focus(); });
  app.querySelector("[data-new-analysis]").addEventListener("click", () => { resetFile(); showOnly(form); form.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" }); dropZone.focus(); });
  app.querySelector("[data-evidence-toggle]").addEventListener("click", (event) => {
    const button = event.currentTarget; const content = app.querySelector("[data-evidence-content]"); const opening = content.hidden;
    content.hidden = !opening; button.setAttribute("aria-expanded", String(opening));
    button.querySelector("[data-disclosure-label]").textContent = opening ? "Hide evidence" : "Show evidence";
    button.lastElementChild.textContent = opening ? "−" : "+";
  });
}
