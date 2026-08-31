const ANALYSIS_ENDPOINT = "/api/v1/analyses";
const FEEDBACK_ENDPOINT = "/api/v1/feedback";
const REQUEST_TIMEOUT_MS = 300_000;

export class AnalysisError extends Error {
  constructor(code, message, options = {}) {
    super(message);
    this.name = "AnalysisError";
    this.code = code;
    this.retryable = options.retryable ?? true;
    this.status = options.status ?? 0;
  }
}

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function validateAnalysisDTO(payload) {
  if (!isObject(payload) || payload.status !== "complete") return false;
  if (typeof payload.request_id !== "string" || typeof payload.analysis_id !== "string") return false;
  if (typeof payload.critique !== "string" || !isObject(payload.evidence)) return false;
  const evidence = payload.evidence;
  if (!isObject(evidence.recognition) || typeof evidence.recognition.text !== "string") return false;
  if (!isObject(evidence.scene) || typeof evidence.scene.type !== "string") return false;
  if (!isObject(evidence.grounding) || !["available", "empty", "disabled"].includes(evidence.grounding.state)) return false;
  if (!Array.isArray(evidence.grounding.boxes) || !Array.isArray(evidence.claim_traces)) return false;
  if (!Array.isArray(payload.limitations) || !payload.limitations.every((item) => typeof item === "string")) return false;
  return isObject(payload.meta) && payload.meta.contract_version === "1";
}

export function validateFeedbackDTO(payload, analysisId) {
  return isObject(payload)
    && payload.status === "recorded"
    && payload.analysis_id === analysisId
    && typeof payload.request_id === "string"
    && isObject(payload.meta)
    && payload.meta.contract_version === "1";
}

async function parseJSON(response) {
  try { return await response.json(); } catch { return null; }
}

function messageForStatus(status, payload) {
  void payload;
  if (status === 400) return new AnalysisError("invalid_image", "This file could not be read as a supported photograph.", { retryable: false, status });
  if (status === 415) return new AnalysisError("invalid_image", "This file is not a supported JPEG, PNG, or WebP photograph.", { retryable: false, status });
  if (status === 413) return new AnalysisError("oversized", "The photograph is larger than the 12 MB upload limit.", { retryable: false, status });
  if (status === 429) return new AnalysisError("rate_limited", "FRAMED is receiving more requests than it can process right now. Please wait a moment and try again.", { status });
  if (status === 504) return new AnalysisError("timeout", "The critique took longer than expected and was stopped. You can safely try again.", { status });
  if (status === 503) return new AnalysisError("unavailable", "The critique service is temporarily unavailable. Your photograph was not successfully analyzed.", { status });
  if (status >= 500) return new AnalysisError("server_error", "FRAMED encountered a service error before the critique was completed.", { status });
  return new AnalysisError("request_failed", "The critique request could not be completed.", { status });
}

export async function requestAnalysis(file, externalSignal) {
  const controller = new AbortController();
  let didTimeout = false;
  const timer = window.setTimeout(() => { didTimeout = true; controller.abort(); }, REQUEST_TIMEOUT_MS);
  const abortFromExternal = () => controller.abort();
  externalSignal?.addEventListener("abort", abortFromExternal, { once: true });
  const body = new FormData();
  body.append("image", file);
  body.append("mentor_mode", "balanced");
  try {
    const response = await fetch(ANALYSIS_ENDPOINT, { method: "POST", body, signal: controller.signal, headers: { Accept: "application/json" } });
    const payload = await parseJSON(response);
    if (!response.ok) throw messageForStatus(response.status, payload);
    if (!validateAnalysisDTO(payload)) throw new AnalysisError("malformed_response", "FRAMED returned an incomplete response. No critique is being shown as complete.");
    return payload;
  } catch (error) {
    if (error instanceof AnalysisError) throw error;
    if (error?.name === "AbortError") {
      if (didTimeout) throw new AnalysisError("timeout", "The critique took longer than expected and was stopped. You can safely try again.");
      throw new AnalysisError("aborted", "The analysis was cancelled.", { retryable: true });
    }
    throw new AnalysisError("network_error", "FRAMED could not reach the critique service. Check your connection and try again.");
  } finally {
    window.clearTimeout(timer);
    externalSignal?.removeEventListener("abort", abortFromExternal);
  }
}

export async function sendFeedback({ analysisId, useful, comment }) {
  const response = await fetch(FEEDBACK_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ analysis_id: analysisId, useful, comment: comment || "" }),
  });
  const payload = await parseJSON(response);
  if (!response.ok || !validateFeedbackDTO(payload, analysisId)) {
    throw new AnalysisError("feedback_failed", "Feedback could not be saved. Your critique remains available on this page.");
  }
  return payload;
}
