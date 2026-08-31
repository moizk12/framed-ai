function setText(element, value, fallback = "Not available") {
  element.textContent = typeof value === "string" && value.trim() ? value : fallback;
}

function confidenceLabel(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "Confidence not reported";
  const percent = Math.round(Math.max(0, Math.min(1, value)) * 100);
  if (percent >= 80) return "High confidence";
  if (percent >= 55) return "Moderate confidence";
  return "Low confidence";
}

function groundingLabel(state, boxes) {
  if (state === "disabled") return "Detailed grounding was disabled for this analysis.";
  if (state === "empty") return "No localized grounding was available.";
  const count = Array.isArray(boxes) ? boxes.length : 0;
  return count ? `${count} localized evidence ${count === 1 ? "region" : "regions"} available.` : "Grounding available; no regions were supplied.";
}

function traceText(trace) {
  if (typeof trace === "string") return trace;
  if (!trace || typeof trace !== "object") return "";
  const claim = typeof trace.claim === "string" ? trace.claim : "Claim";
  const support = typeof trace.support === "string" ? trace.support : (typeof trace.reason === "string" ? trace.reason : "Evidence recorded");
  return `${claim}: ${support}`;
}

export function renderResult(root, payload, previewURL) {
  root.querySelector("[data-result-image]").src = previewURL;
  setText(root.querySelector("[data-analysis-id]"), payload.analysis_id.slice(0, 8));
  const duration = payload.meta?.duration_ms;
  root.querySelector("[data-duration]").textContent = typeof duration === "number" && duration >= 0 ? `${Math.max(1, Math.round(duration / 1000))} sec` : "Not reported";
  setText(root.querySelector("[data-critique]"), payload.critique, "No critique was returned.");

  const evidence = payload.evidence || {};
  setText(root.querySelector("[data-recognition]"), evidence.recognition?.text);
  root.querySelector("[data-confidence]").textContent = confidenceLabel(evidence.recognition?.confidence);
  setText(root.querySelector("[data-scene]"), evidence.scene?.type);
  root.querySelector("[data-grounding]").textContent = groundingLabel(evidence.grounding?.state, evidence.grounding?.boxes);

  const traces = Array.isArray(evidence.claim_traces) ? evidence.claim_traces.map(traceText).filter(Boolean) : [];
  const traceRegion = root.querySelector("[data-claim-traces]");
  const traceList = root.querySelector("[data-claim-trace-list]");
  traceList.replaceChildren();
  traces.forEach((trace) => { const item = document.createElement("li"); item.textContent = trace; traceList.append(item); });
  traceRegion.hidden = traces.length === 0;
  const hasEvidence = Boolean(evidence.recognition?.text || evidence.scene?.type || traces.length || evidence.grounding?.state === "available");
  root.querySelector("[data-evidence-empty]").hidden = hasEvidence;

  const limitations = Array.isArray(payload.limitations) && payload.limitations.length ? payload.limitations : ["FRAMED did not report a specific limitation. Treat the critique as one evidence-based reading, not a final judgment."];
  const limitationList = root.querySelector("[data-limitations]");
  limitationList.replaceChildren();
  limitations.forEach((limitation) => { const item = document.createElement("li"); item.textContent = limitation; limitationList.append(item); });
}
