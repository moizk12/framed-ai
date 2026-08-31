import { sendFeedback } from "./analysis-client.js";

export function setupFeedback(root, getAnalysisId) {
  const buttons = [...root.querySelectorAll("[data-feedback-value]")];
  const status = root.querySelector("[data-feedback-status]");
  const noteToggle = root.querySelector("[data-feedback-note-toggle]");
  const note = root.querySelector("[data-feedback-note]");
  const comment = root.querySelector("#feedback-comment");
  let selected = null;
  let sending = false;

  noteToggle.addEventListener("click", () => {
    const opening = note.hidden;
    note.hidden = !opening;
    noteToggle.setAttribute("aria-expanded", String(opening));
    if (opening) comment.focus();
  });

  buttons.forEach((button) => button.addEventListener("click", async () => {
    if (sending) return;
    selected = button.dataset.feedbackValue;
    buttons.forEach((candidate) => candidate.setAttribute("aria-pressed", String(candidate === button)));
    sending = true;
    buttons.forEach((candidate) => { candidate.disabled = true; });
    status.textContent = "Recording your feedback…";
    try {
      await sendFeedback({ analysisId: getAnalysisId(), useful: selected === "useful", comment: comment.value.trim() });
      status.textContent = "Thank you. Your feedback was attached to this analysis.";
    } catch (error) {
      status.textContent = error.message;
      buttons.forEach((candidate) => { candidate.disabled = false; });
    } finally { sending = false; }
  }));
}
