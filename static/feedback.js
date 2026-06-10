(function () {
  const form = document.getElementById("upload-form");
  const critiqueEl = document.getElementById("critique");
  const fbRow = document.getElementById("feedback-buttons");
  const correctionEl = document.getElementById("correction");
  const statusEl = document.getElementById("status");
  let lastPayload = null;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    statusEl.textContent = "Analyzing…";
    const fd = new FormData(form);
    const res = await fetch("/analyze", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) {
      statusEl.textContent = data.error || "Analyze failed";
      return;
    }
    const critique = data.critique || (data._ui && data._ui.critique) || "";
    critiqueEl.textContent = critique || "(no critique)";
    lastPayload = {
      image_id: data.metadata && data.metadata.photo_id ? data.metadata.photo_id : "",
      signature: (data.metadata && data.metadata.content_hash) || critique.slice(0, 64),
      critique_excerpt: critique.slice(0, 500),
    };
    fbRow.hidden = false;
    correctionEl.style.display = "block";
    statusEl.textContent = "Critique ready — send quick feedback below.";
  });

  async function sendFeedback(button) {
    if (!lastPayload) return;
    const body = {
      button,
      image_id: lastPayload.image_id,
      signature: lastPayload.signature,
      critique_excerpt: lastPayload.critique_excerpt,
      correction: correctionEl.value.trim(),
    };
    const res = await fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    statusEl.textContent = res.ok ? `Saved: ${button}` : (data.error || "Feedback failed");
  }

  fbRow.querySelectorAll("button[data-btn]").forEach((btn) => {
    btn.addEventListener("click", () => sendFeedback(btn.dataset.btn));
  });
})();
