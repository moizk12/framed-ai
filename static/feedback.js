(function () {
  const form = document.getElementById("upload-form");
  const critiqueEl = document.getElementById("critique");
  const fbRow = document.getElementById("feedback-buttons");
  const correctionEl = document.getElementById("correction");
  const statusEl = document.getElementById("status");
  const evidenceDetails = document.getElementById("evidence-details");
  const evidenceInspector = document.getElementById("evidence-inspector");
  let lastPayload = null;

  function tierClass(tier) {
    if (tier === "Restricted") return "ei-block ei-tier-restricted";
    if (tier === "Limited·cautious") return "ei-block ei-tier-cautious";
    return "ei-block";
  }

  function renderEvidenceInspector(inspector) {
    if (!evidenceInspector || !evidenceDetails) return;
    evidenceInspector.innerHTML = "";
    if (!inspector || typeof inspector !== "object") {
      evidenceDetails.hidden = true;
      return;
    }
    evidenceDetails.hidden = false;

    const rec = inspector.recognition || {};
    const recBlock = document.createElement("div");
    recBlock.className = "ei-block";
    recBlock.innerHTML =
      "<strong>Recognition (Inferred)</strong><p>" +
      (rec.what_i_see || "(none)") +
      "</p>";
    evidenceInspector.appendChild(recBlock);

    const scene = inspector.scene || {};
    const sceneBlock = document.createElement("div");
    sceneBlock.className = "ei-block";
    sceneBlock.innerHTML =
      "<strong>Scene</strong><p>type=" +
      (scene.scene_type || "—") +
      " · category=" +
      (scene.category || "—") +
      "</p>";
    evidenceInspector.appendChild(sceneBlock);

    const grounding = inspector.grounding || {};
    const gBlock = document.createElement("div");
    gBlock.className = "ei-block";
    const boxCount = Array.isArray(grounding.render_boxes) ? grounding.render_boxes.length : 0;
    gBlock.innerHTML =
      "<strong>Grounding (Observed)</strong><p>state=" +
      (grounding.state || "unknown") +
      " · render_boxes=" +
      boxCount +
      "</p>";
    evidenceInspector.appendChild(gBlock);

    const traces = inspector.claim_traces || [];
    if (traces.length) {
      const ct = document.createElement("div");
      ct.className = "ei-block";
      ct.innerHTML = "<strong>Claim licensing</strong>";
      const ul = document.createElement("ul");
      traces.forEach(function (t) {
        const li = document.createElement("li");
        li.textContent = (t.claim || "?") + ": " + (t.tier || "Unavailable");
        ul.appendChild(li);
      });
      ct.appendChild(ul);
      evidenceInspector.appendChild(ct);
    }

    const prov = inspector.provenance || {};
    const note = document.createElement("p");
    note.className = "ei-muted";
    note.textContent =
      "Read-only evidence chain · probe_enabled=" +
      String(!!prov.grounding_probe_enabled) +
      " · not causal proof";
    evidenceInspector.appendChild(note);
  }

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
    const inspector = data._ui && data._ui.evidence_inspector;
    renderEvidenceInspector(inspector);
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
