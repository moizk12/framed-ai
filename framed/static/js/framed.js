// ========== FRAMED VISUAL SOUL UI ENGINE ==========

// === Global Elements ===
const introSound = document.getElementById("introSound");
const idleLoop = document.getElementById("idleLoop");
const analyzePing = document.getElementById("analyzePing");
const preview = document.getElementById("imagePreview");

// === Image Preview ===
function previewImage(input) {
  const file = input.files[0];
  if (file && preview) {
    const reader = new FileReader();
    reader.onload = function (e) {
      preview.src = e.target.result;
      preview.classList.remove("hidden");
    };
    reader.readAsDataURL(file);
  }
}

// === Show Loading Spinner ===
function showLoading(id, message = "Analyzing your image...") {
  const el = document.getElementById(id);
  el.innerHTML = `
    <div class="loading-container">
      <div class="spinner"></div>
      <p class="loading-text">${message}</p>
    </div>
  `;
}

// === Tab Switching ===
function switchTab(tabId) {
    const tabs = document.querySelectorAll(".tab");
    const panels = document.querySelectorAll(".tab-panel");
  
    tabs.forEach(tab => tab.classList.remove("active"));
    panels.forEach(panel => {
      panel.classList.add("hidden");
      panel.classList.remove("fade-in");
    });
  
    document.getElementById(tabId).classList.add("active");
  
    const targetPanel = document.getElementById(`${tabId}Panel`);
    targetPanel.classList.remove("hidden");
    targetPanel.classList.add("fade-in");
}
  

// === Play Analyze Ping ===
function playAnalyzePing() {
  analyzePing.currentTime = 0;
  analyzePing.volume = 0.6;
  analyzePing.play();
}

// === Submit Form to /analyze ===
async function submitImage(event) {
  event.preventDefault();
  const form = document.getElementById("uploadForm");
  const formData = new FormData(form);

  showLoading("results", "Interpreting your visual soul...");

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      const message =
        (data && data.error) ||
        "FRAMED fell silent for a moment. Try another image, or breathe and try again.";
      document.getElementById("results").innerHTML = `
        <div class="result-shell result-shell-error fade-in">
          <div class="result-section result-section-error">
            <div class="result-label">Something went quiet</div>
            <p class="result-body">${message}</p>
          </div>
        </div>
      `;
      switchTab("resultsTab");
      return;
    }

    // STEP 2: Enforce _ui as the ONLY render source
    // If _ui does NOT exist, render a gentle error message
    // DO NOT render data directly
    if (!data._ui) {
      document.getElementById("results").innerHTML = `
        <div class="result-shell result-shell-error fade-in">
          <div class="result-section result-section-error">
            <div class="result-label">The reading is incomplete</div>
            <p class="result-body">FRAMED received the image but could not prepare a quiet reading. Please try again, or return with another frame.</p>
          </div>
        </div>
      `;
      switchTab("resultsTab");
      return;
    }

    // Use ONLY _ui - no fallbacks to raw data
    const view = data._ui;

    // Extract fields from _ui (flat structure, not nested)
    const caption = view.caption || "";
    const critique = view.critique || "";
    const remix = view.remix_prompt || "";
    const mood = view.emotional_mood || "";
    const poeticMood = view.poetic_mood || "";
    const genre =
      (typeof view.genre === "string" && view.genre) ||
      (view.genre && view.genre.genre) ||
      "";
    const subgenre = view.subgenre || "";
    const colorMood = view.color_mood || "";
    const lighting = view.lighting_direction || "";

    const mentorModeSelect = document.getElementById("mentorMode");
    const mentorTone = mentorModeSelect ? mentorModeSelect.value : "Balanced Mentor";

    const resultsEl = document.getElementById("results");
    const renderedContent = `
      <div class="result-shell fade-in">
        <div class="result-header">
          <div class="result-title">A Quiet Reading of Your Frame</div>
          <div class="result-mentor">Mentor Tone · ${mentorTone}</div>
        </div>

        <div class="result-section result-section-core">
          <div class="result-label">Core Critique</div>
          <p class="result-body">${critique ||
            "The image has been read, but FRAMED is choosing silence over empty words."}</p>
        </div>

        <div class="result-section-grid">
          <div class="result-section">
            <div class="result-label">Visual Mood</div>
            <p class="result-body">
              ${poeticMood ||
                mood ||
                "A mood still forming — soft, undecided, almost on the verge of speaking."}
            </p>
          </div>
          <div class="result-section">
            <div class="result-label">Color &amp; Light</div>
            <p class="result-body">
              ${colorMood || lighting
                ? [colorMood, lighting].filter(Boolean).join(" · ")
                : "Light and color sit quietly here, more whisper than announcement."}
            </p>
          </div>
          <div class="result-section">
            <div class="result-label">Subject &amp; Genre</div>
            <p class="result-body">
              ${caption
                ? `“${caption}”`
                : "The subject stays unnamed, but the frame still remembers being seen."}
              ${genre || subgenre
                ? `<br/><span class="result-subtext">${genre}${
                    genre && subgenre ? " · " : ""
                  }${subgenre || ""}</span>`
                : ""}
            </p>
          </div>
        </div>

        <div class="result-section result-section-remix">
          <div class="result-label">If You Want to Push Further</div>
          <p class="result-body">
            ${
              remix ||
              "Remix mode hums in the background. Set an OpenAI key on the host to let FRAMED dream new variations out loud."
            }
          </p>
        </div>
      </div>
    `;

    // STEP 4: Temporary DOM-level verification
    if (typeof renderedContent === "string" && renderedContent.includes("{")) {
      console.error("❌ Raw JSON detected in UI rendering path");
    }

    resultsEl.innerHTML = renderedContent;

    // Verify no raw JSON in DOM
    const domText = resultsEl.textContent || "";
    if (domText.includes('"') && (domText.includes("brightness") || domText.includes("contrast") || domText.includes("errors"))) {
      console.error("❌ Raw JSON detected in DOM output");
    }

    switchTab("resultsTab");

    if (mood) {
      applyMoodAudio(mood);
    }
  } catch (err) {
    console.error("Analysis failed:", err);
    document.getElementById("results").innerHTML = `
      <div class="result-shell result-shell-error fade-in">
        <div class="result-section result-section-error">
          <div class="result-label">The server lost its light for a moment</div>
          <p class="result-body">Please try again with the same image, or return with a new frame when you are ready.</p>
        </div>
      </div>
    `;
    switchTab("resultsTab");
  }
}


// === Ask ECHO Question ===
async function askEchoQuestion() {
  const question = document.getElementById("echoQuestion").value.trim();
  const resultBox = document.getElementById("echoResult");

  if (!question) return;

  showLoading("echoResult", "ECHO is reflecting...");

  try {
    const response = await fetch("/ask-echo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question }),
    });

    const payload = await response.json();
    const text = payload.answer || payload.response || "ECHO is quiet, but your question is still echoing in the darkroom.";
    resultBox.innerHTML = `<div class="echo-output">${text}</div>`;
  } catch (err) {
    console.error("ECHO failed:", err);
    resultBox.innerHTML = `<p class="error">Something went wrong.</p>`;
  }
}

// === On Window Load: Preloader + Audio Boot ===
window.addEventListener("load", () => {
    const preloader = document.getElementById("preloader");
    if (preloader) {
      setTimeout(() => {
        preloader.style.display = "none";
      }, 3000);
    }
  
    // Start ambient intro audio
    if (introSound && idleLoop) {
      introSound.volume = 0.4;
      introSound.play().catch(e => console.log("Intro autoplay blocked"));
  
      setTimeout(() => {
        idleLoop.volume = 0.2;
        idleLoop.play().catch(e => console.log("Idle loop autoplay blocked"));
      }, 3000);
    }
  });
  

// ========== MOOD AUDIO CONTROLLER ==========

const moodAudio = document.getElementById("moodAudio");

// Mood to Audio File Mapping
const moodTracks = {
  "dreamy": "ethereal_ambient.mp3",
  "moody": "melancholy_lull.mp3",
  "joyful": "fractured_life.mp3",
  "peaceful": "paper_plane.mp3",
  "chaotic or busy": "dynamic_urban_vibes.mp3",
  "cold and distant": "cold_isolation.mp3"
};

// Function to Fade Out Audio
function fadeOut(audio) {
  const fade = setInterval(() => {
    if (audio.volume > 0.05) {
      audio.volume -= 0.05;
    } else {
      audio.pause();
      clearInterval(fade);
    }
  }, 100);
}

// Function to Fade In Audio
function fadeIn(audio, src) {
  audio.src = src;
  audio.volume = 0;
  audio.play();
  const rise = setInterval(() => {
    if (audio.volume < 0.5) {
      audio.volume += 0.05;
    } else {
      clearInterval(rise);
    }
  }, 100);
}

// Function to Apply Mood-Based Audio
function applyMoodAudio(mood) {
  const moodKey = Object.keys(moodTracks).find(key => mood.includes(key));
  if (!moodKey) return;

  const audioFile = `/static/audio/${moodTracks[moodKey]}`;
  fadeOut(moodAudio);
  setTimeout(() => fadeIn(moodAudio, audioFile), 1000);
}


document.addEventListener("DOMContentLoaded", () => {
    const audio = document.getElementById('ambientTrack');
    if (audio) {
        document.body.addEventListener("click", () => {
            audio.play().catch(err => console.warn("Audio autoplay failed:", err));
        }, { once: true });
    }
});




  
