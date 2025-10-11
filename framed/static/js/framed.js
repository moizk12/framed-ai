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
  
      const html = await response.text();
      document.getElementById("results").innerHTML = html;
      switchTab("resultsTab");
  
      // Extract mood from hidden field if sent
      const match = html.match(/data-mood="([^"]+)"/);
      if (match && match[1]) {
        applyMoodAudio(match[1]);
      }
    } catch (err) {
      console.error("Analysis failed:", err);
      alert("Something went wrong. Try again.");
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

    const { response: text } = await response.json();
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




  
