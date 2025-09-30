---
title: FRAMED — Visual Soul Companion
emoji: 🎞️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# FRAMED — Visual Soul Companion (framed-ai)

![Build](https://github.com/moizk12/framed-ai/actions/workflows/ci.yml/badge.svg?branch=main)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Stars](https://img.shields.io/github/stars/moizk12/framed-ai?style=social)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

**Live app (Space):** https://huggingface.co/spaces/moizk12/framed-ai  
**Direct app URL:** https://moizk12-framed-ai.hf.space  
**GitHub repo:** https://github.com/moizk12/framed-ai

FRAMED analyzes a photograph and speaks back in a mentor’s voice — mapping genre/subgenre, emotional mood, lighting/tonal structure, line/symmetry, color harmony, subject framing, and more. It then proposes shot-recipes (“Remix”) and answers questions via **Ask-ECHO**. Cloud Enhance (OpenAI) is **host-side** only — users never need their own keys.

---

## Why it matters
- **Find your voice:** turn raw analysis into guidance that sounds like a human mentor.  
- **Actionable next steps:** shot recipes, framing moves, light changes, timing suggestions.  
- **Built for photographers:** fast local analyzers; optional GPT enhancement on the server.

---

## What’s inside
- **Analyzers:** YOLOv8 (object cues + framing), OpenCLIP (caption/tags), DeepFace (+ CLIP fallback), color clusters + harmony, lines/symmetry, tonal/lighting stats, visual interpretation.
- **ECHO:** memory scaffolding + Q&A endpoint (rate-limit ready).
- **Remix Lab:** prompt → shot recipe generator.
- **Web:** Flask, templates, static JS/CSS, JSON endpoints.
- **Infra:** Dockerfile (ports 7860), Git LFS for audio, CI (ruff/pytest/bandit).

---

## Quickstart (local)

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
set OPENAI_API_KEY=your_key_here   # PowerShell: $env:OPENAI_API_KEY="..."
python app.py
# visit http://127.0.0.1:5000
