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

# FRAMED — Visual Soul Companion

**Live app (Space):** https://huggingface.co/spaces/moizk12/framed-ai  
**Direct app URL:** https://moizk12-framed-ai.hf.space  
**GitHub repo:** https://github.com/moizk12/framed-ai

FRAMED analyzes a photograph and responds like a mentor: genre/subgenre, mood, lighting/tonal structure, lines/symmetry, color harmony, subject framing, shot recipes (“Remix”), and Ask-ECHO. Cloud Enhance (OpenAI) runs **server-side**; users never need their own key.

## Quickstart (local)

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
# optional for cloud enhance (server-side in Space)
set OPENAI_API_KEY=your_key_here
python app.py   # http://127.0.0.1:5000
