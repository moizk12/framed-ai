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

# FRAMED — Public Photography Critique Beta

> **Evidence-aware photography critique** | Upload a photograph, receive one Balanced Mentor critique with evidence and limitations, then attach feedback to that analysis.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)

**📦 GitHub Repository:** https://github.com/moizk12/framed-ai

The public beta exposes only `/api/v1/analyses` and `/api/v1/feedback`. Research cognition, ECHO, memory controls, research identities, ledgers, run IDs, and mentor-mode switching are not public capabilities.

---

## 🎯 What is FRAMED?

FRAMED is a **visual-cognition companion for photographers**. It analyzes a photograph through:

- **Classical Vision Signals**: Frames, objects, symmetry, color, tonality
- **Modern Embeddings**: CLIP (semantic understanding) and YOLOv8 (object detection)
- **Mentor-Like Response**: Diagnoses what is working, what is uncertain, and offers a specific photographic experiment
- **Public feedback**: A durable PostgreSQL relationship between an analysis and the user’s useful/not-useful response

Unlike simple image classifiers, FRAMED responds like a thoughtful human mentor—blending technical critique with artistic inspiration, referencing the wisdom of legendary photographers (Ansel Adams, Cartier-Bresson, Dorothea Lange, Fan Ho, and more).

### Key Features

- 📸 **Comprehensive Image Analysis**: Technical metrics (brightness, contrast, sharpness) + AI semantic understanding
- 🎨 **Color & Composition Analysis**: Color harmony, tonal range, lighting direction, symmetry, framing
- 🧠 **Genre & Mood Detection**: Automatically identifies genre (Portrait, Street, Landscape, etc.) and emotional mood
- 💬 **Evidence and limitations**: Public responses expose an allowlisted, versioned contract
- 🎭 **Balanced Mentor**: The only public critique mode
- 🔄 **Durable feedback**: Useful/not-useful responses attach to the originating public analysis
- ☁️ **Host-only providers**: Model credentials remain server-side; users do not submit API keys

---

## 🚀 Quick Start

### Local Development

#### Prerequisites

- Python 3.11+
- pip
- (Optional) OpenAI API key for Cloud Enhance features

#### Installation

```bash
# Clone the repository
git clone https://github.com/moizk12/framed-ai.git
cd framed-ai

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Set OpenAI API key for Cloud Enhance
# Windows:
set OPENAI_API_KEY=your_key_here
# macOS/Linux:
export OPENAI_API_KEY=your_key_here

# Run the application
python run.py
```

The application will be available at `http://127.0.0.1:7860`

#### Performance knobs (HF + local)

- **FRAMED_DATA_DIR**: Base writable data dir (caches, uploads, memory). On Hugging Face Spaces use `/data/framed`.
- **FRAMED_PERCEPTION_WORKERS**: Parallel perception worker cap (lower if you hit GPU/CPU pressure).
- **FRAMED_LOG_STAGE_TIMINGS**: Set to `true` to log per-request stage timings.
- **FRAMED_COMBINED_LAYERS_2_7**: Set to `true` (default) to combine layers 2–7 into one reasoning call; falls back to separate calls if needed.

Cold-start invariant: `/health` should not trigger model weight loads (CLIP/YOLO/NIMA) or OpenAI client initialization.

#### Sample Images

Try these types of images for interesting results:
- **Portraits**: Close-up human subjects with varied lighting
- **Street Photography**: Urban scenes with dynamic compositions
- **Landscapes**: Natural scenes with dramatic skies or lighting
- **Abstract/Conceptual**: Minimalist or experimental compositions

---

## 🏗️ System Architecture

```
┌─────────────┐
│   User      │
│  Uploads    │
│   Photo     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│         Flask Application           │
│  (run.py → framed/__init__.py)       │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│      Image Preprocessing            │
│  (OpenCV: resize, normalize)       │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│    Parallel Analysis Pipeline       │
│                                     │
│  ┌──────────┐  ┌──────────┐       │
│  │  YOLOv8   │  │   CLIP   │       │
│  │ (Objects) │  │ (Semantic)│      │
│  └──────────┘  └──────────┘       │
│                                     │
│  ┌──────────┐  ┌──────────┐       │
│  │  Color   │  │ Symmetry │       │
│  │ Analysis │  │  & Lines │       │
│  └──────────┘  └──────────┘       │
│                                     │
│  ┌──────────┐  ┌──────────┐       │
│  │ Lighting │  │  Tonal   │       │
│  │ Direction│  │  Range   │       │
│  └──────────┘  └──────────┘       │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│    Critique Synthesis               │
│  (Optional: OpenAI Cloud Enhance)   │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│    Allowlisted v1 response          │
│    + PostgreSQL feedback link       │
└─────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Core Framework
- **Python 3.11**: Modern Python with type hints
- **Flask 3.0.0**: Lightweight web framework
- **Gunicorn**: Production WSGI server

### Computer Vision & ML
- **OpenCV**: Image preprocessing and analysis
- **YOLOv8 (Ultralytics)**: Object detection and framing analysis
- **CLIP (OpenAI)**: Semantic understanding and genre detection
- **scikit-learn**: K-Means clustering for color analysis
- **PyTorch**: Deep learning backend

### AI Integration
- **OpenAI API**: GPT-4 Turbo for mentor critiques and Remix suggestions
- **Transformers (Hugging Face)**: CLIP model loading

### Optional Features
- **DeepFace**: Facial emotion analysis (feature-flagged via `DEEPFACE_ENABLE`)
- **NIMA (TensorFlow)**: Aesthetic scoring (optional, requires model weights)

### Deployment
- **Docker**: Containerized deployment
- **Hugging Face Spaces**: One-click deployment with Docker support

---

## 📖 Project Mission

### The Problem

Photographers often struggle to receive **specific, constructive, and inspiring feedback** on their work. Generic comments like "nice photo" or technical critiques without artistic context don't help photographers grow. There's a gap between technical analysis and meaningful artistic guidance.

### The Solution

FRAMED bridges this gap by:

1. **Combining Technical and Artistic Analysis**: Not just "brightness: 120" but "moody low-key exposure, reserved and atmospheric"
2. **Referencing Photography Masters**: Drawing from the wisdom of Ansel Adams, Cartier-Bresson, Dorothea Lange, Fan Ho, and others
3. **Providing Actionable Guidance**: Not just critique, but "Remix" suggestions—specific experiments to try next
4. **Keeping boundaries explicit**: Public analysis/feedback persistence remains separate from research cognition state

### Origin Story

FRAMED began as a university project where we had to ship quickly. Some components were removed to stabilize the demo. The core question: **"Can we turn image analysis into guidance that feels like a thoughtful human mentor?"**

The constraint: simple hosting, no user API keys, low RAM targets; evolve into something original and independent (not "just ChatGPT").

---

## 🔧 Public production configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `DATABASE_URL` | Dedicated Track A PostgreSQL connection URL | Yes | — |
| `SECRET_KEY` | Random Flask secret, at least 32 characters | Yes | — |
| `FRAMED_ENV` | Runtime validation mode | Yes | `production` in container |
| `FRAMED_PUBLIC_BETA_ONLY` | Fail closed around legacy/research-capable routes | Yes | `true` in container |
| `FRAMED_COGNITION_V1` | Research cognition switch; must remain false publicly | Yes | `false` in container |
| `OPENAI_API_KEY` | Required only when configured models use OpenAI | Conditional | — |
| `FRAMED_MODEL_A`, `FRAMED_MODEL_B` | Configured analysis/expression providers | Yes | See `.env.example` |
| `FRAMED_VERSION` | Safe release identifier returned by `/version` | Build metadata | `dev` |
| `FRAMED_BUILD_SHA` | Lowercase Git SHA returned by `/version` | Build metadata | `unknown` |
| `FRAMED_MAX_UPLOAD_BYTES` | Request body limit, 1–25 MiB | No | `12582912` |
| `FRAMED_MAX_IMAGE_PIXELS` | Decoded image pixel limit | No | `40000000` |
| `FRAMED_DATA_DIR` | Writable model cache and temporary upload root | No | `/data/framed` |
| `PUBLIC_AUTO_MIGRATE` | In-process migration fallback | No | `false` |

PostgreSQL must be reachable before application startup. Use a dedicated public database/user with permission to create and alter the `public_schema_migrations`, `public_analyses`, and `public_feedback` tables and their index. Do not point `DATABASE_URL` at the research cognition store. The container runs `python -m framed.public_migrations` transactionally under a PostgreSQL advisory lock before starting Gunicorn.

Runtime probes are deliberately separate: `/health` is process liveness, `/ready` checks public PostgreSQL connectivity, and `/version` returns only service/version/build/API-contract metadata.

### LM Studio (default local LLM)

1. Open **LM Studio**, download/load **Qwen2.5-VL-7B-Instruct** (or align `MODEL_CONFIGS` / `FRAMED_LOCAL_MODEL_*` with the exact id shown under **Local Server**).
2. Start the local server (default **http://localhost:1234**).
3. Defaults: `FRAMED_MODEL_A` and `FRAMED_MODEL_B` are `LOCAL_QWEN25_VL_7B` (see `.env.example`). Optional: set both to `LOCAL_GEMMA4_E4B` for Gemma, or override model id strings with `FRAMED_LOCAL_MODEL_A` / `FRAMED_LOCAL_MODEL_B`.
4. `FRAMED_STRICT_LOCAL=true` makes missing server/model a hard error (no silent fallback).

### Feature Flags

- **DeepFace**: Set `DEEPFACE_ENABLE=true` to enable facial emotion analysis (requires `deepface` package)
- **NIMA**: Aesthetic scoring is optional and requires TensorFlow + model weights

---

## 📁 Project Structure

```
framed-ai/
├── framed/                 # Main application package
│   ├── __init__.py        # Flask app factory
│   ├── routes.py           # API routes and endpoints
│   ├── analysis/
│   │   └── vision.py       # Core image analysis functions
│   ├── templates/          # HTML templates
│   │   ├── index.html
│   │   ├── upload.html
│   │   └── result.html
│   └── static/            # Static assets
│       ├── css/
│       ├── js/
│       └── audio/
├── run.py                  # Application entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
└── README.md              # This file
```

---

## 🧪 Testing

```bash
pytest -q
```

Health check only:

```bash
pytest tests/test_health.py -q
```

### Local model A/B (optional)

With LM Studio running and a vision model loaded:

```bash
python scripts/ab_test_local_models.py --image path/to/your.jpg
```

Outputs are saved under `eval_outputs/`.

---

## 🚢 Deployment

Deploy the production container on a host that provides persistent writable cache storage, a reachable dedicated PostgreSQL database, and the required environment variables below. No deployment occurs automatically from this branch or PR.

### Docker

Build and run locally:

```bash
docker build -t framed-ai --build-arg FRAMED_VERSION=beta.1 --build-arg FRAMED_BUILD_SHA=$(git rev-parse HEAD) .
docker run --read-only --tmpfs /tmp --mount type=volume,src=framed-data,dst=/data/framed -p 7860:7860 -e DATABASE_URL=postgresql://... -e SECRET_KEY=replace-with-32-random-characters -e OPENAI_API_KEY=... framed-ai
```

The public v1 API requires a PostgreSQL `DATABASE_URL`. App startup applies pending
public-only migrations automatically; they can also be applied explicitly with:

```bash
python -m framed.public_migrations
```

The public tables are independent of all research cognition and SQLite state.

The image installs `requirements.lock` with `pip --require-hashes`. Regenerate the lock intentionally with Python 3.11 resolution using `scripts/lock_runtime_dependencies.ps1` (requires `uv`), then review and commit the lock diff. The production startup sequence is:

```bash
python -m framed.public_migrations
exec gunicorn -k gthread --threads 4 -w 1 --timeout 120 --keep-alive 5 -b 0.0.0.0:${PORT:-7860} run:app
```

---

## 🔒 Security & Privacy

- **Host-Only API Keys**: OpenAI API key is stored server-side only; users never need their own keys
- **Image Processing**: Images are processed in-container and deleted after analysis (configurable retention)
- **No Tracking**: `.env` files are not tracked; `.env.example` documents required keys
- **CI/CD**: Automated testing with ruff, pytest, and bandit

---

## Future ideas

- Research-only ECHO clustering/graph experiments (not a public capability)
- Assignment cards + export
- Remix improvements
- More mentor voices
- Hosted protections (rate limits, caching)
- Mobile/PWA helper, plugins

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

FRAMED is inspired by the wisdom and vision of legendary photographers:
- Ansel Adams (tonal previsualization)
- Henri Cartier-Bresson (decisive moments)
- Dorothea Lange (human condition)
- Fan Ho (light and shadow)
- Gregory Crewdson (cinematic narrative)
- Saul Leiter (color as emotion)
- And many more...

---

## 📧 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/moizk12/framed-ai/issues)
- **Live Demo**: [Try FRAMED on Hugging Face Spaces](https://huggingface.co/spaces/moizk12/framed-ai)

---

**Built with ❤️ for photographers who seek to grow beyond technical perfection into artistic expression.**
