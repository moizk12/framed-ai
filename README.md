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

> **AI-Powered Photography Mentor** | Analyze your photographs through classical vision signals and modern embeddings, then receive thoughtful, mentor-like guidance on composition, lighting, mood, and artistic direction.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)

**🌐 Live Application:** [Try FRAMED on Hugging Face Spaces](https://huggingface.co/spaces/moizk12/framed-ai)  
**🔗 Direct URL:** https://moizk12-framed-ai.hf.space  
**📦 GitHub Repository:** https://github.com/moizk12/framed-ai

---

## 🎯 What is FRAMED?

FRAMED is a **visual-cognition companion for photographers**. It analyzes a photograph through:

- **Classical Vision Signals**: Frames, objects, symmetry, color, tonality
- **Modern Embeddings**: CLIP (semantic understanding) and YOLOv8 (object detection)
- **Mentor-Like Response**: Diagnoses what's working, what isn't, and suggests specific experiments ("Remix") and next shots
- **AskECHO**: A conversational Q&A endpoint grounded in image analysis and a light memory scaffold

Unlike simple image classifiers, FRAMED responds like a thoughtful human mentor—blending technical critique with artistic inspiration, referencing the wisdom of legendary photographers (Ansel Adams, Cartier-Bresson, Dorothea Lange, Fan Ho, and more).

### Key Features

- 📸 **Comprehensive Image Analysis**: Technical metrics (brightness, contrast, sharpness) + AI semantic understanding
- 🎨 **Color & Composition Analysis**: Color harmony, tonal range, lighting direction, symmetry, framing
- 🧠 **Genre & Mood Detection**: Automatically identifies genre (Portrait, Street, Landscape, etc.) and emotional mood
- 💬 **AskECHO**: Conversational AI that reflects on your photographic style and patterns
- 🎭 **Mentor Modes**: Choose from Balanced Mentor, Gentle Guide, Radical Visionary, Philosopher, or Curator
- 🔄 **Remix Suggestions**: AI-generated shot recipes and experimental ideas for your next shoot
- ☁️ **Server-Side Cloud Enhance**: OpenAI integration runs on the host—no user API keys required

---

## 🚀 Quick Start

### Option 1: Try the Live Application

**[Click here to use FRAMED on Hugging Face Spaces](https://huggingface.co/spaces/moizk12/framed-ai)**

No setup required—just upload a photo and get instant analysis!

### Option 2: Local Development

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
│    JSON Response + Remix            │
│    + ECHO Memory Update             │
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
4. **Building Context**: ECHO memory tracks your photographic patterns, allowing for deeper, personalized insights

### Origin Story

FRAMED began as a university project where we had to ship quickly. Some components were removed to stabilize the demo. The core question: **"Can we turn image analysis into guidance that feels like a thoughtful human mentor?"**

The constraint: simple hosting, no user API keys, low RAM targets; evolve into something original and independent (not "just ChatGPT").

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for Cloud Enhance features | No | None (features disabled) |
| `DEEPFACE_ENABLE` | Enable DeepFace for emotion analysis | No | `false` |
| `DATA_ROOT` | Root directory for persistent data | No | `/data` |
| `UPLOAD_DIR` | Directory for uploaded images | No | `/data/uploads` |
| `SECRET_KEY` | Flask secret key (change in production!) | No | `dev-secret-key-change-in-production` |

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

Run the import test to verify all dependencies are correctly installed:

```bash
python test_imports.py
```

Run the basic sanity test:

```bash
pytest tests/test_sanity.py
```

---

## 🚢 Deployment

### Hugging Face Spaces

The project is configured for one-click deployment on Hugging Face Spaces:

1. Push to your Hugging Face Space repository
2. Set the `OPENAI_API_KEY` secret in Space settings
3. The Dockerfile handles all dependencies and configuration

### Docker

Build and run locally:

```bash
docker build -t framed-ai .
docker run -p 7860:7860 -e OPENAI_API_KEY=your_key framed-ai
```

---

## 🔒 Security & Privacy

- **Host-Only API Keys**: OpenAI API key is stored server-side only; users never need their own keys
- **Image Processing**: Images are processed in-container and deleted after analysis (configurable retention)
- **No Tracking**: `.env` files are not tracked; `.env.example` documents required keys
- **CI/CD**: Automated testing with ruff, pytest, and bandit

---

## 🗺️ Roadmap

### Phase 2 (Planned)
- **ECHO Graph**: Unsupervised learning with UMAP/K-Means to cluster photos and discover patterns
- **Intent Lens → Assignment Cards**: Generate PDF assignment cards with signed ProofLens manifests
- **Remix Lab 3.0**: Enhanced shoot plans with PDF export
- **Mentor Selector**: Choose from different mentor personalities (Balanced/Gentle/Radical/Philosopher/Curator)
- **Rate Limits & Caching**: Hosted protections and performance optimizations

### Phase 3 (Future)
- Team/community features
- Mobile/PWA capture assistant
- Plugin system for custom analyzers
- Personal model adaptation
- Anonymized usage statistics

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
