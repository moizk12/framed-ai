<h1 align="center">FRAMED — Visual Soul Companion (framed-ai)</h1>

<p align="center">
  <a href="https://github.com/moizk12/framed-ai/actions/workflows/ci.yml">
    <img alt="Build" src="https://github.com/moizk12/framed-ai/actions/workflows/ci.yml/badge.svg?branch=main">
  </a>
  <a href="LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg">
  </a>
  <a href="https://github.com/moizk12/framed-ai/stargazers">
    <img alt="Stars" src="https://img.shields.io/github/stars/moizk12/framed-ai?style=social">
  </a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
</p>

<p align="center"><b>AI that maps your photographic identity, writes poetic critiques, and gives assignment cards that push your craft.</b></p>

<p align="center">
  <img src="docs/demo.gif" alt="FRAMED demo" width="760">
</p>


## Why it matters
- **Identity, not just analysis:** Builds your ECHO voiceprint across many photos.
- **Growth you can shoot:** Generates concrete assignment cards (lens/time/scene/sequence).
- **Independent & verifiable:** Local analysis + optional cloud “poet,” all artifacts signed.

## What’s inside
- **Core analyzers:** YOLOv8, OpenCLIP, DeepFace (optional), NIMA, color/tonal/lines
- **Critique brain:** Local poet (offline) with optional GPT “Cloud Enhance”
- **ECHO graph:** Clusters motifs and style evolution over time
- **Remix 2.0:** Shot recipes and next-shoot plans
- **Proof:** JSON + PDF reports cryptographically signed (ProofLens-ready)

## 1-minute Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python app/main.py
# Open http://127.0.0.1:5000
How it works
mermaid
Copy code
flowchart LR
A[Upload Photos] --> B[Preprocess]
B --> C[Local Analyzers: YOLO/CLIP/DeepFace/NIMA]
C --> D[Consolidated JSON Schema]
D --> E[ECHO Identity & Clusters]
D --> F[Critique & Remix (Local Poet / Cloud)]
E --> G[Assignment Cards]
F --> H[PDF & Signed Manifests]
Roadmap
 Portfolio ingest + ECHO graph (UMAP + KMeans)

 Assignment Cards (Intent Lens)

 Local Poet (LoRA) + RAG quotes (offline)

 Cloud Enhance toggle (GPT)

 Export: PDF + signed JSON

License
MIT © Moiz Kashif