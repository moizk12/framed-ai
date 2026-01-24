# FRAMED Update: Comprehensive Change Log

## Overview

This document details all changes made to FRAMED since fixing the GPT critique generation issue. This represents a comprehensive evolution of FRAMED from a working ML application to a production-grade visual intelligence system with a "visual conscience" that prevents contradictions and enforces truth.

---

## Table of Contents

1. [Initial Fix: GPT Critique Generation Issue](#initial-fix-gpt-critique-generation-issue)
2. [Phase II: Canonical Schema & Core Refactor](#phase-ii-canonical-schema--core-refactor)
3. [Phase III-A: Prompt Refactor (Evidence-Driven)](#phase-iii-a-prompt-refactor-evidence-driven)
4. [Phase III-B: Semantic Anchors & Grounded Critique](#phase-iii-b-semantic-anchors--grounded-critique)
5. [Phase III-C: Scene Understanding Synthesis Layer](#phase-iii-c-scene-understanding-synthesis-layer)
6. [Phase III-D: Unified Scene Understanding & Critique Governance](#phase-iii-d-unified-scene-understanding--critique-governance)
7. [Visual Conscience: Deterministic Computer Vision Grounding](#visual-conscience-deterministic-computer-vision-grounding)
8. [Visual Conscience Refinements](#visual-conscience-refinements)
9. [Technical Improvements](#technical-improvements)

---

## Initial Fix: GPT Critique Generation Issue

### Problem
The critique generation was not being called for the default "Balanced Mentor" mode, resulting in silent failures and placeholder text in the UI.

### Root Cause
In `framed/routes.py`, a conditional check `if mentor_mode and mentor_mode != "Balanced Mentor"` prevented `generate_merged_critique()` from being called for the default mode.

### Solution
- Removed the conditional check
- Critique generation now *always* runs for all mentor modes
- Fixed `NameError: name 'logger' is not defined` in `generate_merged_critique()` by adding a safeguard

### Files Changed
- `framed/routes.py`: Removed conditional check, always generate critique
- `framed/analysis/vision.py`: Added logger safeguard in `generate_merged_critique()`

### Impact
- Critiques now generate for all mentor modes
- No more silent failures or placeholder text
- Proper error logging for debugging

---

## Phase II: Canonical Schema & Core Refactor

### Objective
Refactor the FRAMED image analysis pipeline so that all perception and analysis output conforms to a single, deterministic schema, suitable for long-term memory (ECHO), portfolio analysis, and downstream intelligence.

### Key Changes

#### 1. Canonical Schema Introduction
- **New File**: `framed/analysis/schema.py`
- Defines `AnalysisResult` with top-level structure:
  ```python
  {
    "metadata": {...},
    "perception": {
      "technical": {...},
      "composition": {...},
      "color": {...},
      "lighting": {...},
      "aesthetics": {...},
      "semantics": {...},
      "emotion": {...}
    },
    "derived": {
      "emotional_mood": str,
      "genre": {...},
      "visual_interpretation": {...}
    },
    "confidence": {...},
    "errors": {}
  }
  ```

#### 2. `analyze_image()` Refactor
- Now builds canonical schema explicitly
- Never returns flat legacy dict
- Never raises on analyzer failure
- All analyzers wrapped: `{ "available": false }` on failure, `{ "available": true, ...results }` on success

#### 3. File Hash Caching
- SHA-256 hash computed for every uploaded image
- Cached analysis results stored under centralized runtime directory
- Cache lookup before analysis, cache save after analysis

#### 4. Downstream Compatibility
- Updated `run_full_analysis()`, `update_echo_memory()`, `summarize_echo_memory()`, `clean_result_for_ui()` to read from new schema
- Never assume optional fields exist
- Never crash if perception module unavailable

### Files Changed
- `framed/analysis/schema.py`: NEW - Canonical schema definition
- `framed/analysis/vision.py`: Refactored `analyze_image()` to use schema
- `framed/analysis/__init__.py`: Exported schema functions
- `framed/routes.py`: Updated to handle canonical schema

### Impact
- Consistent, deterministic output structure
- Robust error handling (partial results still returned)
- File hash caching improves performance
- Foundation for ECHO memory and portfolio analysis

---

## Phase III-A: Prompt Refactor (Evidence-Driven)

### Objective
Refactor FRAMED's critique prompt construction so that prompts receive ONLY observed facts from the canonical schema, and no interpretive prose is passed into prompts. Interpretation happens inside the prompt voice.

### Key Changes

#### 1. Evidence Extraction
- `generate_merged_critique()` now extracts values directly from canonical schema:
  - `perception.technical` → brightness, contrast, sharpness, tonal_range
  - `perception.composition` → symmetry, subject_position, subject_size, framing_style
  - `perception.color` → color_mood, color_harmony
  - `perception.lighting` → lighting_direction
  - `derived.genre` → genre_name, subgenre_name
  - `derived.emotional_mood` → emotional_mood

#### 2. Prompt Structure
- New prompt format with placeholders:
  ```
  TECHNICAL STATE
  - Brightness: {brightness}
  - Contrast: {contrast}
  ...
  
  COMPOSITION
  - Symmetry: {symmetry}
  ...
  ```

#### 3. Preserved Elements
- All mentor personas (Balanced, Radical, Philosopher, Curator)
- Philosophical voice and lineage (Adams, Fan Ho, Sontag, etc.)
- Refusal to be instructional
- Essay-like flow
- Provocation at the end

### Files Changed
- `framed/analysis/vision.py`: Refactored `generate_merged_critique()`

### Impact
- Critiques are sharper and more specific
- Less generic, more grounded in actual image data
- Interpretation happens in prompt voice, not in data preparation

---

## Phase III-B: Semantic Anchors & Grounded Critique

### Objective
Enhance critique specificity by introducing "Semantic Anchors" (high-confidence, multi-signal labels) and a "Critique Contract" that mandates their use.

### Key Concepts

#### 1. Semantic Anchors
- High-confidence, low-risk labels derived from multiple signals (CLIP, YOLO, composition)
- Sparse (only present if confidence > threshold)
- Examples: `scene_type: "Religious architecture at night"`, `structure_elements: ["domes", "minarets"]`

#### 2. CLIP Re-tasking
- Changed CLIP prompt from "Describe this image" to inventory-style extraction
- Three targeted prompts:
  - Structural inventory
  - Material & Condition inventory (NEW, critical for ivy/weathering)
  - Atmosphere & Environment inventory
- Returns list of nouns/attributes, not prose

#### 3. `generate_semantic_anchors()`
- Fuses CLIP inventory, tags, YOLO objects, composition data
- Uses keyword matching against canonicalized terms
- Implements per-anchor thresholds
- Returns sparse dictionary of anchors

#### 4. Critique Contract
- If anchors present, critic *must* name them
- If not, must not be implied
- Every interpretive claim must reference visible element or anchor

### Files Changed
- `framed/analysis/vision.py`: 
  - Added `get_clip_inventory()` (multi-prompt inventory extraction)
  - Added `generate_semantic_anchors()`
  - Updated `analyze_image()` to generate anchors
  - Updated `generate_merged_critique()` to include anchors in prompt
- `framed/analysis/schema.py`: Added `semantic_anchors` as optional field

### Impact
- Critiques can now name specific structures and elements
- Prevents generic descriptions
- Grounds interpretation in visible evidence

---

## Phase III-C: Scene Understanding Synthesis Layer

### Objective
Introduce a "Pre-Critique Scene Understanding Pass" to synthesize contextual meaning (material, temporal, atmospheric, emotional) before critique, making FRAMED "see" before it "speaks."

### Key Concepts

#### 1. Scene Understanding Layer
- Answers "What is materially, temporally, and atmospherically happening in this image?"
- Universal dimensions:
  - Material condition (surface_state, organic_growth, erosion_level, age_indicators)
  - Temporal context (time_scale, pace, endurance, change_indicators, moment_type)
  - Organic interaction (relationship, dominance, integration_level, specific_indicators)
  - Emotional substrate (temperature, pace, presence, quality, corrective_signals)
  - Contextual relationships

#### 2. `synthesize_scene_understanding()`
- Universal, lightweight, rule-based heuristics
- Sparse & confidence-gated (only emit when confidence is high)
- Corrective signals softly reweight interpretation

#### 3. Integration
- Called after all perception analysis, before semantic anchors
- Added to canonical schema as optional `scene_understanding` field
- Fed to critique as context (clearly labeled, read-only for critic)

### Files Changed
- `framed/analysis/vision.py`: 
  - Added `synthesize_scene_understanding()`
  - Integrated into `analyze_image()`
  - Updated `generate_merged_critique()` to include Scene Understanding
- `framed/analysis/schema.py`: Added `scene_understanding` as optional field

### Impact
- FRAMED now "sees" before it "speaks"
- Contextual understanding prevents misreads
- Corrective signals prevent obvious contradictions

---

## Phase III-D: Unified Scene Understanding & Critique Governance

### Objective
Ensure Scene Understanding *governs* interpretation, making corrective signals mandatory and enforcing hard governance rules to prevent contradictions.

### Key Changes

#### 1. Enhanced CLIP Inventory
- Replaced single inventory prompt with three targeted prompts:
  - Structural inventory
  - Material & Condition inventory (NEW, critical for ivy/weathering)
  - Atmosphere & Environment inventory
- Merged and deduplicated

#### 2. Scene Understanding - Authoritative
- Scene Understanding is now authoritative context, not optional
- Its output governs downstream interpretation

#### 3. Corrective Signals - Mandatory Locks
- Corrective signals are binding constraints
- If signal exists (e.g., `cold → warm_patience`), the "from" state is forbidden
- The "to" state becomes baseline
- Critique may not contradict

#### 4. Critique Governance Rules
- Added explicit, non-negotiable rules to prompt:
  - Must not contradict Scene Understanding
  - Must apply all Corrective Signals (mandatory locks, not suggestions)
  - Must reference organic growth/weathering if present
  - Must not invent human subjects if none detected
  - Must not describe as cold/sterile if warmth/organic integration present
  - Every interpretive claim must be grounded in Scene Understanding, Anchors, or Measured Evidence

#### 5. Prompt Structure Reorganization
- Strict order enforced:
  1. Mentor Instruction
  2. SCENE UNDERSTANDING (AUTHORITATIVE)
  3. CORRECTIVE SIGNALS (MANDATORY LOCKS)
  4. SEMANTIC ANCHORS
  5. VERIFIED OBSERVATIONS
  6. Task
  7. GOVERNANCE RULES

### Files Changed
- `framed/analysis/vision.py`: 
  - Enhanced `get_clip_inventory()` with multi-prompt extraction
  - Updated Scene Understanding heuristics
  - Reorganized `generate_merged_critique()` prompt structure
  - Added hard governance rules

### Impact
- Scene Understanding is now authoritative
- Corrective signals are mandatory, not suggestions
- Hard governance rules prevent contradictions
- Prompt structure enforces cognitive ordering

---

## Visual Conscience: Deterministic Computer Vision Grounding

### Objective
Move beyond text-based matching (CLIP) to actual visual perception using deterministic computer vision, providing ground truth that text matching cannot.

### Key Concepts

#### 1. Visual Feature Extraction
Three deterministic, provable functions:

**`detect_organic_growth(image_path)`**
- Uses HSV color thresholds to detect green pixels (ivy, moss, vegetation)
- Calculates coverage ratio (0.0-1.0)
- Spatial distribution analysis (vertical_surfaces, foreground, background, distributed)
- Cluster counting (connected components)
- Evidence: `["green_coverage=0.42", "green_clusters=5", "spatial_distribution=vertical_surfaces"]`
- Confidence: 0.95 for high coverage, 0.50 for minimal

**`detect_material_condition(image_path)`**
- Uses texture variance to detect surface roughness
- Edge degradation analysis (age indicators)
- Condition inference: "weathered" | "pristine" | "moderate" | "degraded"
- Evidence: `["texture_variance=0.68", "edge_degradation=0.45", "condition=weathered"]`
- Confidence: 0.90 for clear signals, 0.65 for ambiguous

**`detect_organic_integration(image_path, green_mask, structure_edges)`**
- Uses morphological operations to detect overlap
- Calculates overlap ratio (green pixels overlapping structure edges)
- Proximity ratio (green near structure)
- Relationship: "reclamation" | "integration" | "coexistence" | "none"
- Evidence: `["overlap_ratio=0.65", "proximity_ratio=0.80", "relationship=reclamation"]`
- Confidence: 0.95 for clear overlap, 0.50 for no green

**`extract_visual_features(image_path)`**
- Main entry point
- Calls all three functions
- Returns unified visual evidence dict
- Overall confidence (weighted average)

#### 2. Constrained Emotional Synthesis
**`synthesize_emotional_substrate_constrained()`**
- Every emotional output must be explainable by upstream signals
- Each field contains:
  - `value`: emotional value (e.g., "warm_patience")
  - `evidence`: list of explainable evidence strings
  - `confidence`: float (0.0-1.0)
  - `source`: "visual_analysis" | "technical_analysis" | "multi_modal" | "default"
  - `contradictions`: forbidden states and reasons (when applicable)

#### 3. Scene Understanding Integration
- Visual evidence used as primary source (ground truth)
- Text matching used as fallback only if no visual evidence
- Evidence marked with "(visual)" and `source="visual_analysis"`
- Confidence weighting: visual (0.95) > text (0.70)

#### 4. Multi-Modal Fusion
- Material condition: visual evidence (0.95) > text matching (0.70)
- Organic interaction: visual evidence (0.90) > text matching (0.65)
- Emotional substrate: synthesized from visual + technical + scene context

#### 5. Critique Prompt Enhancement
- Scene Understanding section indicates when visual evidence is used
- "SCENE UNDERSTANDING (AUTHORITATIVE - GROUND TRUTH FROM PIXELS)" when visual evidence present
- Explicitly forbids contradictions
- Mandates referencing visual evidence explicitly

#### 6. Visual Evidence Validation
**`validate_visual_evidence()`**
- Checks overall confidence thresholds
- Detects internal contradictions
- Issues warnings for low-confidence evidence
- Flags critical issues

#### 7. Contradiction Detection
**`detect_contradictions()`**
- Compares visual evidence (ground truth) vs text inference
- Detects mismatches
- Recommends overrides when visual confidence is high (>0.75)
- Provides explainable reasons

#### 8. Edge Case Handling
- Dark images (< 30 brightness): Adjusted HSV thresholds, reduced confidence
- Bright images (> 225 brightness): Adjusted HSV thresholds, reduced confidence
- Prevents false positives in difficult conditions

#### 9. Enhanced Logging
- Detailed visual evidence logging
- Validation warnings and issues
- Contradiction detection results
- Override recommendations

### Files Changed
- `framed/analysis/vision.py`: 
  - Added `detect_organic_growth()`, `detect_material_condition()`, `detect_organic_integration()`
  - Added `extract_visual_features()`
  - Added `synthesize_emotional_substrate_constrained()`
  - Added `validate_visual_evidence()`, `detect_contradictions()`
  - Updated `synthesize_scene_understanding()` to use visual evidence
  - Updated `analyze_image()` to extract visual features
  - Updated `generate_merged_critique()` to indicate visual evidence

### Impact
- FRAMED now has ground truth from pixels, not just text inference
- Visual evidence prevents contradictions
- Edge cases handled appropriately
- Comprehensive logging for debugging

---

## Visual Conscience Refinements

### Objective
Add four critical refinements to strengthen the visual conscience and prevent incorrect interpretations.

### Key Changes

#### 1. Negative Evidence Concept
**Problem**: Critiques were inventing absence-as-meaning incorrectly (e.g., "no people" = alienation).

**Solution**: Added `negative_evidence` tracking:
```python
"negative_evidence": {
  "no_human_presence": True,
  "no_motion_detected": True,
  "no_artificial_surface_uniformity": True
}
```

**Impact**: 
- Prevents critiques from misinterpreting absence
- Distinguishes "no people" ≠ alienation, "no people" = stillness / endurance / pause
- Strengthens silence interpretation

#### 2. Separate Coverage from Salience
**Problem**: 42% green coverage could mean ivy on facade OR grass in foreground - different meanings.

**Solution**: Added `salience` field:
```python
"organic_growth": {
  "coverage": 0.42,
  "salience": "structural"  # structural | incidental | peripheral | distributed | minimal
}
```

**Impact**:
- Distinguishes ivy on facade (structural) from grass in foreground (incidental)
- Prevents misclassification in landscapes vs architecture
- Uses existing spatial distribution analysis

#### 3. Temporal Direction, Not Just Pace
**Problem**: "slow" and "historical" don't distinguish growth from decay.

**Solution**: Added `temporal_direction`:
```python
"temporal_direction": "accreting" | "decaying" | "static"
```

**Examples**:
- Ivy growth → accreting
- Crumbling stone → decaying
- Clean monument → static

**Impact**:
- Feeds emotional engine beautifully
- Avoids vague "timeless" language
- Distinguishes growth from decay

#### 4. Explicit Invariant Documentation
**Problem**: Future changes might accidentally violate the visual evidence hierarchy.

**Solution**: Added explicit invariant comment:
```python
# ========================================================
# INVARIANT (CRITICAL - DO NOT VIOLATE):
# Visual evidence > text inference > stylistic voice
# If violated, critique is invalid.
# ========================================================
```

**Impact**:
- Protects future-you from accidental regressions
- Makes hierarchy explicit in code
- Serves as documentation for maintainers

### Files Changed
- `framed/analysis/vision.py`: 
  - Added `negative_evidence` tracking in `synthesize_scene_understanding()`
  - Added `salience` to `detect_organic_growth()` return value
  - Added `salience` to material condition tracking
  - Added `temporal_direction` to temporal context
  - Added invariant comment before material condition section

### Impact
- Negative evidence prevents incorrect interpretations
- Salience distinguishes structural from incidental growth
- Temporal direction distinguishes growth from decay
- Invariant protects against regressions

---

## Technical Improvements

### 1. Error Handling & Graceful Degradation
- All analysis steps wrapped with `safe_analyze()`
- Partial results returned with error metadata
- One failing model doesn't crash entire analysis

### 2. Lazy Loading
- All heavy models (YOLO, CLIP, NIMA, OpenAI client) lazy-loaded
- Models load only on first use
- Subsequent calls reuse cached instance
- `/health` endpoint never triggers model loading

### 3. Runtime Paths
- Centralized runtime directory strategy
- All writable files go to `/data/framed` (HF) or `/tmp/framed` (local)
- Environment variable `FRAMED_DATA_DIR` for configuration
- All directories created with `os.makedirs(..., exist_ok=True)`

### 4. Dockerfile Fix
- Base image: `python:3.11-slim`
- No deprecated packages
- `gunicorn` in PATH
- `git-lfs` installed
- Single worker
- Environment variables set correctly

### 5. Logging
- INFO-level logging for model loading
- Detailed error logging with `exc_info=True`
- Visual evidence extraction logging
- Contradiction detection logging

---

## Summary of Evolution

FRAMED has evolved from a working ML application to a production-grade visual intelligence system with:

1. **Deterministic Schema**: All analysis conforms to canonical schema
2. **Evidence-Driven Prompts**: Prompts receive only observed facts
3. **Semantic Anchors**: High-confidence, multi-signal labels
4. **Scene Understanding**: Contextual synthesis before critique
5. **Visual Conscience**: Ground truth from pixels, not text inference
6. **Governance Rules**: Hard rules prevent contradictions
7. **Negative Evidence**: Tracks absence to prevent misinterpretation
8. **Salience**: Distinguishes structural from incidental growth
9. **Temporal Direction**: Distinguishes growth from decay
10. **Invariant Protection**: Explicit hierarchy prevents regressions

The system now "sees" before it "speaks," with visual evidence serving as ground truth that text inference cannot provide. The visual conscience ensures that FRAMED cannot contradict what the image actually contains, even if it would sound poetic or plausible.

---

## Files Modified

- `framed/analysis/vision.py`: Major refactoring, added visual grounding, Scene Understanding, semantic anchors
- `framed/analysis/schema.py`: NEW - Canonical schema definition
- `framed/analysis/__init__.py`: Exported schema functions
- `framed/routes.py`: Updated to handle canonical schema, always generate critique
- `framed/__init__.py`: Updated paths, health endpoint
- `framed/static/js/framed.js`: Removed raw JSON rendering, enforced `_ui` as only render source
- `framed/static/css/style.css`: Ethereal visual styling
- `framed/templates/index.html`: Updated structure
- `Dockerfile`: Fixed for production deployment
- `requirements.txt`: Updated dependencies

---

## Testing Recommendations

1. **Visual Evidence**: Test with images containing ivy, weathering, organic growth
2. **Edge Cases**: Test with very dark, very bright, monochrome images
3. **Negative Evidence**: Test with images containing no humans, no motion
4. **Salience**: Test with structural vs incidental organic growth
5. **Temporal Direction**: Test with accreting vs decaying vs static scenes
6. **Contradiction Detection**: Test with images where visual and text evidence conflict
7. **Governance Rules**: Verify critiques don't contradict Scene Understanding

---

## Future Enhancements

1. **Additional Visual Features**: Shadows, light patterns, motion blur detection
2. **Material Classification**: When confidence is high, classify materials (stone, concrete, wood)
3. **Learned Models**: If deterministic methods fail, add learned models with explainability
4. **Portfolio Mode**: Use canonical schema for portfolio-level analysis
5. **ECHO Enhancement**: Use Scene Understanding and visual evidence in ECHO memory

---

*Last Updated: 2026-01-22*
*Version: Visual Conscience v1.0*
