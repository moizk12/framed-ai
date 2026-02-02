# FRAMED Update: Comprehensive Change Log

## Overview

This document details all changes made to FRAMED since fixing the GPT critique generation issue. This represents a comprehensive evolution of FRAMED from a working ML application to a production-grade reasoning-first visual intelligence system that "sees → understands → decides → speaks."

**Last Updated:** 2026-01-31  
**Version:** Reasoning-First Architecture v1.1

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
9. [Phase 0: Foundations (Reasoning Architecture)](#phase-0-foundations-reasoning-architecture)
10. [Phase 1: Visual Evidence Enhancements](#phase-1-visual-evidence-enhancements)
11. [Phase 2: Semantic Signals Refinements](#phase-2-semantic-signals-refinements)
12. [Phase 3: Interpretive Reasoner (The Brain)](#phase-3-interpretive-reasoner-the-brain)
13. [Phase 4: Interpretive Memory (Learning)](#phase-4-interpretive-memory-learning)
14. [Phase 5: Reflection Loop (Self-Validation)](#phase-5-reflection-loop-self-validation)
15. [Phase 6: Critique Voice (Expression Only)](#phase-6-critique-voice-expression-only)
16. [Phase 7: Governance (Simplified & Strong)](#phase-7-governance-simplified--strong)
17. [Technical Improvements](#technical-improvements)

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
    "perception": {...},
    "derived": {...},
    "confidence": {...},
    "errors": {},
    "semantic_anchors": {},
    "scene_understanding": {},
    "visual_evidence": {},
    "interpretive_conclusions": {},
    "reflection_report": {}
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
- New prompt format with placeholders for verified observations
- No interpretive prose passed into prompts

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
  - Temporal context (time_scale, pace, endurance, change_indicators, moment_type, temporal_direction)
  - Organic interaction (relationship, dominance, integration_level, specific_indicators)
  - Emotional substrate (temperature, pace, presence, quality, corrective_signals)
  - Contextual relationships
  - Negative evidence (no_human_presence, no_motion_detected, no_artificial_surface_uniformity)

#### 2. `synthesize_scene_understanding()`
- Universal, lightweight, rule-based heuristics
- Sparse & confidence-gated (only emit when confidence is high)
- Corrective signals softly reweight interpretation
- Uses visual evidence as primary source (ground truth from pixels)

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
- Added explicit, non-negotiable rules to prompt
- Must not contradict Scene Understanding
- Must apply all Corrective Signals (mandatory locks, not suggestions)
- Must reference organic growth/weathering if present
- Must not invent human subjects if none detected
- Must not describe as cold/sterile if warmth/organic integration present

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
- Salience determination (structural | incidental | peripheral | distributed | minimal)
- Evidence: `["green_coverage=0.42", "green_clusters=5", "spatial_distribution=vertical_surfaces", "salience=structural"]`
- Confidence: 0.95 for high coverage, 0.50 for minimal

**`detect_material_condition(image_path)`**
- Uses texture variance to detect surface roughness
- Edge degradation analysis (age indicators)
- Color uniformity analysis (paint vs organic detection)
- Condition inference: "weathered" | "pristine" | "moderate" | "degraded"
- Evidence: `["texture_variance=0.68", "edge_degradation=0.45", "color_uniformity=0.32", "condition=weathered"]`
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
- Validation and contradiction detection

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
  "no_artificial_surface_uniformity": True,
  "evidence": "YOLO: no person, CLIP: no human terms, temporal: static"
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

## Phase 0: Foundations (Reasoning Architecture)

### Objective
Establish the foundational structure for the reasoning-first architecture, including schema updates and evidence reliability hierarchy.

### Key Changes

#### 1. Schema Extensions
- Added `visual_evidence` field to canonical schema
- Added `interpretive_conclusions` field (output from interpretive reasoner)
- Added `reflection_report` field (self-validation results)
- All fields are optional and sparse (backward compatible)

#### 2. Evidence Reliability Hierarchy
- Documented hierarchy: `visual_pixels > semantic_models > technical_stats > stylistic_inference`
- Every field carries `{value, confidence, source}` metadata
- Reasoner uses reliability weights in confidence calculation

### Files Changed
- `framed/analysis/schema.py`: Added new fields to `create_empty_analysis_result()`

### Impact
- Foundation for reasoning-first architecture
- Clear evidence hierarchy
- Backward compatible

---

## Phase 1: Visual Evidence Enhancements

### Objective
Complete visual evidence extraction with negative evidence detection and enhanced validation.

### Key Changes

#### 1. Color Uniformity Analysis
- Added to `detect_material_condition()`
- Calculates hue variance in green regions (if significant green coverage)
- Distinguishes paint (high uniformity) from organic growth (low uniformity)
- Returns `color_uniformity` (0.0-1.0) in material condition dict

#### 2. Negative Evidence Detection
- **New File**: `framed/analysis/negative_evidence.py`
- **Function**: `detect_negative_evidence(analysis_result)`
- Detects:
  - `no_human_presence`: True if YOLO, CLIP, and emotion detection all show no humans
  - `no_motion_detected`: True if temporal pace is static/slow and CLIP shows no motion terms
  - `no_artificial_surface_uniformity`: True if pristine condition + high uniformity + minimal organic + low texture
- Returns explainable evidence string
- Integrated into `analyze_image()` before interpretive reasoner

#### 3. Evidence Validation Layer
- `validate_visual_evidence()` already exists (from Visual Conscience)
- Checks internal consistency
- Flags low-confidence evidence
- Detects contradictions

### Files Changed
- `framed/analysis/vision.py`: 
  - Enhanced `detect_material_condition()` with color uniformity
  - Integrated negative evidence detection
- `framed/analysis/negative_evidence.py`: NEW - Negative evidence detection module

### Impact
- Paint vs organic growth now distinguishable
- Negative evidence prevents false interpretations
- Validation ensures evidence quality

---

## Phase 2: Semantic Signals Refinements

### Objective
Enhance CLIP inventory with confidence scoring and source attribution.

### Key Changes

#### 1. CLIP Inventory Enhancement
- `get_clip_inventory()` now returns list of dicts instead of list of strings
- Each dict contains:
  ```python
  {
    "item": "ivy",
    "confidence": 0.92,
    "source": "material_condition_prompt"  # structural_prompt | material_condition_prompt | atmosphere_prompt
  }
  ```
- Tracks which prompt detected each item
- Confidence scores from CLIP softmax probabilities

#### 2. Backward Compatibility
- `generate_semantic_anchors()` handles both formats (list of dicts or list of strings)
- `interpret_scene()` extracts items from dict format for reasoner
- Legacy code continues to work

### Files Changed
- `framed/analysis/vision.py`: 
  - Enhanced `get_clip_inventory()` to return metadata
  - Updated `generate_semantic_anchors()` to handle both formats
  - Updated `analyze_image()` to handle new format

### Impact
- CLIP inventory items now have confidence scores
- Source attribution enables debugging
- Better deduplication possible

---

## Phase 3: Interpretive Reasoner (The Brain)

### Objective
Implement the interpretive reasoning layer that interprets evidence before critique generation, following the principle "Reason first, then speak."

### Key Concepts

#### 1. Plausibility Gate (Pre-LLM Filter)
- **Function**: `generate_plausible_interpretations(visual_evidence, semantic_signals)`
- Cheap filter to limit interpretation space before expensive LLM call
- Generates plausible interpretations based on visual evidence patterns
- Examples:
  - `green_coverage > 0.25` + `distribution == "vertical_surfaces"` + CLIP mentions "ivy" → `"ivy_on_structure"`
  - `color_uniformity > 0.8` + `texture_variance < 0.2` → `"painted_surface"`
- Returns list of interpretations with confidence hints
- LLM only chooses from these, not invents new categories

#### 2. Interpretive Reasoner (LLM, Silent)
- **New File**: `framed/analysis/interpret_scene.py`
- **Function**: `interpret_scene(visual_evidence, semantic_signals, technical_stats, interpretive_memory_patterns)`**
- Responsibilities:
  - Multi-hypothesis reasoning
  - Evidence weighting (respects reliability hierarchy)
  - Alternative rejection (with reasons)
  - Confidence scoring (0-1)
  - Uncertainty detection (first-class)
- Answers only 5 questions:
  1. What is most likely happening?
  2. What else could be happening?
  3. Why did you reject alternatives?
  4. How confident are you (0-1)?
  5. What emotional reading follows (one sentence max)?
- Output (STRICT JSON):
  ```json
  {
    "primary_interpretation": {
      "conclusion": "ivy on cathedral facade",
      "confidence": 0.78,
      "evidence_chain": ["green_coverage=0.42 (visual)", "CLIP_detects='ivy' (semantic)", ...],
      "reasoning": "brief explanation"
    },
    "alternatives": [
      {
        "interpretation": "green-painted facade",
        "confidence": 0.18,
        "reason_rejected": "why this is less likely"
      }
    ],
    "uncertainty": {
      "present": false,
      "confidence_threshold": 0.65,
      "requires_uncertainty_acknowledgment": false,
      "reason": "why uncertain if present"
    },
    "emotional_reading": {
      "primary": "warm_patience",
      "secondary": "quiet_endurance",
      "confidence": 0.81,
      "reasoning": "one sentence max"
    }
  }
  ```
- Key Constraints:
  - ❌ No open-ended philosophical reasoning
  - ❌ No emotional prose
  - ❌ No "What does this mean to humanity?"
  - ✅ Answer only 5 questions
  - ✅ Structured JSON output
  - ✅ Low temperature (0.3) for consistent reasoning

#### 3. Integration
- Called in `analyze_image()` after visual evidence extraction, before scene understanding
- Stores conclusions in `result["interpretive_conclusions"]`
- Stores interpretation in memory for learning
- Graceful fallback if reasoner unavailable (backward compatibility)

### Files Changed
- `framed/analysis/interpret_scene.py`: NEW - Interpretive reasoner module
- `framed/analysis/vision.py`: 
  - Integrated interpretive reasoner into `analyze_image()`
  - Handles new CLIP inventory format
- `framed/analysis/__init__.py`: Exported reasoner functions

### Impact
- Reasoning happens before language generation
- Multi-hypothesis reasoning prevents single-answer bias
- Uncertainty is first-class, not optional
- Structured output enables downstream processing

---

## Phase 4: Interpretive Memory (Learning)

### Objective
Implement pattern-based memory that learns from past interpretations, improving confidence calibration over time without retraining models.

### Key Concepts

#### 1. Pattern Memory Store
- **New File**: `framed/analysis/interpretive_memory.py`
- Stores decision snapshots (NOT images or full evidence bundles)
- Pattern signature (bucketed):
  ```python
  {
    "green_coverage_bucket": "high",  # "low" | "medium" | "high"
    "surface_type": "vertical",
    "texture": "rough",
    "clip_token": "ivy",
    "yolo_object": "building"
  }
  ```
- Decision snapshot:
  ```python
  {
    "pattern_signature": {...},
    "chosen_interpretation": "ivy_on_structure",
    "confidence": 0.78,
    "user_feedback": "felt_accurate",  # "felt_accurate" | "felt_wrong" | null
    "timestamp": "2026-01-24T10:30:00Z"
  }
  ```
- Storage: JSON file at `/data/framed/interpretive_memory/pattern_memory.json`
- Keeps last 1000 entries (prevents unbounded growth)

#### 2. Pattern Matching
- **Function**: `query_memory_patterns(pattern_signature, limit=5)`
- Simple matching: counts how many signature fields match
- Returns historical decisions with similarity scores
- Used by interpretive reasoner to inform confidence

#### 3. Pattern Statistics
- **Function**: `get_pattern_statistics(pattern_signature)`
- Returns:
  - `times_seen`: int
  - `accuracy_rate`: float (if user_feedback available)
  - `average_confidence`: float
  - `most_common_interpretation`: str
- Enables confidence calibration: "In 83% of similar cases, ivy was correct"

#### 4. Correction Learning
- **Function**: `update_pattern_confidence(pattern_signature, original_interpretation, user_feedback, correct_interpretation)`
- When user says "this felt wrong", updates pattern accuracy
- Affects future confidence calibration for similar patterns
- No retraining required

#### 5. Integration
- Called in `analyze_image()` before interpretive reasoner
- Queries memory for similar patterns
- Passes patterns to reasoner as context
- Stores interpretation after reasoner completes

### Files Changed
- `framed/analysis/interpretive_memory.py`: NEW - Interpretive memory module
- `framed/analysis/vision.py`: 
  - Integrated memory queries and storage
  - Creates pattern signatures before reasoner
  - Stores interpretations after reasoner

### Impact
- Learning without training
- Confidence calibration improves over time
- Statistics inform reasoner confidence
- Privacy-safe (no images stored)

---

## Phase 5: Reflection Loop (Self-Validation)

### Objective
Implement post-critique self-validation to catch contradictions, invented facts, ignored uncertainty, and generic language.

### Key Concepts

#### 1. Reflection Pass
- **New File**: `framed/analysis/reflection.py`
- **Function**: `reflect_on_critique(critique_text, interpretive_conclusions)`
- Checks:
  1. **Contradiction with reasoner** (0-1 score)
     - Does critique contradict interpreted conclusions?
     - Example: Reasoner says "ivy on facade", critique says "green building"
  2. **Invented facts** (0-1 score)
     - Does critique claim facts not in evidence?
     - Example: Critique says "ancient temple" but no evidence of age
  3. **Ignored uncertainty** (0-1 score)
     - If reasoner is uncertain, does critique acknowledge it?
     - Example: Reasoner confidence = 0.61, critique is overly confident
  4. **Generic language** (0-1 score)
     - Does critique use generic, non-specific language?
     - Example: "beautiful image" vs "weathered facade with ivy"

#### 2. Quality Scoring
- Each check returns 0-1 score
- Quality score = average of all scores
- If `quality_score < 0.70` → requires regeneration

#### 3. Regeneration Rules
- Maximum 1 regeneration attempt
- No infinite loops
- No analysis paralysis
- If regeneration fails, return original critique with reflection report

#### 4. Integration
- Called in `framed/routes.py` after critique generation
- Checks `interpretive_conclusions` if available
- Regenerates critique once if quality is too low
- Stores reflection report in `result["reflection_report"]`

### Files Changed
- `framed/analysis/reflection.py`: NEW - Reflection loop module
- `framed/routes.py`: 
  - Integrated reflection loop after critique generation
  - Regenerates critique if quality < 0.70

### Impact
- Bad critiques are caught and regenerated
- No infinite loops
- Reflection is fast (< 1 second)
- Quality improves automatically

---

## Phase 6: Critique Voice (Expression Only)

### Objective
Update critique generation to receive interpretive conclusions (not raw evidence) and generate critique that never contradicts reasoner conclusions.

### Key Changes

#### 1. Prompt Structure Update
- **OLD**: Prompt received raw evidence (Scene Understanding, Semantic Anchors, Verified Observations)
- **NEW**: Prompt receives interpretive conclusions FIRST, then supporting context
- New order:
  1. INTERPRETED SCENE CONCLUSIONS (AUTHORITATIVE)
  2. UNCERTAINTY FLAGS (MANDATORY ACKNOWLEDGMENT)
  3. SCENE UNDERSTANDING (FALLBACK/LEGACY - only if no interpretive conclusions)
  4. SEMANTIC ANCHORS (NAMING PERMISSION)
  5. VERIFIED OBSERVATIONS (TECHNICAL - SUPPORTING CONTEXT)
  6. Task
  7. GOVERNANCE RULES

#### 2. Interpretive Conclusions Section
- Primary interpretation with confidence and evidence chain
- Alternatives considered (with rejection reasons)
- Emotional reading from reasoner
- Uncertainty flags (if confidence < 0.65)

#### 3. Uncertainty Acknowledgment
- If `requires_uncertainty_acknowledgment=true`, critique MUST use uncertainty language
- Examples: "perhaps", "might", "suggests", "appears", "uncertain"
- Prevents false authority when confidence is low

#### 4. Governance Rules Update
- **OLD**: Vocabulary locks, resolved contradictions, visual evidence enforcement
- **NEW**: Conclusion consistency enforcement, uncertainty acknowledgment enforcement
- Rules:
  - Must not contradict Interpretive Conclusions
  - Must ground critique in primary interpretation and evidence chain
  - Must not use rejected interpretations
  - Must acknowledge uncertainty if flagged
  - Must not invent facts not in evidence chain

#### 5. Fallback Support
- If interpretive conclusions not available, falls back to Scene Understanding (backward compatibility)
- Legacy format still supported

### Files Changed
- `framed/analysis/vision.py`: 
  - Updated `generate_merged_critique()` to extract and use interpretive conclusions
  - Reorganized prompt structure
  - Updated governance rules
  - Removed vocabulary locks and resolved contradictions sections (replaced with conclusion enforcement)

### Impact
- Critique never contradicts reasoner conclusions
- Uncertainty is acknowledged when required
- Critique is grounded in interpreted conclusions
- Backward compatible with legacy format

---

## Phase 7: Governance (Simplified & Strong)

### Objective
Replace rigid vocabulary locks with intelligent conclusion enforcement, simplifying governance while maintaining strength.

### Key Changes

#### 1. Removed Vocabulary Locks
- **OLD**: Hard word bans ("FORBIDDEN: cold, sterile"), absolute emotional constraints
- **NEW**: Conclusion consistency enforcement
- Rule: "You may not contradict interpreted conclusions or suppress uncertainty."

#### 2. Removed Resolved Contradictions
- **OLD**: Explicit list of resolved contradictions, valid tension points
- **NEW**: Conclusion enforcement prevents contradictions automatically
- If reasoner rejected an interpretation, critique cannot use it

#### 3. Simplified Governance Rules
- **OLD**: Complex rules with vocabulary locks, resolved contradictions, visual evidence enforcement, negative evidence enforcement
- **NEW**: Simple but strong rules:
  - Must not contradict Interpretive Conclusions
  - Must ground critique in primary interpretation
  - Must not use rejected interpretations
  - Must acknowledge uncertainty if flagged
  - Must not invent facts

#### 4. Intelligence Replaces Enforcement
- Reasoning consistency enforcement (intelligent)
- Uncertainty acknowledgment enforcement (intelligent)
- Conclusion grounding enforcement (intelligent)
- No brittle rule explosions
- No vocabulary prisons

### Files Changed
- `framed/analysis/vision.py`: 
  - Removed `generate_vocabulary_locks()` and `generate_resolved_contradictions()` from prompt
  - Updated governance rules to use conclusion enforcement
  - Simplified prompt structure

### Impact
- No brittle rule explosions
- Intelligence replaces enforcement
- Governance is simple but strong
- Easier to maintain

---

## Technical Improvements

### 1. Error Handling & Graceful Degradation
- All analysis steps wrapped with `safe_analyze()`
- Partial results returned with error metadata
- One failing model doesn't crash entire analysis
- Interpretive reasoner is optional (backward compatibility)
- Reflection loop is optional (non-fatal)

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
- Interpretive memory stored in `/data/framed/interpretive_memory/`

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
- Interpretive reasoner logging
- Reflection loop logging

### 6. Backward Compatibility
- All new components are optional
- Legacy format still supported
- Graceful fallbacks if new modules unavailable
- No breaking changes to existing routes

---

## Summary of Evolution

FRAMED has evolved from a working ML application to a production-grade reasoning-first visual intelligence system with:

1. **Deterministic Schema**: All analysis conforms to canonical schema
2. **Evidence-Driven Prompts**: Prompts receive only observed facts
3. **Semantic Anchors**: High-confidence, multi-signal labels
4. **Scene Understanding**: Contextual synthesis before critique
5. **Visual Conscience**: Ground truth from pixels, not text inference
6. **Negative Evidence**: Tracks absence to prevent misinterpretation
7. **Salience**: Distinguishes structural from incidental growth
8. **Temporal Direction**: Distinguishes growth from decay
9. **Invariant Protection**: Explicit hierarchy prevents regressions
10. **Interpretive Reasoner**: Silent brain that reasons before speaking
11. **Plausibility Gate**: Cheap filter prevents hallucination
12. **Interpretive Memory**: Learning without training
13. **Reflection Loop**: Self-validation catches mistakes
14. **Conclusion Enforcement**: Intelligence replaces rigid rules

### Architectural Shift

**OLD Pipeline:**
```
Image → Heuristics → Rules → Prompt → Critique
```

**NEW Pipeline:**
```
Image
 ↓
Visual Evidence (pixels)
 ↓
Semantic Signals (CLIP / YOLO)
 ↓
PLAUSIBILITY GATE (cheap filter)
 ↓
INTERPRETIVE REASONER (silent brain)
 ↓
INTERPRETIVE MEMORY (learning)
 ↓
REFLECTION LOOP (self-check)
 ↓
CRITIQUE VOICE (expression only)
```

**Key Principle:** "Reason first, then speak" — separate interpretation from expression.

The system now "sees → understands → decides → speaks," with visual evidence serving as ground truth, interpretive reasoning providing probabilistic conclusions, and critique generation expressing those conclusions in a mentor's voice.

---

## Files Modified

### New Files
- `framed/analysis/schema.py`: Canonical schema definition
- `framed/analysis/interpret_scene.py`: Interpretive reasoner module
- `framed/analysis/interpretive_memory.py`: Pattern-based memory module
- `framed/analysis/reflection.py`: Reflection loop module
- `framed/analysis/negative_evidence.py`: Negative evidence detection module
- `IMPLEMENTATION_STATUS.md`: Implementation tracking document

### Modified Files
- `framed/analysis/vision.py`: 
  - Major refactoring for canonical schema
  - Added visual grounding (HSV, texture, morphology)
  - Added Scene Understanding synthesis
  - Added semantic anchors generation
  - Enhanced CLIP inventory with confidence/source
  - Integrated interpretive reasoner
  - Integrated negative evidence detection
  - Updated critique generation to use interpretive conclusions
  - Removed vocabulary locks, replaced with conclusion enforcement
- `framed/analysis/schema.py`: Added new fields (visual_evidence, interpretive_conclusions, reflection_report)
- `framed/analysis/__init__.py`: Exported new functions
- `framed/routes.py`: 
  - Updated to handle canonical schema
  - Always generate critique
  - Integrated reflection loop
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
7. **Interpretive Reasoner**: Test with diverse images, verify structured output
8. **Plausibility Gate**: Test with edge cases (green building vs ivy)
9. **Interpretive Memory**: Test pattern matching and statistics
10. **Reflection Loop**: Test with critiques that contradict reasoner
11. **Uncertainty Acknowledgment**: Test with low-confidence interpretations
12. **Conclusion Enforcement**: Verify critiques don't contradict reasoner

---

## Future Enhancements

1. **Phase 8: UX Alignment**: Optional toggles to reveal thinking
2. **Phase 9: Future**: Optional fine-tuning, learned visual models
3. **Additional Visual Features**: Shadows, light patterns, motion blur detection
4. **Material Classification**: When confidence is high, classify materials (stone, concrete, wood)
5. **Learned Models**: If deterministic methods fail, add learned models with explainability
6. **Portfolio Mode**: Use canonical schema for portfolio-level analysis
7. **ECHO Enhancement**: Use Scene Understanding and visual evidence in ECHO memory
8. **User Feedback Integration**: Connect user corrections to interpretive memory

---

### Recovery & Stress Test (2026-01-29 to 2026-01-31)

**OneDrive Recovery & Forensics:** Local files lost/overwritten; forensics classified `framed/analysis/` into Bucket A (aligned), C (legacy fallback). Reference: FRAMED_CONSTITUTION, FRAMED_INTELLIGENCE_MASTER_PLAN.

**Git LFS:** `stress_test_master/images.zip` (~9.1 GB) migrated to LFS; `*.zip` in `.gitattributes`.

**Vision.py:** Explicit comment—`interpret_scene` and `interpretive_memory` are legacy fallbacks only when `FRAMED_USE_INTELLIGENCE_CORE` is enabled.

**Dataset (`scripts/dataset_download_and_categorize.py`):** Places365 val_256 → architecture, interiors, street; Open Images V7 → portraits, mixed; Unsplash Lite → nature, ambiguous, artistic. SHA256 zero-overlap verified.

**Stress test (run_2026_01_29_020225):** 2,000 images, 19h 27m 40s, 0 failures. Output: `framed/tests/test_runs/run_YYYY_MM_DD_HHMMSS/`.

**Model A (Reasoning):** gpt-5.2 — reasoning.effort: medium, text.verbosity: low, temperature omitted.  
**Model B (Expression):** gpt-5-mini — text.verbosity: medium, temp 0.7.

**Other changes:** `disable_cache` in `analyze_image()`, `--disable_cache` flag, expression model log, `.env` + dotenv, `framed/tests/` (datasets.py, test_intelligence_pipeline, metrics, reporting), `scripts/dataset_download_and_categorize.py`, 17 redundant docs deleted.

---

*Last Updated: 2026-01-31*  
*Version: Reasoning-First Architecture v1.1*
