# FRAMED — COMPLETE FUTURE STEPS & IMPLEMENTATION PLAN

**Last Updated:** 2026-01-29  
**Status:** Calibration complete (Steps 8.2–8.4, Phase 9). HITL protocol live. Dataset v2 (36 images) ready. Phase 10–11 next.  
**Goal:** Transform FRAMED into a reasoning-first visual intelligence system that sees → understands → decides → speaks

---

## 🔴 CORE ARCHITECTURAL SHIFT (NON-NEGOTIABLE)

### Old Pipeline (Problematic)
```
Image → Heuristics → Rules → Prompt → Critique
```

**Problems:**
- Hard-coded rules break for edge cases (green building vs ivy)
- Vocabulary locks are too rigid
- No learning or memory
- No self-validation
- Intelligence downstream (in critique), not upstream

### New Pipeline (Final)
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

**Key Principle:** **"Reason first, then speak"** — separate interpretation from expression.

---

## 📋 COMPLETE IMPLEMENTATION PLAN

### PHASE 0 — FOUNDATIONS (STRUCTURE & SAFETY)

#### 0.1 Canonical Data Contracts
**Files:** `schema.py`

**Add / Finalize:**
- `visual_evidence` (structured visual features)
- `interpretive_conclusions` (reasoner output)
- `interpretive_confidence` (0-1 scores)
- `uncertainty_flags` (boolean + reasons)
- `reflection_report` (self-validation results)

**Done when:**
- Every stage outputs structured JSON
- No stage reads raw image except visual extraction
- All data flows through canonical schema

#### 0.2 Evidence Reliability Hierarchy
**Hard Invariant:**
```
visual_pixels > semantic_models > technical_stats > stylistic_inference
```

**Enforce:**
- Every downstream consumer must know evidence source + confidence
- Evidence weighting in reasoner respects this hierarchy

**Done when:**
- Every field carries `{value, confidence, source}`
- Reasoner uses reliability weights in confidence calculation

---

### PHASE 1 — VISUAL EVIDENCE (GROUND TRUTH)

#### 1.1 Visual Feature Extraction (CV, not ML)
**Files:** `vision.py`

**Status:** ✅ Partially implemented (needs enhancement)

**Current Implementation:**
- `detect_organic_growth()` - HSV color segmentation
- `detect_material_condition()` - Texture variance
- `detect_organic_integration()` - Morphological overlap

**Enhancements Needed:**
- Better spatial distribution analysis (vertical vs ground vs background)
- Edge degradation detection (age indicators)
- Color uniformity analysis (paint vs organic)
- Texture variance refinement (weathered vs pristine)

**Output Format:**
```json
{
  "organic_growth": {
    "coverage": 0.42,
    "salience": "structural",
    "distribution": "vertical_surfaces",
    "clusters": 3,
    "confidence": 0.95,
    "source": "visual_analysis"
  },
  "material_condition": {
    "surface": "weathered",
    "roughness": 0.68,
    "edge_degradation": 0.45,
    "confidence": 0.90,
    "source": "visual_analysis"
  },
  "organic_integration": {
    "relationship": "reclamation",
    "overlap_ratio": 0.78,
    "integration_level": "high",
    "confidence": 0.88,
    "source": "visual_analysis"
  }
}
```

**Done when:**
- Ivy ≠ grass ≠ green paint are distinguishable probabilistically
- Evidence is explainable from pixels
- All visual features have confidence scores

#### 1.2 Negative Evidence Detection
**Purpose:** Prevent false interpretations

**Examples:**
- No humans detected (YOLO + CLIP)
- No motion indicators (temporal analysis)
- No reflective modern materials (visual analysis)
- No artificial uniformity (color + texture analysis)

**Output:**
```json
{
  "negative_evidence": {
    "no_human_presence": true,
    "no_motion_detected": true,
    "no_artificial_surface_uniformity": false,
    "evidence": "YOLO: no person, CLIP: no human terms, temporal: static"
  }
}
```

**Done when:**
- Absence informs reasoning (stillness ≠ alienation)
- Negative evidence prevents false positive interpretations

#### 1.3 Evidence Validation Layer
**Functions:**
- `validate_visual_evidence()` - Check internal consistency
- `detect_internal_contradictions()` - Flag conflicts

**Done when:**
- Low-confidence evidence is flagged
- Contradictory signals are surfaced explicitly
- Validation warnings are logged

---

### PHASE 2 — SEMANTIC SIGNALS (SUPPORTING, NOT LEADING)

#### 2.1 CLIP Inventory (Expanded)
**Status:** ✅ Implemented (needs refinement)

**Current Implementation:**
- Multi-prompt inventory (Structural, Material & Condition, Atmosphere)
- Returns nouns/descriptors only

**Refinements Needed:**
- Better deduplication
- Confidence scoring per item
- Source attribution (which prompt detected it)

**Output:**
```json
{
  "clip_inventory": [
    {"item": "ivy", "confidence": 0.92, "source": "material_condition_prompt"},
    {"item": "weathered stone", "confidence": 0.88, "source": "material_condition_prompt"},
    {"item": "cathedral", "confidence": 0.85, "source": "structural_prompt"}
  ]
}
```

**Done when:**
- CLIP supports but never overrides visual evidence
- Inventory items have confidence scores
- Source is tracked for debugging

#### 2.2 YOLO Objects (Contextual Only)
**Use for:**
- Scale (monumental vs intimate)
- Presence (humans, objects)
- Type hints (building, structure, etc.)

**Never use for:**
- Emotional inference
- Material condition inference
- Organic growth detection

**Done when:**
- YOLO is used only for contextual information
- Not used for interpretation

---

### PHASE 3 — INTERPRETIVE REASONER (THE BRAIN)

⚠️ **THIS IS THE BIG SHIFT**

#### 3.1 Plausibility Gate (Pre-LLM)
**Purpose:** Cheap filter to limit interpretation space before expensive LLM call

**Implementation:**
```python
def generate_plausible_interpretations(visual_evidence, semantic_signals):
    """
    Cheap logic that generates plausible interpretations.
    LLM will only choose from these, not invent new categories.
    """
    plausible = []
    
    # Organic growth interpretations
    if visual_evidence["organic_growth"]["coverage"] > 0.25:
        if visual_evidence["organic_growth"]["distribution"] == "vertical_surfaces":
            if "ivy" in semantic_signals["clip_inventory"]:
                plausible.append({
                    "interpretation": "ivy_on_structure",
                    "confidence_hint": 0.75
                })
        elif visual_evidence["organic_growth"]["distribution"] == "foreground":
            plausible.append({
                "interpretation": "foreground_vegetation",
                "confidence_hint": 0.70
            })
    
    # Painted surface interpretations
    if visual_evidence["material_condition"]["color_uniformity"] > 0.8:
        if visual_evidence["material_condition"]["texture_variance"] < 0.2:
            plausible.append({
                "interpretation": "painted_surface",
                "confidence_hint": 0.65
            })
    
    # ... more patterns
    
    return plausible
```

**Done when:**
- LLM only chooses from plausible interpretations
- No hallucinated categories
- No overthinking

#### 3.2 Interpretive Reasoner (LLM, Silent)
**New Module:** `interpret_scene.py`

**Responsibilities:**
- Multi-hypothesis reasoning
- Evidence weighting (respects reliability hierarchy)
- Alternative rejection (with reasons)
- Confidence scoring (0-1)
- Uncertainty detection (first-class)

**Input:**
- Visual evidence (structured)
- Semantic signals (CLIP + YOLO)
- Technical stats (brightness, contrast, etc.)
- Plausible interpretations (from gate)
- Interpretive memory patterns (if available)

**Output (STRICT JSON):**
```json
{
  "primary_interpretation": {
    "conclusion": "ivy on cathedral facade",
    "confidence": 0.78,
    "evidence_chain": [
      "green_coverage=0.42 (visual, confidence=0.95)",
      "spatial_distribution=vertical_surfaces (visual, confidence=0.92)",
      "CLIP_detects='ivy' (semantic, confidence=0.88)",
      "YOLO_detects='building' (semantic, confidence=0.85)",
      "spatial_overlap=0.78 (fusion, confidence=0.90)"
    ],
    "reasoning": "Multiple signals converge: green pixels align with vertical surfaces, CLIP confirms 'ivy', YOLO confirms 'building', high spatial overlap suggests organic growth on structure"
  },
  "alternatives": [
    {
      "interpretation": "green-painted facade",
      "confidence": 0.18,
      "reason_rejected": "Irregular boundaries, high texture variance, CLIP detects 'ivy' not 'paint', vertical climbing pattern suggests organic growth"
    },
    {
      "interpretation": "decorative cladding",
      "confidence": 0.04,
      "reason_rejected": "No uniform edges, texture suggests organic not manufactured, CLIP strongly favors vegetation"
    }
  ],
  "uncertainty": {
    "present": false,
    "confidence_threshold": 0.65,
    "requires_uncertainty_acknowledgment": false
  },
  "emotional_reading": {
    "primary": "warm_patience",
    "secondary": "quiet_endurance",
    "confidence": 0.81,
    "reasoning": "Organic growth + weathering suggests time-integrated calm, not sterility"
  }
}
```

**Key Constraints:**
- ❌ No open-ended philosophical reasoning
- ❌ No emotional prose
- ❌ No "What does this mean to humanity?"
- ✅ Answer only 5 questions:
  1. What is most likely happening?
  2. What else could be happening?
  3. Why did you reject alternatives?
  4. How confident are you (0-1)?
  5. What emotional reading follows (one sentence max)?

**Done when:**
- Reasoner never writes prose
- Always considers alternatives
- Can say "I'm not fully sure"
- Output is structured JSON
- Latency is low (< 3 seconds)

---

### PHASE 4 — INTERPRETIVE MEMORY (LEARNING WITHOUT TRAINING)

#### 4.1 Pattern Memory Store
**New File:** `interpretive_memory.py`

**Store (Decision Snapshots Only):**
```json
{
  "pattern_signature": {
    "green_coverage_bucket": "high",  // "low" | "medium" | "high"
    "surface_type": "vertical",
    "texture": "rough",
    "clip_token": "ivy",
    "yolo_object": "building"
  },
  "chosen_interpretation": "ivy_on_structure",
  "confidence": 0.78,
  "user_feedback": "felt_accurate",  // "felt_accurate" | "felt_wrong" | null
  "timestamp": "2026-01-24T10:30:00Z"
}
```

**Do NOT Store:**
- ❌ Raw images
- ❌ Full evidence bundles
- ❌ Massive JSON blobs
- ❌ Image embeddings

**Benefits:**
- Learning without training
- Statistics: "In 83% of similar cases, ivy was correct"
- Confidence calibration over time
- No privacy or storage issues

**Done when:**
- FRAMED improves confidence calibration over time
- Pattern matching retrieves historical decisions
- Statistics inform reasoner confidence

#### 4.2 Correction Learning
**Mechanism:**
- User feedback updates pattern confidence
- No retraining required
- Pattern accuracy rates adjust dynamically

**Example:**
```python
# When user says "this felt wrong"
update_pattern_confidence(
    pattern_signature=pattern,
    chosen_interpretation="green_painted_facade",
    user_feedback="felt_wrong",
    correct_interpretation="ivy_on_structure"
)

# Pattern accuracy decreases
# Future similar patterns will have lower confidence
# System becomes more cautious
```

**Done when:**
- System becomes more confident where it's often right
- More cautious where it's often wrong
- User corrections improve future interpretations

---

### PHASE 5 — REFLECTION LOOP (SELF-VALIDATION)

#### 5.1 Reflection Pass
**After critique generation, check:**

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

**Output:**
```json
{
  "reflection": {
    "contradiction_score": 0.0,  // 0 = no contradiction, 1 = major contradiction
    "invented_facts_score": 0.0,
    "uncertainty_acknowledged": true,
    "generic_language_score": 0.1,
    "quality_score": 0.90,  // Average of all scores (inverted)
    "requires_regeneration": false  // If quality_score < threshold
  }
}
```

**Regeneration Rules:**
- If `quality_score < 0.70` → regenerate once
- No infinite loops
- No analysis paralysis
- Maximum 1 regeneration attempt

**Done when:**
- Bad critiques are regenerated once
- No infinite loops
- Reflection is fast (< 1 second)

---

### PHASE 6 — CRITIQUE VOICE (EXPRESSION ONLY)

#### 6.1 Critique Generator
**Receives:**
- Interpretive conclusions ONLY (not raw evidence)
- Mentor persona (Balanced, Radical, etc.)
- Uncertainty flags (if `requires_uncertainty_acknowledgment = true`)

**Forbidden:**
- ❌ Raw evidence access
- ❌ Independent interpretation
- ❌ Contradicting reasoner conclusions

**Rules:**
- Must ground in conclusions
- Must acknowledge uncertainty if flagged
- Must not invent facts
- Must use mentor voice

**Prompt Structure:**
```
You are FRAMED — the Artistic Mentor.

INTERPRETED SCENE CONCLUSIONS (AUTHORITATIVE):
{format_conclusions(interpretive_conclusions)}

UNCERTAINTY FLAGS:
{format_uncertainty_flags(uncertainty_flags)}

MENTOR PERSONA:
{mentor_mode}

Your task:
1. Interpret what these conclusions reveal about the photographer's intent.
2. Identify where the image is honest — and where it is safe.
3. Speak to the photograph as a serious work, not a draft.
4. Surface a tension, contradiction, or unanswered question.
5. End with a provocation that suggests evolution — not instruction.

RULES:
- You must ground your critique in the interpreted conclusions.
- You must not contradict the reasoner's conclusions.
- If uncertainty is flagged, you must acknowledge ambiguity.
- You must not invent facts not in the conclusions.
- Do NOT describe the image literally.
- Do NOT list tips.
- Do NOT sound instructional.
- Do NOT flatter.

Generate critique.
```

**Done when:**
- Voice never contradicts brain
- Uncertainty is acknowledged when required
- Critique is grounded in interpreted conclusions

---

### PHASE 7 — GOVERNANCE (SIMPLIFIED & STRONG)

#### 7.1 Replace Vocabulary Locks
**Old Approach (Rigid):**
- ❌ Hard word bans ("FORBIDDEN: cold, sterile")
- ❌ Absolute emotional constraints
- ❌ Vocabulary prisons

**New Approach (Intelligent):**
- ✅ Reasoning consistency enforcement
- ✅ Uncertainty acknowledgment enforcement
- ✅ Conclusion grounding enforcement

**Rule:**
> "You may not contradict interpreted conclusions or suppress uncertainty."

**Done when:**
- No brittle rule explosions
- Intelligence replaces enforcement
- Governance is simple but strong

---

### PHASE 8 — UX ALIGNMENT (OPTIONAL BUT IMPORTANT)

#### 8.1 Reveal Thinking (Selective)
**Optional Toggles:**
- "How FRAMED interpreted this" (show reasoner output)
- "What alternatives were considered" (show rejected interpretations)
- "Where uncertainty exists" (highlight uncertain areas)

**Benefits:**
- Trust is built, not demanded
- Transparency without overwhelming
- Educational value

**Done when:**
- Users can optionally see reasoning
- UI is clean and non-technical
- Trust is improved

---

### PHASE 9 — FUTURE (NOT NOW)

#### 🚫 Do NOT Do Yet:
- Fine-tuning
- End-to-end CNN emotion models
- Custom vision transformers
- Learned material classifiers

#### ✅ Later (When Core is Stable):
- Learned material classifiers (faster than CLIP)
- Faster distilled reasoner (lower latency)
- Memory-guided fine-tuning (improve with experience)
- Custom vision models (if needed)

---

## 🔑 KEY IMPROVEMENTS (ADDED TO ORIGINAL PLAN)

### Improvement #1: Make the Reasoner Narrower, Not Smarter
**Constraint:** The reasoner should answer only 5 questions, always:
1. What is most likely happening?
2. What else could be happening?
3. Why did you reject alternatives?
4. How confident are you (0-1)?
5. What emotional reading follows (one sentence max)?

**Nothing else.**
- Keeps latency low
- Outputs structured
- Mistakes debuggable
- Treat like a scientific observer, not an artist

### Improvement #2: Add a "Plausibility Gate" Before the LLM
**Purpose:** Cheap filter to limit interpretation space

**Before calling reasoning LLM:**
```python
plausible_interpretations = generate_plausible_interpretations(
    visual_evidence, semantic_signals
)
```

**Then tell LLM:**
> "Only choose between these interpretations. You may reject all but one, but do not invent new categories."

**Prevents:**
- Hallucinated interpretations
- Overthinking
- Creative drift

**Think of it as:** Reasoning with guardrails, not rules.

### Improvement #3: Interpretive Memory Should Store Decisions, Not Images
**Important Correction:**

**❌ Don't Store:**
- Raw images
- Full evidence bundles
- Massive JSON blobs

**✅ Store Decision Snapshots:**
```json
{
  "pattern_signature": {
    "green_coverage_bucket": "high",
    "surface_type": "vertical",
    "texture": "rough",
    "clip_token": "ivy"
  },
  "chosen_interpretation": "ivy_on_structure",
  "confidence": 0.78,
  "user_feedback": "felt_accurate",
  "timestamp": "2026-01-24"
}
```

**Benefits:**
- Learning
- Statistics
- Confidence calibration
- No privacy or storage issues

**Over time:**
> "In 83% of similar cases, ivy was correct."

That's experience, not training.

### Improvement #4: Uncertainty Must Be First-Class, Not Optional
**Hard Rule:**
```json
{
  "confidence": 0.61,
  "requires_uncertainty": true  // If confidence < 0.65
}
```

**Critique Prompt Rule:**
> "If `requires_uncertainty=true`, you must explicitly acknowledge ambiguity."

**Prevents:** False authority

### Improvement #5: Reflection Loop Should Be Cheap and Brutal
**Keep Only 4 Checks:**
1. Did the critique contradict the reasoner? (0-1)
2. Did it invent facts? (0-1)
3. Did it ignore uncertainty? (0-1)
4. Did it drift generic? (0-1)

**Scoring:**
- Score each 0-1
- Total = average of all scores
- If total < threshold (e.g., 0.70) → regenerate once
- No infinite loops
- No analysis paralysis

---

## 📅 IMPLEMENTATION ORDER

### Phase 1 (Now) — Core Reasoning
1. ✅ Interpretive Reasoner (narrow, structured)
2. ✅ Plausibility Gate
3. ✅ Confidence + Uncertainty Enforcement

### Phase 2 — Learning
1. ✅ Interpretive Memory (decision-level storage)
2. ✅ Confidence Calibration
3. ✅ User Correction Hooks

### Phase 3 — Self-Validation
1. ✅ Reflection Loop
2. ✅ Quality Scoring
3. ✅ Regeneration on Failure

### Later — Optional Enhancements
1. Optional fine-tuning
2. Optional learned visual models
3. Custom vision transformers (if needed)

---

## 🎯 SUCCESS CRITERIA

### Technical
- [ ] Reasoner outputs structured JSON in < 3 seconds
- [ ] Plausibility gate filters interpretations correctly
- [ ] Memory system improves confidence calibration over time
- [ ] Reflection loop catches contradictions
- [ ] Critique never contradicts reasoner conclusions
- [ ] Uncertainty is acknowledged when required

### User Experience
- [ ] Critiques are specific, not generic
- [ ] Critiques acknowledge uncertainty when present
- [ ] Critiques are grounded in visual evidence
- [ ] System learns from user feedback
- [ ] Trust is built through transparency

### Architecture
- [ ] Clear separation: reasoning vs expression
- [ ] Evidence reliability hierarchy enforced
- [ ] No hard-coded rules (intelligence replaces enforcement)
- [ ] Universal: works for any image type
- [ ] Explainable: every conclusion has evidence chain

---

## 📝 NOTES

### Why This Architecture Works
1. **Intelligence Upstream:** Reasoning happens before language generation
2. **Learning:** Memory system improves over time without retraining
3. **Uncertainty Handling:** Acknowledges when uncertain, prevents false authority
4. **Multi-Hypothesis:** Considers alternatives, not just one answer
5. **Self-Validation:** Reflection loop catches contradictions
6. **Simpler Governance:** Enforces conclusions, not raw evidence
7. **Universal:** Works for any image type (green building, blue sky, etc.)

### What This Solves
- **Green Building Problem:** Reasoner sees green + uniform texture + no CLIP('ivy') → "painted surface", not "organic growth"
- **Blue Sky Problem:** Reasoner sees blue + spatial distribution + CLIP('sky') → "sky", not "water"
- **Uncertainty Problem:** If reasoner is uncertain, critique acknowledges it
- **Generic Language Problem:** Reflection loop catches generic critiques
- **Contradiction Problem:** Reflection loop catches contradictions with reasoner

### What Stays from Current System
- ✅ Visual feature extraction (enhanced)
- ✅ Evidence confidence scoring (enhanced)
- ✅ Scene Understanding schema (adapted)
- ✅ Negative evidence tracking (enhanced)
- ✅ CLIP inventory (enhanced)
- ✅ YOLO objects (contextual only)

### What Changes
- ❌ Hard vocabulary locks → ✅ Conclusion consistency enforcement
- ❌ Absolute emotional constraints → ✅ Interpreted emotional reading
- ❌ Rule-based interpretation → ✅ LLM reasoning
- ❌ No memory → ✅ Interpretive memory
- ❌ No self-validation → ✅ Reflection loop

---

## 🔄 ITERATION PLAN

### Iteration 1: Core Reasoner
- Implement plausibility gate
- Implement interpretive reasoner (narrow, structured)
- Integrate with existing visual evidence extraction
- Test on diverse images

### Iteration 2: Memory & Learning
- Implement interpretive memory store
- Implement pattern matching
- Implement correction learning
- Test confidence calibration

### Iteration 3: Reflection & Governance
- Implement reflection loop
- Simplify governance (conclusion enforcement)
- Update critique generation
- Test end-to-end

### Iteration 4: Polish & UX
- Add optional "reveal thinking" UI
- Improve error handling
- Optimize latency
- User testing

---

**END OF DOCUMENT**
