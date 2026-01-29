# FRAMED Intelligence Implementation Instructions

## 🎯 Overview

This document provides step-by-step instructions for implementing FRAMED's intelligence architecture. The implementation is designed to be modular, with placeholders for LLM models that can be replaced when models are chosen.

---

## 📋 Prerequisites

### Completed:
- ✅ Model abstraction layer (`framed/analysis/llm_provider.py`) - Placeholder implementation
- ✅ Architecture documentation (`FRAMED_INTELLIGENCE_MASTER_PLAN.md`)
- ✅ Two-model architecture (`TWO_MODEL_ARCHITECTURE.md`)

### Required:
- Python 3.11+
- Existing FRAMED codebase
- Visual analysis pipeline (YOLO, CLIP, OpenCV) - already exists

---

## 🚀 Implementation Phases

### Phase 0: Model Abstraction Layer ✅ COMPLETE

**Status:** Placeholder implementation is complete and ready to use.

**What Was Done:**
- Created `LLMProvider` abstract base class
- Implemented `PlaceholderProvider` for development
- Added `call_model_a()` and `call_model_b()` functions
- Implemented retry logic and fallback mechanisms
- Made models switchable via environment variables

**Usage:**
```python
from framed.analysis.llm_provider import call_model_a, call_model_b

# Model A (Reasoning) - returns structured JSON
result = call_model_a(
    prompt="Reason about this image...",
    system_prompt="You are FRAMED's intelligence core...",
    response_format={"type": "json_object"}
)

# Model B (Expression) - returns prose
result = call_model_b(
    prompt="Transform this reasoning into a poetic critique...",
    system_prompt="You are FRAMED's mentor voice..."
)
```

**Next Step:** Proceed to Phase 1. Placeholders will be replaced in Phase 6.

---

### Phase 1: Intelligence Core (Foundation)

**Goal:** Build the 7-layer reasoning engine (Model A)

**File to Create:** `framed/analysis/intelligence_core.py`

**Implementation Steps:**

#### Step 1.1: Create Base Structure

```python
"""
FRAMED Intelligence Core

The 7-layer reasoning engine that gives FRAMED its brain.
Each layer reasons about images, not just matches patterns.
"""

from framed.analysis.llm_provider import call_model_a
from typing import Dict, Any, Optional

def framed_intelligence(
    visual_evidence: Dict[str, Any],
    temporal_memory: Optional[Dict[str, Any]] = None,
    user_history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main intelligence core function.
    
    Args:
        visual_evidence: Visual analysis results (YOLO, CLIP, OpenCV)
        temporal_memory: Past interpretations and evolution
        user_history: User trajectory and patterns
    
    Returns:
        Structured intelligence output with all 7 layers
    """
    # Layer 1: Certain Recognition
    recognition = reason_about_recognition(visual_evidence)
    
    # Layer 2: Meta-Cognition
    meta_cognition = reason_about_thinking(recognition, temporal_memory)
    
    # Layer 3: Temporal Consciousness
    temporal = reason_about_evolution(meta_cognition, temporal_memory)
    
    # Layer 4: Emotional Resonance
    emotion = reason_about_feeling(meta_cognition, temporal)
    
    # Layer 5: Continuity of Self
    continuity = reason_about_trajectory(emotion, user_history)
    
    # Layer 6: Mentor Voice (Reasoning)
    mentor = reason_about_mentorship(continuity, user_history)
    
    # Layer 7: Self-Critique
    self_critique = reason_about_past_errors(mentor, temporal_memory)
    
    return {
        "recognition": recognition,
        "meta_cognition": meta_cognition,
        "temporal": temporal,
        "emotion": emotion,
        "continuity": continuity,
        "mentor": mentor,
        "self_critique": self_critique
    }
```

#### Step 1.2: Implement Layer 1 (Certain Recognition)

```python
def reason_about_recognition(visual_evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Layer 1: Certain Recognition
    
    LLM reasons about what it sees, not just matches patterns.
    """
    prompt = f"""
You are FRAMED's recognition engine. You see images with certainty.

VISUAL EVIDENCE:
{format_visual_evidence(visual_evidence)}

REASONING TASK:
What are you seeing? Be certain, not tentative. Provide evidence.

OUTPUT FORMAT (JSON):
{{
    "what_i_see": "I see weathered stone with ivy reclaiming the structure",
    "evidence": ["green_coverage=0.42", "condition=weathered", "integration=reclamation"],
    "confidence": 0.92
}}
"""
    
    result = call_model_a(
        prompt=prompt,
        system_prompt="You are FRAMED's recognition engine. You see images with certainty.",
        response_format={"type": "json_object"}
    )
    
    if result.get("error"):
        return {"what_i_see": "", "evidence": [], "confidence": 0.0, "error": result["error"]}
    
    return json.loads(result["content"])
```

#### Step 1.3: Implement Remaining Layers

Follow the same pattern for Layers 2-7:
- Build reasoning prompt
- Call `call_model_a()` with JSON response format
- Parse and return structured output

**Key Points:**
- All layers use `call_model_a()` (Model A - Reasoning)
- All prompts request structured JSON output
- All reasoning is internal (not exposed to user)
- Evidence chains are tracked internally

**Dependencies:**
- `framed/analysis/llm_provider.py` (Phase 0)
- Visual evidence from existing analysis pipeline

**Testing:**
- Test each layer independently
- Verify JSON output structure
- Check error handling

---

### Phase 2: Temporal Memory System

**Goal:** Build memory that learns and evolves

**File to Create:** `framed/analysis/temporal_memory.py`

**Implementation Steps:**

#### Step 2.1: Pattern Signature Creation

```python
import hashlib
import json

def create_pattern_signature(visual_evidence: Dict[str, Any], semantic_signals: Dict[str, Any]) -> str:
    """
    Create a hashable signature from evidence.
    
    Used to find similar past interpretations.
    """
    # Normalize evidence
    normalized = {
        "visual": {
            "green_coverage": round(visual_evidence.get("organic_growth", {}).get("green_coverage", 0), 2),
            "condition": visual_evidence.get("material_condition", {}).get("condition", "unknown"),
            # ... other key visual features
        },
        "semantic": {
            "objects": sorted(semantic_signals.get("objects", [])),
            "tags": sorted(semantic_signals.get("tags", [])),
            # ... other key semantic features
        }
    }
    
    # Create hash
    signature_str = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(signature_str.encode()).hexdigest()[:16]
```

#### Step 2.2: Memory Storage

```python
import os
from datetime import datetime

TEMPORAL_MEMORY_PATH = os.path.join(
    os.getenv("FRAMED_DATA_DIR", "/tmp/framed"),
    "temporal_memory.json"
)

def store_interpretation(
    signature: str,
    interpretation: Dict[str, Any],
    confidence: float,
    user_feedback: Optional[Dict[str, Any]] = None,
):
    """
    Store interpretation in temporal memory.
    
    Tracks evolution over time.
    """
    memory = load_temporal_memory()
    
    if signature not in memory["patterns"]:
        memory["patterns"][signature] = {
            "first_seen": datetime.now().isoformat(),
            "interpretations": []
        }
    
    entry = {
        "date": datetime.now().isoformat(),
        "interpretation": interpretation,
        "confidence": confidence,
    }
    
    if user_feedback:
        entry["user_feedback"] = user_feedback
    
    memory["patterns"][signature]["interpretations"].append(entry)
    
    # Keep last 1000 entries per pattern
    if len(memory["patterns"][signature]["interpretations"]) > 1000:
        memory["patterns"][signature]["interpretations"] = \
            memory["patterns"][signature]["interpretations"][-1000:]
    
    save_temporal_memory(memory)
```

#### Step 2.3: Memory Query

```python
def query_memory_patterns(signature: str, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Find similar past interpretations.
    
    Returns evolution history for similar patterns.
    """
    memory = load_temporal_memory()
    
    # Find similar patterns (simplified - can be enhanced with similarity scoring)
    similar_patterns = []
    for pattern_sig, pattern_data in memory["patterns"].items():
        # Calculate similarity (simplified)
        similarity = calculate_similarity(signature, pattern_sig)
        if similarity >= similarity_threshold:
            similar_patterns.append(pattern_data)
    
    return similar_patterns
```

**Key Points:**
- Memory stores reasoning, not just results
- Tracks evolution over time
- Enables temporal consciousness reasoning

**Dependencies:**
- `framed/analysis/intelligence_core.py` (Phase 1)

---

### Phase 3: Expression Layer (Model B)

**Goal:** Transform reasoning into poetic critique

**File to Create:** `framed/analysis/expression_layer.py`

**Implementation Steps:**

#### Step 3.1: Expression Generation

```python
from framed.analysis.llm_provider import call_model_b

def generate_poetic_critique(
    intelligence_output: Dict[str, Any],
    mentor_mode: str = "Balanced Mentor",
) -> str:
    """
    Transform structured intelligence output into poetic critique.
    
    Uses Model B (Expression) - warm, articulate, human.
    """
    prompt = f"""
You are FRAMED's mentor voice. You speak with wisdom, warmth, and poetry.

INTELLIGENCE OUTPUT (from reasoning core):
{format_intelligence_output(intelligence_output)}

MENTOR MODE: {mentor_mode}

MENTOR INSTRUCTION:
Transform this reasoning into a poetic critique. Speak as a mentor, not a tool.

REQUIREMENTS:
- Certainty embodied, not announced ("I see weathered stone" not "I think I see...")
- Poetic voice (gallery placard, not technical report)
- Mentor hierarchy:
  - Observations (frequent): "You've resolved something here..."
  - Questions (strategic): "You keep circling this theme — why?"
  - Challenges (rare): "This contradicts your trajectory — intentional?"
- Evolutionary self-correction: "I used to see this as X. Looking at your recent work, I now see Y."
- Warm, human, articulate

OUTPUT:
Poetic critique, not JSON. Human voice, not data dump.
"""
    
    result = call_model_b(
        prompt=prompt,
        system_prompt=f"You are FRAMED's {mentor_mode}. You speak with wisdom, warmth, and poetry.",
        max_tokens=2000,
        temperature=0.8,  # Slightly higher for creativity
    )
    
    if result.get("error"):
        return f"[Error generating critique: {result['error']}]"
    
    return result["content"]
```

**Key Points:**
- Uses `call_model_b()` (Model B - Expression)
- Takes structured intelligence output (JSON)
- Returns poetic critique (prose)
- Embodies certainty, not announces it

**Dependencies:**
- `framed/analysis/intelligence_core.py` (Phase 1)
- `framed/analysis/llm_provider.py` (Phase 0)

---

### Phase 4: Learning System

**Goal:** Implicit learning with explicit calibration

**File to Create:** `framed/analysis/learning_system.py`

**Implementation Steps:**

#### Step 4.1: Pattern Recognition

```python
def recognize_patterns(
    analysis_history: List[Dict[str, Any]],
    user_feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Identify patterns in user's work and FRAMED's interpretations.
    """
    # Analyze user themes
    user_themes = extract_themes(analysis_history)
    
    # Analyze FRAMED's interpretation patterns
    interpretation_patterns = extract_interpretation_patterns(analysis_history)
    
    # Identify growth edges
    growth_edges = identify_growth_edges(user_themes, interpretation_patterns)
    
    return {
        "user_themes": user_themes,
        "interpretation_patterns": interpretation_patterns,
        "growth_edges": growth_edges,
    }
```

#### Step 4.2: Implicit Learning

```python
def learn_implicitly(
    analysis_result: Dict[str, Any],
    user_history: Dict[str, Any],
):
    """
    Learn from observation - no explicit feedback needed.
    """
    # Track recurring themes
    update_theme_tracking(analysis_result, user_history)
    
    # Notice what user ignores vs pursues
    update_attention_tracking(analysis_result, user_history)
    
    # Observe which provocations lead to change
    update_provocation_tracking(analysis_result, user_history)
    
    # Update pattern memory
    update_pattern_memory(analysis_result, user_history)
```

#### Step 4.3: Explicit Calibration

```python
def calibrate_explicitly(
    user_feedback: Dict[str, Any],
    interpretation: Dict[str, Any],
):
    """
    Calibrate from explicit feedback - rare but powerful.
    """
    if user_feedback.get("missed_the_point"):
        # Recalibrate confidence
        adjust_confidence(interpretation, -0.1)
        # Re-weight interpretation patterns
        reweight_interpretation_patterns(interpretation, user_feedback)
    
    elif user_feedback.get("felt_exactly_right"):
        # Reinforce pattern
        reinforce_pattern(interpretation)
        # Increase confidence
        adjust_confidence(interpretation, +0.05)
```

**Key Points:**
- Learning happens through observation
- Explicit feedback is rare but powerful
- Recalibrates confidence, not content

**Dependencies:**
- `framed/analysis/temporal_memory.py` (Phase 2)
- `framed/analysis/intelligence_core.py` (Phase 1)

---

### Phase 5: Integration into Pipeline

**Goal:** Replace rule-based systems with intelligence calls

**Files to Modify:**
- `framed/analysis/vision.py` - Integrate intelligence core
- `framed/routes.py` - Use expression layer for critique

**Implementation Steps:**

#### Step 5.1: Integrate Intelligence Core

In `framed/analysis/vision.py`:

```python
from framed.analysis.intelligence_core import framed_intelligence
from framed.analysis.temporal_memory import query_memory_patterns, store_interpretation

def analyze_image(path, photo_id: str = "", filename: str = ""):
    # ... existing visual analysis ...
    
    # NEW: Add intelligence core
    visual_evidence = {
        "organic_growth": extract_visual_features(path).get("organic_growth", {}),
        "material_condition": extract_visual_features(path).get("material_condition", {}),
        # ... other visual evidence
    }
    
    # Query temporal memory
    temporal_memory = query_memory_patterns(create_pattern_signature(visual_evidence, semantic_signals))
    
    # Get user history (if available)
    user_history = get_user_history(photo_id)
    
    # Run intelligence core
    intelligence_output = framed_intelligence(
        visual_evidence=visual_evidence,
        temporal_memory=temporal_memory,
        user_history=user_history,
    )
    
    # Store in result
    result["intelligence"] = intelligence_output
    
    # Store interpretation in memory
    store_interpretation(
        signature=create_pattern_signature(visual_evidence, semantic_signals),
        interpretation=intelligence_output,
        confidence=intelligence_output.get("meta_cognition", {}).get("confidence", 0.85),
    )
    
    return result
```

#### Step 5.2: Replace Critique Generation

In `framed/routes.py`:

```python
from framed.analysis.expression_layer import generate_poetic_critique

# In analyze() route:
analysis_result = run_full_analysis(image_path)

# NEW: Use expression layer
if analysis_result.get("intelligence"):
    critique = generate_poetic_critique(
        intelligence_output=analysis_result["intelligence"],
        mentor_mode=mentor_mode or "Balanced Mentor",
    )
    analysis_result["critique"] = critique
```

**Key Points:**
- Keep existing visual analysis (YOLO, CLIP, OpenCV)
- Add intelligence core on top
- Replace rule-based synthesis with reasoning
- Replace rule-based critique with expression

**Dependencies:**
- All previous phases

---

### Phase 6: Model Implementation (FINAL STEP)

**Goal:** Replace placeholders with actual model implementations

**File to Modify:** `framed/analysis/llm_provider.py`

**⚠️ IMPORTANT: This phase should only be done after all other phases are complete and tested.**

#### Step 6.1: Choose Models

Decide on:
- **Model A (Reasoning):** Claude 3.5 Sonnet OR GPT-4 o1-mini
- **Model B (Expression):** Claude 3.5 Sonnet

#### Step 6.2: Implement Providers

Add to `framed/analysis/llm_provider.py`:

```python
# For Anthropic (Claude)
class AnthropicProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any], role: str):
        import anthropic
        api_key = os.getenv(config["api_key_env"])
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = config["model_name"]
        self.role = role
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def call(self, prompt, system_prompt=None, max_tokens=None, temperature=None, response_format=None):
        # Implement Anthropic API call
        messages = []
        if system_prompt:
            messages.append({"role": "user", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens or 4096,
            temperature=temperature,
            messages=messages,
        )
        
        return {
            "content": response.content[0].text,
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            "model": self.model_name,
            "error": None,
        }

# For OpenAI (GPT-4, o1)
class OpenAIProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any], role: str):
        from openai import OpenAI
        api_key = os.getenv(config["api_key_env"])
        self.client = OpenAI(api_key=api_key)
        self.model_name = config["model_name"]
        self.role = role
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def call(self, prompt, system_prompt=None, max_tokens=None, temperature=None, response_format=None):
        # Implement OpenAI API call
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_tokens or 4096,
            temperature=temperature,
            messages=messages,
            response_format=response_format,
        )
        
        return {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "model": self.model_name,
            "error": None,
        }
```

#### Step 6.3: Update Configuration

```python
MODEL_CONFIGS = {
    "CLAUDE_3_5_SONNET": {
        "provider": "anthropic",
        "model_name": "claude-3-5-sonnet-20241022",
        "api_key_env": "ANTHROPIC_API_KEY",
        "max_tokens": 4096,
        "temperature": 0.7,
    },
    "GPT4_O1_MINI": {
        "provider": "openai",
        "model_name": "o1-mini",
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens": 16384,
        "temperature": None,  # o1 doesn't use temperature
    },
    # ... add other models
}
```

#### Step 6.4: Update Factory Function

```python
def create_provider(model_type: str, role: Literal["reasoning", "expression"]) -> LLMProvider:
    config = MODEL_CONFIGS.get(model_type)
    
    if not config:
        logger.warning(f"Unknown model type: {model_type}. Using placeholder.")
        return PlaceholderProvider(model_name=f"placeholder-{role}")
    
    provider_name = config["provider"]
    
    if provider_name == "placeholder":
        return PlaceholderProvider(model_name=f"placeholder-{role}")
    elif provider_name == "anthropic":
        return AnthropicProvider(config, role)
    elif provider_name == "openai":
        return OpenAIProvider(config, role)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
```

#### Step 6.5: Set Environment Variables

```bash
# For Claude 3.5 Sonnet (both models)
export FRAMED_MODEL_A="CLAUDE_3_5_SONNET"
export FRAMED_MODEL_B="CLAUDE_3_5_SONNET"
export ANTHROPIC_API_KEY="your-key"

# OR for o1-mini (reasoning) + Claude (expression)
export FRAMED_MODEL_A="GPT4_O1_MINI"
export FRAMED_MODEL_B="CLAUDE_3_5_SONNET"
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

**Key Points:**
- Models are switchable via environment variables
- No code changes needed in intelligence core or expression layer
- Retry and fallback mechanisms already in place

**Dependencies:**
- All previous phases complete
- Model decision made
- API keys configured

---

## 🎯 Risk Mitigation

### Model Availability
- ✅ Retry logic (3 attempts with exponential backoff)
- ✅ Fallback models (configurable via environment variables)
- ✅ Graceful degradation (returns error, doesn't crash)

### Cost Management
- ✅ Cost tracking infrastructure (ready for actual costs)
- ✅ Token usage tracking
- ✅ Configurable rate limits

### API Failures
- ✅ Retry with exponential backoff
- ✅ Fallback to alternative models
- ✅ Error handling and logging

### Model Switching
- ✅ Environment variable configuration
- ✅ No code changes needed
- ✅ Lazy loading (models only initialized when needed)

---

## ✅ Ready to Start Implementation

**Status:** All architecture and placeholders are ready.

**Next Steps:**
1. Start with Phase 1 (Intelligence Core)
2. Test each phase independently
3. Proceed to next phase only after current phase is complete and tested
4. Replace placeholders in Phase 6 (final step)

**No model decision needed to start coding!** Placeholders allow full development and testing.
