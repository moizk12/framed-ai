"""
LLM Provider Abstraction Layer

This module provides a unified interface for LLM calls, allowing easy switching
between different models (Claude, GPT-4, o1, etc.) without changing calling code.

Architecture:
- Model A (Reasoning): Intelligence core - structured reasoning, meta-cognition
- Model B (Expression): Mentor voice - poetic critique, warm articulation

Current Status: PLACEHOLDER IMPLEMENTATION
Replace with actual model implementations after all phases are complete.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Literal
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ========================================================
# MODEL CONFIGURATION (PLACEHOLDER - REPLACE LATER)
# ========================================================

# Model selection via environment variables
MODEL_A_TYPE = os.getenv("FRAMED_MODEL_A", "GPT_5_2")   # Reasoning model (brain)
MODEL_B_TYPE = os.getenv("FRAMED_MODEL_B", "GPT_5_MINI")  # Expression model (voice)

# Model-specific configurations
MODEL_CONFIGS = {
    "PLACEHOLDER": {
        "provider": "placeholder",
        "model_name": "placeholder",
        "api_key_env": None,
        "max_tokens": 4000,
        "temperature": 0.7,
    },
    "GPT_5_2": {
        "provider": "openai",
        "model_name": "gpt-5.2",
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens": 8192,
        "temperature": None,
        "reasoning": {"effort": "medium"},
        "text": {"verbosity": "low"},
    },
    "GPT_5_MINI": {
        "provider": "openai",
        "model_name": "gpt-5-mini",
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens": 2048,
        "temperature": 1,  # Model only supports default (1); other values cause 400
        "text": {"verbosity": "medium"},
    },
}

# ========================================================
# RISK MITIGATION CONFIGURATION
# ========================================================

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

# Fallback configuration
FALLBACK_MODEL_A = os.getenv("FRAMED_FALLBACK_MODEL_A", "PLACEHOLDER")
FALLBACK_MODEL_B = os.getenv("FRAMED_FALLBACK_MODEL_B", "PLACEHOLDER")

# Rate limiting (to be configured per model)
RATE_LIMIT_CALLS_PER_MINUTE = {
    "PLACEHOLDER": 1000,  # No real limit for placeholder
    # Add actual rate limits when models are chosen
}

# Cost tracking (to be configured per model)
COST_PER_1M_TOKENS = {
    "PLACEHOLDER": {"input": 0.0, "output": 0.0},
    # Add actual costs when models are chosen
    # Example:
    # "CLAUDE_3_5_SONNET": {"input": 3.0, "output": 15.0},
    # "GPT4_O1_MINI": {"input": 0.15, "output": 0.60},
}


# ========================================================
# ABSTRACT BASE CLASSES
# ========================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make an LLM call.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (None for deterministic models)
            response_format: Optional response format (e.g., {"type": "json_object"})
        
        Returns:
            Dict with:
                - content: str - The generated text
                - usage: dict - Token usage information
                - model: str - Model used
                - error: Optional[str] - Error message if failed
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available (API key, etc.)"""
        pass


# ========================================================
# OPENAI PROVIDER (MODEL A & B)
# ========================================================


def _extract_text_from_responses_output(output: Any) -> str:
    """Extract text from Responses API output array."""
    if not output:
        return ""
    for item in output:
        if hasattr(item, "content") and item.content:
            for part in item.content:
                if hasattr(part, "text") and part.text:
                    return part.text
        if hasattr(item, "type") and item.type == "message":
            if hasattr(item, "content") and item.content:
                for part in item.content:
                    if getattr(part, "type", None) == "output_text" and hasattr(part, "text"):
                        return part.text
    return ""


class OpenAIProvider(LLMProvider):
    """OpenAI provider for Model A (gpt-5.2 reasoning) and Model B (gpt-5-mini expression)."""
    
    def __init__(self, config: Dict[str, Any], role: Literal["reasoning", "expression"]):
        self.config = config
        self.role = role
        self.model_name = config["model_name"]
        self._api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        self._client = None
    
    def _get_client(self):
        api_key = os.getenv(self._api_key_env, "").strip()
        # Fallback: load .env if key is missing (e.g. when run from different entry point)
        if not api_key:
            try:
                from pathlib import Path
                from dotenv import load_dotenv
                for d in (Path(__file__).resolve().parents[2], Path.cwd()):
                    env_path = d / ".env"
                    if env_path.exists():
                        load_dotenv(env_path)
                        api_key = os.getenv(self._api_key_env, "").strip()
                        break
            except ImportError:
                pass
        if not api_key:
            return None
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except Exception as e:
                logger.error(f"OpenAI client init failed: {e}")
                return None
        return self._client
    
    def is_available(self) -> bool:
        api_key = os.getenv(self._api_key_env, "").strip()
        if not api_key:
            return False
        return self._get_client() is not None
    
    def call(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: Optional[int] = None,
             temperature: Optional[float] = None, response_format: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        client = self._get_client()
        if not client:
            return {"content": "", "usage": {}, "model": self.model_name, "error": "OpenAI client not available"}
        max_tokens = max_tokens or self.config.get("max_tokens", 4096)
        if self.role == "reasoning":
            return self._call_reasoning(client, prompt, system_prompt, max_tokens, response_format)
        return self._call_expression(client, prompt, system_prompt, max_tokens,
                                     temperature if temperature is not None else self.config.get("temperature"))
    
    def _call_reasoning(self, client, prompt: str, system_prompt: Optional[str], max_tokens: int,
                        response_format: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if hasattr(client, "responses") and hasattr(client.responses, "create"):
            try:
                kwargs = {"model": self.model_name, "input": [{"role": "user", "content": prompt}], "max_output_tokens": max_tokens}
                if system_prompt:
                    kwargs["input"] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
                kwargs["reasoning"] = self.config.get("reasoning", {"effort": "medium"})
                kwargs["text"] = self.config.get("text", {"verbosity": "low"})
                # Note: Responses API does not support response_format; JSON request in prompt if needed
                resp = client.responses.create(**kwargs)
                text = getattr(resp, "output_text", None) or ""
                if not text and hasattr(resp, "output") and resp.output:
                    text = _extract_text_from_responses_output(resp.output)
                usage = {}
                if hasattr(resp, "usage") and resp.usage:
                    usage = {"prompt_tokens": getattr(resp.usage, "input_tokens", 0),
                             "completion_tokens": getattr(resp.usage, "output_tokens", 0),
                             "total_tokens": getattr(resp.usage, "total_tokens", 0)}
                return {"content": text, "usage": usage, "model": self.model_name, "error": None}
            except Exception as e:
                logger.warning(f"Responses API failed, falling back to Chat: {e}")
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append({"role": "user", "content": prompt})
        kwargs = {"model": self.model_name, "messages": messages, "max_completion_tokens": max_tokens, "temperature": 0.3}
        if response_format and response_format.get("type") == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message if resp.choices else None
        content = msg.content if msg else ""
        usage = {"prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                 "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                 "total_tokens": resp.usage.total_tokens if resp.usage else 0}
        return {"content": content or "", "usage": usage, "model": self.model_name, "error": None}
    
    def _call_expression(self, client, prompt: str, system_prompt: Optional[str], max_tokens: int,
                         temperature: Optional[float]) -> Dict[str, Any]:
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append({"role": "user", "content": prompt})
        # gpt-5-mini only supports temperature=1; ignore caller's temperature for that model
        if "gpt-5-mini" in (self.model_name or ""):
            temp = 1
        else:
            temp = temperature if temperature is not None else self.config.get("temperature", 0.7)
        resp = client.chat.completions.create(model=self.model_name, messages=messages, max_completion_tokens=max_tokens,
                                              temperature=temp)
        msg = resp.choices[0].message if resp.choices else None
        content = msg.content if msg else ""
        usage = {"prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                 "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                 "total_tokens": resp.usage.total_tokens if resp.usage else 0}
        return {"content": content or "", "usage": usage, "model": self.model_name, "error": None}


# ========================================================
# PLACEHOLDER IMPLEMENTATION
# ========================================================

class PlaceholderProvider(LLMProvider):
    """
    Placeholder LLM provider for development.
    
    Returns structured mock responses to allow development without actual LLM calls.
    Replace with actual provider implementations when models are chosen.
    """
    
    def __init__(self, model_name: str = "placeholder"):
        self.model_name = model_name
        self.is_available_flag = True  # Always available for placeholder
    
    def is_available(self) -> bool:
        return self.is_available_flag
    
    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Placeholder implementation - returns mock response.
        
        TODO: Replace with actual LLM call when model is chosen.
        """
        logger.warning(
            f"PLACEHOLDER LLM CALL - Model: {self.model_name}\n"
            f"System: {system_prompt[:100] if system_prompt else 'None'}...\n"
            f"Prompt: {prompt[:200]}..."
        )
        
        # Mock response structure
        mock_content = {
            "recognition": {
                "what_i_see": "[PLACEHOLDER] I see the image clearly.",
                "evidence": ["[PLACEHOLDER] Visual evidence"],
                "confidence": 0.85
            },
            "meta_cognition": {
                "why_i_believe_this": "[PLACEHOLDER] Based on evidence...",
                "confidence": 0.85,
                "what_i_might_be_missing": "[PLACEHOLDER] Potential gaps..."
            },
            "temporal": {
                "how_i_used_to_see_this": "[PLACEHOLDER] Previously...",
                "how_i_see_it_now": "[PLACEHOLDER] Now...",
                "evolution_reason": "[PLACEHOLDER] Evolution..."
            },
            "emotion": {
                "what_i_feel": "[PLACEHOLDER] I feel...",
                "why": "[PLACEHOLDER] Because...",
                "evolution": "[PLACEHOLDER] Evolution of feeling..."
            },
            "continuity": {
                "user_pattern": "[PLACEHOLDER] User usually...",
                "comparison": "[PLACEHOLDER] This compares to...",
                "trajectory": "[PLACEHOLDER] Trajectory..."
            },
            "mentor": {
                "observations": ["[PLACEHOLDER] Observation..."],
                "questions": ["[PLACEHOLDER] Question..."],
                "challenges": []
            },
            "self_critique": {
                "past_errors": ["[PLACEHOLDER] Past error..."],
                "evolution": "[PLACEHOLDER] How I evolved..."
            }
        }
        
        # If response_format is JSON, return JSON string
        if response_format and response_format.get("type") == "json_object":
            content = json.dumps(mock_content, indent=2)
        else:
            # Otherwise return prose
            content = "[PLACEHOLDER] This is a mock critique response. Replace with actual LLM implementation."
        
        return {
            "content": content,
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(content.split()),
                "total_tokens": len(prompt.split()) + len(content.split())
            },
            "model": self.model_name,
            "error": None
        }


# ========================================================
# PROVIDER FACTORY
# ========================================================

def create_provider(model_type: str, role: Literal["reasoning", "expression"]) -> LLMProvider:
    """
    Factory function to create LLM provider instances.
    
    Args:
        model_type: Model type identifier (e.g., "CLAUDE_3_5_SONNET", "GPT4_O1_MINI")
        role: "reasoning" for Model A, "expression" for Model B
    
    Returns:
        LLMProvider instance
    
    Raises:
        ValueError: If model type is not supported
    """
    config = MODEL_CONFIGS.get(model_type)
    
    if not config:
        logger.warning(f"Unknown model type: {model_type}. Using placeholder.")
        return PlaceholderProvider(model_name=f"placeholder-{role}")
    
    provider_name = config["provider"]
    
    if provider_name == "placeholder":
        return PlaceholderProvider(model_name=f"placeholder-{role}")
    
    if provider_name == "openai":
        return OpenAIProvider(config, role)
    
    logger.warning(f"Provider {provider_name} not implemented. Using placeholder.")
    return PlaceholderProvider(model_name=f"placeholder-{role}")


# ========================================================
# GLOBAL PROVIDER INSTANCES (LAZY-LOADED)
# ========================================================

_model_a_provider: Optional[LLMProvider] = None
_model_b_provider: Optional[LLMProvider] = None


def get_model_a_provider() -> LLMProvider:
    """Get Model A provider (Reasoning) - lazy-loaded"""
    global _model_a_provider
    if _model_a_provider is None:
        _model_a_provider = create_provider(MODEL_A_TYPE, "reasoning")
        logger.info(f"Model A (Reasoning) provider initialized: {MODEL_A_TYPE}")
    return _model_a_provider


def get_model_b_provider() -> LLMProvider:
    """Get Model B provider (Expression) - lazy-loaded"""
    global _model_b_provider
    if _model_b_provider is None:
        _model_b_provider = create_provider(MODEL_B_TYPE, "expression")
        logger.info(f"Model B (Expression) provider initialized: {MODEL_B_TYPE}")
    return _model_b_provider


# ========================================================
# UNIFIED CALL INTERFACE (WITH RETRY & FALLBACK)
# ========================================================

def call_model_a(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    response_format: Optional[Dict[str, Any]] = None,
    use_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Call Model A (Reasoning) with retry and fallback.
    
    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        max_tokens: Maximum tokens
        temperature: Sampling temperature
        response_format: Optional response format
        use_fallback: Whether to use fallback model on failure
    
    Returns:
        Dict with content, usage, model, error
    """
    provider = get_model_a_provider()
    
    # Retry logic
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            if not provider.is_available():
                raise ValueError(f"Model A provider not available: {MODEL_A_TYPE}")
            
            result = provider.call(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )
            
            if result.get("error"):
                raise Exception(result["error"])
            
            return result
        
        except Exception as e:
            last_error = e
            logger.warning(f"Model A call attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                import time
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
    
    # Fallback logic
    if use_fallback and FALLBACK_MODEL_A != MODEL_A_TYPE:
        logger.warning(f"Falling back to Model A fallback: {FALLBACK_MODEL_A}")
        fallback_provider = create_provider(FALLBACK_MODEL_A, "reasoning")
        try:
            return fallback_provider.call(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )
        except Exception as e:
            logger.error(f"Fallback Model A also failed: {e}")
    
    # Return error result
    return {
        "content": "",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "model": MODEL_A_TYPE,
        "error": str(last_error) if last_error else "Unknown error"
    }


def call_model_b(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    use_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Call Model B (Expression) with retry and fallback.
    
    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        max_tokens: Maximum tokens
        temperature: Sampling temperature
        use_fallback: Whether to use fallback model on failure
    
    Returns:
        Dict with content, usage, model, error
    """
    provider = get_model_b_provider()
    
    # Retry logic
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            if not provider.is_available():
                raise ValueError(f"Model B provider not available: {MODEL_B_TYPE}")
            
            result = provider.call(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=None,  # Expression is always prose
            )
            
            if result.get("error"):
                raise Exception(result["error"])
            
            return result
        
        except Exception as e:
            last_error = e
            logger.warning(f"Model B call attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                import time
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
    
    # Fallback logic
    if use_fallback and FALLBACK_MODEL_B != MODEL_B_TYPE:
        logger.warning(f"Falling back to Model B fallback: {FALLBACK_MODEL_B}")
        fallback_provider = create_provider(FALLBACK_MODEL_B, "expression")
        try:
            return fallback_provider.call(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=None,
            )
        except Exception as e:
            logger.error(f"Fallback Model B also failed: {e}")
    
    # Return error result
    return {
        "content": "",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "model": MODEL_B_TYPE,
        "error": str(last_error) if last_error else "Unknown error"
    }


# ========================================================
# COST TRACKING (OPTIONAL)
# ========================================================

def estimate_cost(model_type: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate cost for a model call.
    
    Args:
        model_type: Model type identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
    
    Returns:
        Estimated cost in USD
    """
    costs = COST_PER_1M_TOKENS.get(model_type, {"input": 0.0, "output": 0.0})
    input_cost = (input_tokens / 1_000_000) * costs["input"]
    output_cost = (output_tokens / 1_000_000) * costs["output"]
    return input_cost + output_cost
