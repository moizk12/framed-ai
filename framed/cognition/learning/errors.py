"""Controlled-learning errors. Authority failures are distinct from evaluation blocks."""

from __future__ import annotations


class LearningError(ValueError):
    """Base Slice B learning error."""


class PromotionAuthorityError(LearningError):
    """Raised when a non-external actor attempts to promote or roll back."""


class PromotionBlockedError(LearningError):
    """Raised when accept is attempted without a passing replay/regression evaluation."""


class ProposalImmutableError(LearningError):
    """Raised when a proposal has already been decided."""
