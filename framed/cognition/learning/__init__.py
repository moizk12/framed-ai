"""Slice B controlled-learning loop. Proposal generation is separate from promotion."""

from .authority import accept_proposal, decide_proposal, reject_proposal
from .errors import LearningError, PromotionAuthorityError, PromotionBlockedError, ProposalImmutableError
from .evaluation import evaluate_proposal
from .outcomes import record_outcome
from .proposals import generate_proposal
from .rollback import rollback_promoted_state

__all__ = [
    "record_outcome",
    "generate_proposal",
    "evaluate_proposal",
    "accept_proposal",
    "reject_proposal",
    "decide_proposal",
    "rollback_promoted_state",
    "LearningError",
    "PromotionAuthorityError",
    "PromotionBlockedError",
    "ProposalImmutableError",
]
