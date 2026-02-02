"""
Minimal CLI for submitting HITL feedback.

Usage:
  python -m framed.feedback.submit --image_id architecture_042 --type overconfidence
  python -m framed.feedback.submit --image_id street_001 --type missed_alternative --alternative_hint "green could be painted facade"
  python -m framed.feedback.submit --image_id portrait_003 --type emphasis_misaligned --dimension emotional_weighting
  python -m framed.feedback.submit --image_id nature_002 --type mentor_failure --reason "generic guidance"
"""

import argparse
import json
import sys

from .storage import append_feedback, HITL_FEEDBACK_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Submit HITL (human-in-the-loop) feedback to FRAMED",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Feedback types:
  overconfidence    FRAMED sounded certain when evidence was ambiguous
                    Effect: Tightens confidence governor for similar patterns
                    Use --confidence-delta-hint for magnitude (e.g. -0.15), optional

  missed_alternative  Another reading was clearly viable
                    Effect: Raises multi-hypothesis branching probability
                    Use --alternative-hint for a hint (not a label), e.g. "green could be painted facade"

  emphasis_misaligned  FRAMED focused on the wrong thing
                    Effect: Adjusts salience weighting
                    Use --dimension e.g. emotional_weighting, scale, solitude

  mentor_failure    Tone felt generic, advice obvious, voice drifted
                    Effect: Tightens reflection checks for mentor drift
                    Use --reason e.g. "generic guidance"
        """,
    )
    parser.add_argument("--image_id", "-i", required=True, help="Image identifier (e.g. architecture_042)")
    parser.add_argument("--type", "-t", required=True, choices=["overconfidence", "missed_alternative", "emphasis_misaligned", "mentor_failure"], help="Feedback type")
    parser.add_argument("--signature", "-s", required=True, help="Pattern signature (REQUIRED). Keeps calibration localized.")
    parser.add_argument("--confidence_delta_hint", "-c", type=float, default=None, help="For overconfidence: magnitude hint, e.g. -0.15 (optional; heuristic used if omitted)")
    parser.add_argument("--alternative_hint", "-a", default="", help="For missed_alternative: hint (not label), e.g. 'green could be painted facade'")
    parser.add_argument("--dimension", "-d", default="general", help="For emphasis_misaligned: dimension e.g. emotional_weighting, scale")
    parser.add_argument("--reason", "-r", default="generic", help="For mentor_failure: reason e.g. 'generic guidance'")
    parser.add_argument("--scope", default="belief_calibration", help="For overconfidence: scope (default belief_calibration)")
    args = parser.parse_args()

    feedback = {"type": args.type}
    if args.type == "overconfidence":
        feedback["scope"] = args.scope
        if args.confidence_delta_hint is not None:
            feedback["confidence_delta_hint"] = args.confidence_delta_hint
    elif args.type == "missed_alternative" and args.alternative_hint:
        feedback["alternative_hint"] = args.alternative_hint
    elif args.type == "emphasis_misaligned":
        feedback["dimension"] = args.dimension
    elif args.type == "mentor_failure":
        feedback["reason"] = args.reason

    ok = append_feedback(args.image_id, feedback, signature=args.signature)
    if ok:
        print(f"Feedback submitted: {args.image_id} {args.type}")
        print(f"Stored at: {HITL_FEEDBACK_PATH}")
        return 0
    else:
        print("Failed to submit feedback", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
