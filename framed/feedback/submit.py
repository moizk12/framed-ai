"""
Submit HITL feedback via CLI.

Run: python -m framed.feedback.submit --image_id X --type overconfidence
"""

from .submit_cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
