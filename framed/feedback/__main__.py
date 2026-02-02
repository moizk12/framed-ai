"""
Run: python -m framed.feedback [args]

Same as: python -m framed.feedback.submit [args]
"""

from .submit_cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
