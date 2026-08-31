"""Real public-route UI server with a deterministic analysis-core seam.

Unlike ``preview_server.py``, this runs ``create_app`` and the production
``/api/v1`` routes. Only the expensive model runner is substituted so browser
E2E can verify frontend-to-Flask integration without external providers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framed import create_app
from framed.public_store import MemoryPublicRepository


def deterministic_analysis_runner(_path: str, _filename: str):
    return (
        {
            "critique": (
                "The road gives the eye an entry point, while the bright horizon competes "
                "with the valley's quieter rhythm. A tighter crop from the right would let "
                "the ridge and mist carry the frame."
            ),
            "perception": {"semantics": {"available": True, "caption": "A mountain valley with a winding road."}},
            "visual_evidence": {
                "scene_gate": {"scene_type": "mountain landscape"},
                "grounding": [],
                "theme_claim_license": {"tier": "cautious", "reasons": ["scene-level evidence"]},
            },
            "intelligence": {
                "recognition": {
                    "what_i_see": "A green mountain valley with a winding road, layered ridges, and low cloud.",
                    "confidence": 0.84,
                }
            },
        },
        42,
    )


app = create_app(
    {
        "PUBLIC_ANALYSIS_RUNNER": deterministic_analysis_runner,
        "PUBLIC_REPOSITORY": MemoryPublicRepository(),
    }
)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("FRAMED_API_E2E_PORT", "4174")), debug=False)
