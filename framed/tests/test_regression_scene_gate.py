import os
import unittest
from pathlib import Path


class TestRegressionSceneGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Regression tests must never spend API credits.
        os.environ["FRAMED_ENABLE_INTELLIGENCE_CORE"] = "false"
        os.environ["FRAMED_DISABLE_EXPRESSION"] = "true"
        os.environ["FRAMED_MODEL_B_EMPTY_RETRY"] = "false"

    def _analyze(self, image_rel: str):
        from framed.analysis.vision import analyze_image
        root = Path(__file__).resolve().parents[0]
        img_path = root / "regression_scene_gate" / "images" / image_rel
        self.assertTrue(img_path.exists(), f"Missing regression image: {img_path}")
        return analyze_image(str(img_path), photo_id=f"reg_{img_path.stem}", filename=img_path.name, disable_cache=True)

    def _assert_suppressed(self, ve: dict, expected_scene_type: str):
        sg = (ve or {}).get("scene_gate", {}) or {}
        self.assertEqual(sg.get("scene_type"), expected_scene_type)
        self.assertFalse(bool(sg.get("is_surface_study")), "Expected non-surface scene")
        self.assertTrue(
            (sg.get("surface_study_rejection_reasons") or []),
            "Expected rejection reasons when not surface_study",
        )

        oi = (ve or {}).get("organic_integration", {}) or {}
        self.assertEqual(oi.get("relationship"), "none", "Expected no organic_integration relationship for non-surface scenes")
        # If the gate explicitly disabled an integration claim, it will add a note and cap confidence.
        ev = " ".join([str(x) for x in (oi.get("evidence", []) or [])]).lower()
        if "scene_gate=" in ev:
            self.assertLessEqual(float(oi.get("confidence", 1.0) or 1.0), 0.2, "Expected organic_integration confidence capped when gated")

    def test_interior_scene_gate(self):
        res = self._analyze("interior_001.jpg")
        ve = res.get("visual_evidence", {}) or {}
        self._assert_suppressed(ve, "interior_scene")

    def test_landscape_scene_gate(self):
        res = self._analyze("landscape_001.jpg")
        ve = res.get("visual_evidence", {}) or {}
        self._assert_suppressed(ve, "landscape_scene")

    def test_abstract_scene_gate(self):
        res = self._analyze("abstract_001.jpg")
        ve = res.get("visual_evidence", {}) or {}
        self._assert_suppressed(ve, "abstract_art")

    def test_people_scene_gate(self):
        res = self._analyze("portrait_001.jpg")
        ve = res.get("visual_evidence", {}) or {}
        self._assert_suppressed(ve, "people_scene")

    def test_surface_study_scene_gate(self):
        res = self._analyze("surface_closeup_001.jpg")
        ve = res.get("visual_evidence", {}) or {}
        sg = (ve or {}).get("scene_gate", {}) or {}
        self.assertEqual(sg.get("scene_type"), "surface_study")
        self.assertTrue(bool(sg.get("is_surface_study")), "Expected surface_study to be true for close-up texture crop")
        # For surface studies we do not force-disable organic_integration; relationship may vary.


if __name__ == "__main__":
    unittest.main()

