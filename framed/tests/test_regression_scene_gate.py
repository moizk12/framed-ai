import os
import unittest
from pathlib import Path

from framed.analysis.intelligence_formatting import (
    infer_category_lexicon_key,
    is_portrait_or_single_subject,
    reconcile_recognition_with_scene,
)


class TestRecognitionSceneReconcile(unittest.TestCase):
    def _ve(self, scene_type: str, yolo=None, caption: str = ""):
        return {
            "scene_gate": {
                "scene_type": scene_type,
                "is_surface_study": False,
                "signals": {
                    "yolo_objects": yolo or [],
                    "clip_caption": caption,
                    "places_scene_category": "indoor",
                },
            },
            "organic_growth": {"green_coverage": 0.04},
            "material_condition": {"edge_degradation": 0.3, "color_uniformity": 0.5},
        }

    def test_portrait_not_layered_street_lexicon(self):
        ve = self._ve("people_scene", yolo=["person"], caption="a man reclining on a couch")
        self.assertTrue(is_portrait_or_single_subject(ve))
        self.assertIsNone(infer_category_lexicon_key(ve))

    def test_reconcile_street_hallucination_on_portrait(self):
        ve = self._ve("people_scene", yolo=["person"], caption="portrait of one man")
        primary = (
            "I see a layered street or urban scene with figures, foreground and background depth, "
            "and horizontal bands of activity."
        )
        out = reconcile_recognition_with_scene(primary, ve)
        self.assertIn("portrait", out.lower())
        self.assertNotIn("layered street", out.lower())

    def test_reconcile_ui_hallucination_on_cluttered_interior(self):
        ve = self._ve(
            "interior_scene",
            yolo=["couch", "chair", "clock", "tie"],
            caption="cluttered abandoned room with couch and shelves",
        )
        primary = (
            "I see a screen or digital display showing UI, code, or webpage content — "
            "layout, text readability, contrast, hierarchy, and crop are the primary subjects."
        )
        out = reconcile_recognition_with_scene(primary, ve)
        self.assertIn("interior", out.lower())
        self.assertNotIn("digital display", out.lower())

    def test_reconcile_person_hallucination_on_screenshot_ui(self):
        ve = self._ve(
            "screenshot_ui",
            yolo=["laptop", "keyboard"],
            caption="wikipedia webpage screenshot with article text",
        )
        ve["material_condition"] = {"edge_degradation": 0.8, "color_uniformity": 0.95}
        primary = "I see a person reading books beside a television in an urban scene."
        out = reconcile_recognition_with_scene(primary, ve)
        self.assertIn("screen", out.lower())
        self.assertNotIn("television", out.lower())

    def test_reconcile_surreal_screenshot_on_interior(self):
        ve = self._ve(
            "interior_scene",
            yolo=["couch"],
            caption="cluttered interior room with couch",
        )
        primary = "I see a surreal screenshot UI scene featuring a couch as the primary object"
        out = reconcile_recognition_with_scene(primary, ve)
        self.assertIn("physical", out.lower())
        self.assertNotIn("screenshot ui", out.lower())


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

