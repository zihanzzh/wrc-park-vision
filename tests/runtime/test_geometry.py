from __future__ import annotations

import unittest

from wrc_park_vision.runtime.schemas import BBoxGeometry


class GeometryTests(unittest.TestCase):
    def test_bbox_clipping_and_normalization(self) -> None:
        geometry = BBoxGeometry.from_xyxy((-10, 20, 120, 90), 100, 80)
        self.assertEqual(geometry.bbox_xyxy, (0.0, 20.0, 100.0, 80.0))
        self.assertEqual(geometry.bbox_normalized_xyxy, (0.0, 0.25, 1.0, 1.0))

    def test_invalid_bbox_after_clipping(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid after clipping"):
            BBoxGeometry.from_xyxy((120, 10, 150, 20), 100, 80)

    def test_non_finite_bbox_is_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            BBoxGeometry.from_xyxy((0, 0, float("nan"), 20), 100, 80)


if __name__ == "__main__":
    unittest.main()

