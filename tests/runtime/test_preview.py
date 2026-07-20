from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from wrc_park_vision.runtime.config import PreviewSettings
from wrc_park_vision.runtime.fusion import merge_and_mark_conflicts
from wrc_park_vision.runtime.preview import render_preview

from .helpers import make_observation, make_response, write_test_image


class PreviewTests(unittest.TestCase):
    def test_preview_uses_pipeline_response_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = write_test_image(root / "image.jpg", (100, 80))
            observations = merge_and_mark_conflicts(
                [make_observation("prohibited_items", "prohibited", 0, "spray_can", 0.9, (10, 20, 50, 60))],
                0.75,
            )
            response = make_response(observations, image_path)
            preview_path = root / "preview.jpg"
            render_preview(image_path, response, preview_path, PreviewSettings())
            with Image.open(preview_path) as image:
                preview = image.convert("RGB")
                edge_pixel = preview.getpixel((10, 40))
                untouched_pixel = preview.getpixel((80, 70))
        self.assertGreater(edge_pixel[0], edge_pixel[1] * 2)
        self.assertTrue(all(channel > 220 for channel in untouched_pixel))


if __name__ == "__main__":
    unittest.main()
