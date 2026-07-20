from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wrc_park_vision.runtime.config import PreviewSettings
from wrc_park_vision.runtime.output import write_runtime_outputs

from .helpers import make_observation, make_response, write_test_image


class OutputTests(unittest.TestCase):
    def test_preview_failure_keeps_json_and_observations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = write_test_image(root / "image.jpg")
            response = make_response(
                [make_observation("garbage", "garbage", 0, "plastic_bottle", 0.91, (10, 10, 40, 60))],
                image_path,
            )

            with patch(
                "wrc_park_vision.runtime.output.render_preview",
                side_effect=RuntimeError("preview unavailable"),
            ):
                artifacts = write_runtime_outputs(response, root / "outputs", PreviewSettings(), True)

            payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))

        self.assertIsNone(artifacts.preview_path)
        self.assertEqual(len(payload["observations"]), 1)
        self.assertEqual(payload["observations"][0]["class_name"], "plastic_bottle")
        self.assertEqual(payload["errors"][-1]["code"], "preview_failure")


if __name__ == "__main__":
    unittest.main()
