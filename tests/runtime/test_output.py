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
    def test_raw_vlm_response_is_written_only_to_debug_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response = make_response([], write_test_image(root / "image.jpg"))
            response.review.raw_response_debug = "full raw response tail"

            artifacts = write_runtime_outputs(
                response,
                root / "outputs",
                PreviewSettings(enabled=False),
                False,
            )
            internal = artifacts.json_path.read_text(encoding="utf-8")
            competition = artifacts.competition_json_path.read_text(encoding="utf-8")
            raw_response = artifacts.vlm_raw_response_path.read_text(encoding="utf-8")

        self.assertEqual(raw_response, "full raw response tail")
        self.assertNotIn("full raw response tail", internal)
        self.assertNotIn("full raw response tail", competition)

    def test_successful_preview_timing_is_persisted_to_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = write_test_image(root / "image.jpg")
            response = make_response([], image_path)
            initial_total = response.timing_ms.total

            artifacts = write_runtime_outputs(
                response,
                root / "outputs",
                PreviewSettings(),
                True,
            )

            payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))

        self.assertIsNotNone(artifacts.preview_path)
        self.assertIsNotNone(artifacts.competition_json_path)
        self.assertIsNotNone(payload["timing_ms"]["preview"])
        self.assertIsNotNone(
            payload["timing_ms"]["competition_response_adapter"]
        )
        self.assertIsNotNone(payload["timing_ms"]["output_serialization"])
        self.assertGreaterEqual(payload["timing_ms"]["total"], initial_total)

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
        self.assertIsNotNone(payload["timing_ms"]["preview"])

    def test_competition_mode_writes_sdk_result_before_skipping_preview(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response = make_response(
                [
                    make_observation(
                        "garbage",
                        "garbage",
                        0,
                        "plastic_bottle",
                        0.91,
                        (10, 10, 40, 60),
                    )
                ],
                write_test_image(root / "image.jpg"),
            )

            artifacts = write_runtime_outputs(
                response,
                root / "outputs",
                PreviewSettings(),
                True,
                output_mode="competition",
            )
            payload = json.loads(
                artifacts.competition_json_path.read_text(encoding="utf-8")
            )

        self.assertIsNone(artifacts.json_path)
        self.assertIsNone(artifacts.preview_path)
        self.assertEqual(payload["objects"][0]["class_name"], "plastic_bottle")


if __name__ == "__main__":
    unittest.main()
