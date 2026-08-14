from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from wrc_park_vision.runtime import (
    RequestContext,
    RuntimePipeline,
    build_competition_response,
)
from wrc_park_vision.runtime.backends.base import BackendDetection
from wrc_park_vision.runtime.modules.detection import DetectionModule

from .helpers import FakeBackend, make_config, write_test_image


class IntegrationAPITests(unittest.TestCase):
    def test_public_api_reuses_one_pipeline_for_path_and_pil_frame(self) -> None:
        backend = FakeBackend(
            "garbage_model",
            [
                BackendDetection(
                    0,
                    "garbage_class",
                    0.9,
                    (10, 10, 30, 40),
                )
            ],
        )
        pipeline = RuntimePipeline(
            make_config(("garbage",)),
            [
                DetectionModule(
                    "garbage",
                    "garbage",
                    "garbage_model",
                    backend,
                )
            ],
        )
        context = RequestContext(
            camera_id="camera-01",
            timestamp=datetime(2026, 8, 11, tzinfo=timezone.utc),
            session_id="session-01",
            frame_index=7,
        )

        with tempfile.TemporaryDirectory() as directory:
            image_path = write_test_image(Path(directory) / "frame.jpg")
            path_result = pipeline.process(image_path, context=context)
            with Image.open(image_path) as source:
                frame_result = pipeline.process(source.copy(), context=context)
        product_result = build_competition_response(frame_result)
        pipeline.close()

        self.assertEqual(backend.load_calls, 1)
        self.assertEqual(backend.predict_calls, 2)
        self.assertEqual(path_result.status, "success")
        self.assertEqual(frame_result.input.image_path, "<memory:pillow-image>")
        self.assertEqual(frame_result.input.context.camera_id, "camera-01")
        self.assertEqual(product_result.frame.frame_id, 7)
        self.assertEqual(product_result.objects[0].class_name, "garbage_class")


if __name__ == "__main__":
    unittest.main()
