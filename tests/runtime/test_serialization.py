from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from wrc_park_vision.runtime.output import write_json
from wrc_park_vision.runtime.fusion import merge_and_mark_conflicts
from wrc_park_vision.runtime.schemas import RequestContext

from .helpers import make_observation, make_response, write_test_image


class SerializationTests(unittest.TestCase):
    def test_json_schema_and_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = write_test_image(root / "image.jpg")
            observations = merge_and_mark_conflicts(
                [make_observation("garbage", "garbage", 3, "plastic_drink_bottle", 0.91, (10, 8, 40, 60))],
                0.75,
            )
            response = make_response(observations, image_path)
            output_path = root / "result.json"
            write_json(response, output_path)
            data = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], "1.0")
        self.assertIsNone(data["detection_summary"])
        self.assertEqual(data["fusion"], {"status": "not_run", "decisions": []})
        self.assertEqual(data["observations"][0]["geometry"]["bbox_xyxy"], [10.0, 8.0, 40.0, 60.0])
        self.assertEqual(data["observations"][0]["geometry"]["bbox_normalized_xyxy"], [0.1, 0.1, 0.4, 0.75])
        self.assertIsNone(data["observations"][0]["track_id"])

    def test_track_id_and_datetime_serialize(self) -> None:
        observation = make_observation("garbage", "garbage", 3, "plastic_drink_bottle", 0.91, (10, 8, 40, 60))
        observation.track_id = "track-7"
        context = RequestContext(
            timestamp=datetime(2026, 7, 19, 12, 30, tzinfo=timezone.utc),
            frame_index=0,
        )

        self.assertEqual(observation.model_dump(mode="json")["track_id"], "track-7")
        self.assertEqual(context.model_dump(mode="json")["timestamp"], "2026-07-19T12:30:00Z")

    def test_negative_frame_index_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RequestContext(frame_index=-1)


if __name__ == "__main__":
    unittest.main()
