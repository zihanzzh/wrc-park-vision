from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from wrc_park_vision.runtime.backends.yolo_world import (
    YOLOWorldBackend,
    YOLOWorldClassDefinition,
)
from wrc_park_vision.runtime.schemas import ValidatedImage


class _TensorValues:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class _Boxes:
    def __init__(self, xyxy, class_ids, confidences):
        self.xyxy = _TensorValues(xyxy)
        self.cls = _TensorValues(class_ids)
        self.conf = _TensorValues(confidences)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes
        self.speed = {"inference": 12.5}


class _FakeYOLOWorld:
    instances = []
    result = _Result(_Boxes([], [], []))

    def __init__(self, model_path):
        self.model_path = model_path
        self.names = {}
        self.model = types.SimpleNamespace(clip_model=object())
        self.set_classes_calls = []
        self.predict_calls = []
        self.__class__.instances.append(self)

    def set_classes(self, classes):
        self.set_classes_calls.append(list(classes))
        self.names = list(classes)

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return [self.__class__.result]


class YOLOWorldBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeYOLOWorld.instances = []
        _FakeYOLOWorld.result = _Result(_Boxes([], [], []))

    def make_backend(self, model_path: Path) -> YOLOWorldBackend:
        return YOLOWorldBackend(
            model_path=model_path,
            module_id="world_objects",
            model_id="yolov8s-worldv2",
            classes=[
                YOLOWorldClassDefinition(
                    "prohibited_items",
                    0,
                    "spray_can",
                    ("spray can", "aerosol can"),
                ),
                YOLOWorldClassDefinition(
                    "uncivilized_behavior",
                    0,
                    "person",
                    ("person",),
                ),
            ],
            device="cpu",
            confidence=0.1,
            iou=0.7,
            imgsz=640,
        )

    def test_load_once_and_map_prompts_to_grouped_canonical_detections(self) -> None:
        _FakeYOLOWorld.result = _Result(
            _Boxes(
                [[1, 2, 21, 32], [2, 3, 22, 33], [40, 5, 60, 35]],
                [0, 1, 2],
                [0.9, 0.8, 0.7],
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "world.pt"
            model_path.touch()
            backend = self.make_backend(model_path)
            fake_module = types.SimpleNamespace(YOLOWorld=_FakeYOLOWorld)
            with patch.dict(sys.modules, {"ultralytics": fake_module}):
                backend.load()
                backend.load()
                prediction = backend.predict(
                    ValidatedImage("image.jpg", Image.new("RGB", (100, 80)), 100, 80)
                )

        self.assertEqual(len(_FakeYOLOWorld.instances), 1)
        model = _FakeYOLOWorld.instances[0]
        self.assertEqual(
            model.set_classes_calls,
            [["spray can", "aerosol can", "person"]],
        )
        self.assertIsNone(model.model.clip_model)
        self.assertEqual(len(prediction.detections), 2)
        spray, person = prediction.detections
        self.assertEqual((spray.task_group, spray.class_id, spray.class_name), (
            "prohibited_items",
            0,
            "spray_can",
        ))
        self.assertEqual(spray.metadata["matched_prompt"], "spray can")
        self.assertEqual((person.task_group, person.class_id, person.class_name), (
            "uncivilized_behavior",
            0,
            "person",
        ))
        self.assertEqual(prediction.timing_ms, {"inference": 12.5})

    def test_rejects_garbage_class_definitions(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not provide garbage detections"):
            YOLOWorldBackend(
                model_path=Path("world.pt"),
                module_id="world_objects",
                model_id="yolov8s-worldv2",
                classes=[
                    YOLOWorldClassDefinition(
                        "garbage",
                        0,
                        "plastic_drink_bottle",
                        ("plastic drink bottle",),
                    )
                ],
                device="cpu",
                confidence=0.1,
                iou=0.7,
                imgsz=640,
            )

    def test_missing_weight_does_not_create_or_download_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = self.make_backend(Path(directory) / "missing.pt")
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                backend.load()
        self.assertEqual(_FakeYOLOWorld.instances, [])

    def test_unknown_prompt_class_id_fails_explicitly(self) -> None:
        _FakeYOLOWorld.result = _Result(_Boxes([[1, 2, 21, 32]], [99], [0.9]))
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "world.pt"
            model_path.touch()
            backend = self.make_backend(model_path)
            fake_module = types.SimpleNamespace(YOLOWorld=_FakeYOLOWorld)
            with patch.dict(sys.modules, {"ultralytics": fake_module}):
                backend.load()
                with self.assertRaisesRegex(ValueError, "unknown prompt class ID 99"):
                    backend.predict(
                        ValidatedImage("image.jpg", Image.new("RGB", (100, 80)), 100, 80)
                    )


if __name__ == "__main__":
    unittest.main()
