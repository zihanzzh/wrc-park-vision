from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wrc_park_vision.runtime.backends.base import BackendDetection
from wrc_park_vision.runtime.modules.detection import DetectionModule
from wrc_park_vision.runtime.pipeline import RuntimePipeline
from wrc_park_vision.runtime.review import ReviewProvider
from wrc_park_vision.runtime.schemas import Observation, ValidatedImage

from .helpers import FakeBackend, make_config, write_test_image


class PipelineTests(unittest.TestCase):
    def test_two_modules_run_and_models_load_once(self) -> None:
        prohibited_backend = FakeBackend(
            "prohibited_model",
            [BackendDetection(0, "spray_can", 0.9, (10, 10, 30, 40))],
        )
        garbage_backend = FakeBackend(
            "garbage_model",
            [BackendDetection(3, "plastic_drink_bottle", 0.8, (50, 10, 70, 50))],
        )
        modules = [
            DetectionModule("prohibited", "prohibited_items", "prohibited_model", prohibited_backend),
            DetectionModule("garbage", "garbage", "garbage_model", garbage_backend),
        ]
        config = make_config(("prohibited", "garbage"))
        with tempfile.TemporaryDirectory() as directory:
            image_path = write_test_image(Path(directory) / "image.jpg")
            pipeline = RuntimePipeline(config, modules=modules)
            first = pipeline.process(image_path, request_id="first")
            second = pipeline.process(image_path, request_id="second")
        self.assertEqual(first.status, "success")
        self.assertEqual(second.status, "success")
        self.assertEqual(len(first.observations), 2)
        self.assertEqual(prohibited_backend.load_calls, 1)
        self.assertEqual(garbage_backend.load_calls, 1)
        self.assertEqual(prohibited_backend.predict_calls, 2)

    def test_single_module_failure_keeps_successful_results(self) -> None:
        good = FakeBackend("good", [BackendDetection(0, "spray_can", 0.9, (10, 10, 30, 40))])
        bad = FakeBackend("bad", error=RuntimeError("backend failed"))
        modules = [
            DetectionModule("prohibited", "prohibited", "good", good),
            DetectionModule("garbage", "garbage", "bad", bad),
        ]
        with tempfile.TemporaryDirectory() as directory:
            image_path = write_test_image(Path(directory) / "image.jpg")
            response = RuntimePipeline(make_config(("prohibited", "garbage")), modules).process(image_path)
        self.assertEqual(response.status, "partial_success")
        self.assertEqual(len(response.observations), 1)
        self.assertIn("module_failure", response.review.reasons)
        self.assertEqual(len(response.errors), 1)

    def test_all_modules_failure(self) -> None:
        modules = [
            DetectionModule("prohibited", "prohibited", "bad1", FakeBackend("bad1", error=RuntimeError("one"))),
            DetectionModule("garbage", "garbage", "bad2", FakeBackend("bad2", error=RuntimeError("two"))),
        ]
        with tempfile.TemporaryDirectory() as directory:
            image_path = write_test_image(Path(directory) / "image.jpg")
            response = RuntimePipeline(make_config(("prohibited", "garbage")), modules).process(image_path)
        self.assertEqual(response.status, "failure")
        self.assertEqual(response.observations, [])
        self.assertEqual(len(response.errors), 2)

    def test_invalid_image_is_failure(self) -> None:
        backend = FakeBackend("unused")
        module = DetectionModule("garbage", "garbage", "unused", backend)
        response = RuntimePipeline(make_config(("garbage",)), [module]).process(Path("missing-image.jpg"))
        self.assertEqual(response.status, "failure")
        self.assertEqual(response.errors[0].stage, "input")
        self.assertEqual(backend.predict_calls, 0)

    def test_fusion_failure_keeps_observations_with_stable_ids(self) -> None:
        backend = FakeBackend("good", [BackendDetection(0, "spray_can", 0.9, (10, 10, 30, 40))])
        module = DetectionModule("prohibited", "prohibited_items", "good", backend)
        with tempfile.TemporaryDirectory() as directory:
            image_path = write_test_image(Path(directory) / "image.jpg")
            pipeline = RuntimePipeline(make_config(("prohibited",)), [module])
            with patch(
                "wrc_park_vision.runtime.pipeline.merge_and_mark_conflicts",
                side_effect=RuntimeError("fusion broke"),
            ):
                first = pipeline.process(image_path, request_id="first")
                second = pipeline.process(image_path, request_id="second")

        self.assertEqual(first.status, "partial_success")
        self.assertEqual(len(first.observations), 1)
        self.assertEqual(first.observations[0].id, "obs-0001")
        self.assertEqual(first.observations[0].id, second.observations[0].id)
        self.assertTrue(any(error.stage == "fusion" and error.code == "fusion_failure" for error in first.errors))

    def test_review_failure_keeps_fused_observations(self) -> None:
        class FailingReviewProvider(ReviewProvider):
            def review(self, image: ValidatedImage, observations: list[Observation]) -> list[Observation]:
                raise RuntimeError("review broke")

        backend = FakeBackend("good", [BackendDetection(0, "spray_can", 0.3, (10, 10, 30, 40))])
        module = DetectionModule("prohibited", "prohibited_items", "good", backend)
        with tempfile.TemporaryDirectory() as directory:
            image_path = write_test_image(Path(directory) / "image.jpg")
            response = RuntimePipeline(
                make_config(("prohibited",)),
                [module],
                review_provider=FailingReviewProvider(),
            ).process(image_path)

        self.assertEqual(response.status, "partial_success")
        self.assertEqual(len(response.observations), 1)
        self.assertEqual(response.observations[0].id, "obs-0001")
        self.assertEqual(response.review.reasons, ["review_failure"])
        self.assertTrue(any(error.stage == "review" and error.code == "review_failure" for error in response.errors))
        self.assertFalse(any(error.code == "fusion_failure" for error in response.errors))

    def test_initialization_failure_closes_previously_loaded_modules(self) -> None:
        first_backend = FakeBackend("first", close_error=RuntimeError("close also broke"))
        second_backend = FakeBackend("second")
        third_backend = FakeBackend("third", load_error=RuntimeError("original load failure"))
        modules = [
            DetectionModule("first", "first", "first", first_backend),
            DetectionModule("second", "second", "second", second_backend),
            DetectionModule("third", "third", "third", third_backend),
        ]

        with self.assertRaisesRegex(RuntimeError, "original load failure"):
            RuntimePipeline(make_config(("first", "second", "third")), modules)

        self.assertEqual(first_backend.close_calls, 1)
        self.assertEqual(second_backend.close_calls, 1)
        self.assertEqual(third_backend.close_calls, 0)

    def test_close_is_best_effort(self) -> None:
        first_backend = FakeBackend("first", close_error=RuntimeError("close failed"))
        second_backend = FakeBackend("second")
        modules = [
            DetectionModule("first", "first", "first", first_backend),
            DetectionModule("second", "second", "second", second_backend),
        ]
        pipeline = RuntimePipeline(make_config(("first", "second")), modules)

        pipeline.close()

        self.assertEqual(first_backend.close_calls, 1)
        self.assertEqual(second_backend.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
