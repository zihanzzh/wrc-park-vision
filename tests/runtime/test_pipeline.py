from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wrc_park_vision.runtime.backends.base import BackendDetection
from wrc_park_vision.runtime.modules.detection import DetectionModule
from wrc_park_vision.runtime.config import RuntimeConfig
from wrc_park_vision.runtime.pipeline import (
    RuntimePipeline,
    build_class_catalog,
    build_modules,
    build_visual_class_guide,
)
from wrc_park_vision.runtime.review import ReviewProvider
from wrc_park_vision.runtime.schemas import (
    DetectionSummary,
    VLMFinding,
    VLMReviewDecision,
    VLMReviewResult,
    ValidatedImage,
)

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

    def test_detection_level_task_groups_flow_through_summary_and_review(self) -> None:
        class CapturingReviewProvider(ReviewProvider):
            def review(self, image: ValidatedImage, summary: DetectionSummary) -> VLMReviewResult:
                self.summary = summary
                return VLMReviewResult(
                    provider="fake_vlm",
                    model_id="fake-vl",
                    duration_ms=1,
                    decisions=[
                        VLMReviewDecision(observation_id=item.observation_id, verdict="confirmed")
                        for item in summary.detections
                    ],
                )

        provider = CapturingReviewProvider()
        world_backend = FakeBackend(
            "world",
            [
                BackendDetection(
                    0,
                    "spray_can",
                    0.9,
                    (10, 10, 30, 40),
                    task_group="prohibited_items",
                ),
            ],
        )
        garbage_backend = FakeBackend(
            "garbage_yolo11m",
            [BackendDetection(3, "plastic_drink_bottle", 0.8, (50, 10, 70, 50))],
        )
        modules = [
            DetectionModule("world", "object_detection", "world", world_backend),
            DetectionModule("garbage", "garbage", "garbage_yolo11m", garbage_backend),
        ]
        with tempfile.TemporaryDirectory() as directory:
            image_path = write_test_image(Path(directory) / "image.jpg")
            response = RuntimePipeline(
                make_config(("world", "garbage")),
                modules,
                review_provider=provider,
            ).process(image_path)

        self.assertEqual(
            [observation.task_group for observation in response.observations],
            ["garbage", "prohibited_items"],
        )
        self.assertEqual(
            provider.summary.counts_by_task_group,
            {"garbage": 1, "prohibited_items": 1},
        )
        self.assertEqual(response.review.status, "completed")

    def test_yolo_world_module_factory_and_class_catalog_use_grouped_classes(self) -> None:
        config = RuntimeConfig.model_validate(
            {
                "modules": [
                    {
                        "id": "world",
                        "enabled": True,
                        "type": "detection",
                        "task_group": "object_detection",
                        "backend": "yolo_world",
                        "model_path": Path("world.pt"),
                        "model_id": "world_model",
                        "open_vocabulary_classes": [
                            {
                                "task_group": "prohibited_items",
                                "class_id": 0,
                                "class_name": "spray_can",
                                "prompts": ["spray can"],
                                "visual_description": "带喷头的气雾罐。",
                                "distinguishing_rules": ["普通饮料罐不是喷雾罐。"],
                            },
                            {
                                "task_group": "uncivilized_behavior",
                                "class_id": 0,
                                "class_name": "person",
                                "prompts": ["person"],
                            },
                        ],
                    },
                    {
                        "id": "garbage",
                        "enabled": True,
                        "type": "detection",
                        "task_group": "garbage",
                        "backend": "ultralytics",
                        "model_path": Path("garbage.pt"),
                        "model_id": "garbage_yolo11m",
                        "expected_class_names": [
                            "crumpled_paper_ball",
                            "disposable_food_container",
                            "empty_cigarette_box",
                            "plastic_drink_bottle",
                            "plastic_food_wrapper",
                            "rigid_takeout_bag",
                        ],
                    },
                ]
            }
        )

        with (
            patch("wrc_park_vision.runtime.pipeline.YOLOWorldBackend") as world_backend_class,
            patch("wrc_park_vision.runtime.pipeline.UltralyticsBackend") as garbage_backend_class,
        ):
            modules = build_modules(config)

        self.assertEqual(len(modules), 2)
        world_backend_class.assert_called_once()
        garbage_backend_class.assert_called_once()
        definitions = world_backend_class.call_args.kwargs["classes"]
        self.assertEqual(
            [(item.task_group, item.class_id, item.class_name) for item in definitions],
            [
                ("prohibited_items", 0, "spray_can"),
                ("uncivilized_behavior", 0, "person"),
            ],
        )
        self.assertEqual(
            garbage_backend_class.call_args.kwargs["expected_class_names"],
            [
                "crumpled_paper_ball",
                "disposable_food_container",
                "empty_cigarette_box",
                "plastic_drink_bottle",
                "plastic_food_wrapper",
                "rigid_takeout_bag",
            ],
        )
        self.assertEqual(
            build_class_catalog(config),
            {
                "prohibited_items": ["spray_can"],
                "uncivilized_behavior": ["person"],
                "garbage": [
                    "crumpled_paper_ball",
                    "disposable_food_container",
                    "empty_cigarette_box",
                    "plastic_drink_bottle",
                    "plastic_food_wrapper",
                    "rigid_takeout_bag",
                ],
            },
        )
        self.assertEqual(
            build_visual_class_guide(config),
            {
                "prohibited_items": {
                    "spray_can": {
                        "visual": "带喷头的气雾罐。",
                        "distinguish": ["普通饮料罐不是喷雾罐。"],
                    }
                }
            },
        )

    def test_visual_class_guide_omits_disabled_modules(self) -> None:
        config = RuntimeConfig.model_validate(
            {
                "modules": [
                    {
                        "id": "enabled_world",
                        "enabled": True,
                        "type": "detection",
                        "task_group": "object_detection",
                        "backend": "yolo_world",
                        "model_path": Path("world.pt"),
                        "model_id": "world_model",
                        "open_vocabulary_classes": [
                            {
                                "task_group": "prohibited_items",
                                "class_id": 0,
                                "class_name": "skateboard",
                                "prompts": ["skateboard"],
                                "visual_description": "平板式板面下方有轮子。",
                            }
                        ],
                    },
                    {
                        "id": "disabled_world",
                        "enabled": False,
                        "type": "detection",
                        "task_group": "object_detection",
                        "backend": "yolo_world",
                        "model_id": "unused",
                        "open_vocabulary_classes": [
                            {
                                "task_group": "prohibited_items",
                                "class_id": 0,
                                "class_name": "roller_skates",
                                "prompts": ["roller skates"],
                                "visual_description": "穿在脚上的带轮鞋。",
                            }
                        ],
                    },
                ]
            }
        )

        guide = build_visual_class_guide(config)

        self.assertIn("skateboard", guide["prohibited_items"])
        self.assertNotIn("roller_skates", guide["prohibited_items"])

    def test_all_modules_failure(self) -> None:
        modules = [
            DetectionModule("prohibited", "prohibited", "bad1", FakeBackend("bad1", error=RuntimeError("one"))),
            DetectionModule("garbage", "garbage", "bad2", FakeBackend("bad2", error=RuntimeError("two"))),
        ]
        with tempfile.TemporaryDirectory() as directory:
            image_path = write_test_image(Path(directory) / "image.jpg")
            response = RuntimePipeline(make_config(("prohibited", "garbage")), modules).process(image_path)
        self.assertEqual(response.status, "failed")
        self.assertEqual(response.observations, [])
        self.assertEqual(len(response.errors), 2)

    def test_invalid_image_is_failure(self) -> None:
        backend = FakeBackend("unused")
        module = DetectionModule("garbage", "garbage", "unused", backend)
        response = RuntimePipeline(make_config(("garbage",)), [module]).process(Path("missing-image.jpg"))
        self.assertEqual(response.status, "failed")
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
            def review(self, image: ValidatedImage, summary: DetectionSummary) -> VLMReviewResult:
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
        self.assertEqual(response.review.reasons, ["low_confidence", "review_failure"])
        self.assertTrue(any(error.stage == "review" and error.code == "review_failure" for error in response.errors))
        self.assertFalse(any(error.code == "fusion_failure" for error in response.errors))

    def test_full_image_review_and_fusion_preserve_all_sources(self) -> None:
        class SemanticReviewProvider(ReviewProvider):
            def review(self, image: ValidatedImage, summary: DetectionSummary) -> VLMReviewResult:
                self.image_size = image.image.size
                return VLMReviewResult(
                    provider="fake_vlm",
                    model_id="fake-vl",
                    duration_ms=2,
                    decisions=[
                        VLMReviewDecision(
                            observation_id=summary.detections[0].observation_id,
                            verdict="corrected",
                            corrected_task_group="prohibited",
                            corrected_class_id=0,
                            corrected_class_name="prohibited_class",
                        )
                    ],
                    findings=[
                        VLMFinding(
                            id="vlm-0001",
                            task_group="prohibited",
                            class_id=0,
                            class_name="prohibited_class",
                            reasoning="missed object in full image",
                            bbox_normalized_xyxy=(0.6, 0.1, 0.8, 0.3),
                        )
                    ],
                )

        provider = SemanticReviewProvider()
        backend = FakeBackend("good", [BackendDetection(0, "spray_can", 0.9, (10, 10, 30, 40))])
        module = DetectionModule("prohibited", "prohibited", "good", backend)
        with tempfile.TemporaryDirectory() as directory:
            image_path = write_test_image(Path(directory) / "image.jpg")
            response = RuntimePipeline(
                make_config(("prohibited",)),
                [module],
                review_provider=provider,
            ).process(image_path)

        self.assertEqual(provider.image_size, (100, 80))
        self.assertEqual(response.observations[0].class_name, "prohibited_class")
        self.assertEqual(response.observations[0].confidence, 0.9)
        self.assertEqual(response.observations[0].geometry.bbox_xyxy, (10.0, 10.0, 30.0, 40.0))
        self.assertEqual(len(response.review.findings), 1)
        self.assertEqual(
            [decision.action for decision in response.fusion.decisions],
            ["correct_yolo", "add_vlm_finding"],
        )
        self.assertEqual(
            response.fusion.decisions[1].geometry_source,
            "vlm_full_image",
        )

    def test_dual_pass_runs_once_per_pass_and_maps_both_finding_sources(self) -> None:
        class DualPassProvider(ReviewProvider):
            supports_crop_scan = True
            provider_name = "fake_vlm"
            model_id = "fake-vl"

            def __init__(self) -> None:
                self.full_calls = 0
                self.crop_calls = 0
                self.full_image_size = None
                self.crops = []

            def review(self, image: ValidatedImage, summary: DetectionSummary) -> VLMReviewResult:
                self.full_calls += 1
                self.full_image_size = image.image.size
                return VLMReviewResult(
                    provider=self.provider_name,
                    model_id=self.model_id,
                    duration_ms=2,
                    review_pass="full_image",
                    decisions=[
                        VLMReviewDecision(
                            observation_id=item.observation_id,
                            verdict="confirmed",
                        )
                        for item in summary.detections
                    ],
                    findings=[
                        VLMFinding(
                            id="vlm-full-0001",
                            task_group="garbage",
                            class_id=0,
                            class_name="paper",
                            confidence=0.8,
                            bbox_normalized_xyxy=(0.7, 0.1, 0.9, 0.3),
                        )
                    ],
                )

            def review_crops(self, crops, image, summary, full_image_review):
                self.crop_calls += 1
                self.crops = crops
                return VLMReviewResult(
                    provider=self.provider_name,
                    model_id=self.model_id,
                    duration_ms=3,
                    review_pass="crop_scan",
                    findings=[
                        VLMFinding(
                            id="vlm-crop-0001",
                            task_group="prohibited",
                            class_id=0,
                            class_name="prohibited_class",
                            confidence=0.75,
                            bbox_normalized_xyxy=(0.0, 0.6, 0.3, 0.9),
                            crop_id=crops[0].crop_id,
                            review_pass="crop_scan",
                            geometry_source="vlm_crop",
                        )
                    ],
                )

        provider = DualPassProvider()
        backend = FakeBackend(
            "good",
            [BackendDetection(0, "prohibited_class", 0.9, (10, 10, 30, 40))],
        )
        module = DetectionModule("prohibited", "prohibited", "good", backend)
        with tempfile.TemporaryDirectory() as directory:
            image_path = write_test_image(Path(directory) / "image.jpg", size=(100, 80))
            response = RuntimePipeline(
                make_config(("prohibited",)),
                [module],
                review_provider=provider,
            ).process(image_path)

        self.assertEqual(provider.full_calls, 1)
        self.assertEqual(provider.crop_calls, 1)
        self.assertEqual(provider.full_image_size, (100, 80))
        self.assertGreater(len(provider.crops), 1)
        self.assertTrue(all(crop.image.size == (crop.width, crop.height) for crop in provider.crops))
        self.assertEqual(
            [item.pass_id for item in response.review.passes],
            ["full_image", "crop_scan"],
        )
        self.assertEqual(
            {item.metadata.get("geometry_source") for item in response.observations},
            {None, "vlm_full_image", "vlm_crop"},
        )
        self.assertIsNotNone(response.timing_ms.full_image_review)
        self.assertIsNotNone(response.timing_ms.crop_generation)
        self.assertIsNotNone(response.timing_ms.crop_scan_review)

    def test_crop_scan_timeout_returns_partial_success_with_full_image_result(self) -> None:
        class CropTimeoutProvider(ReviewProvider):
            supports_crop_scan = True
            provider_name = "fake_vlm"
            model_id = "fake-vl"

            def review(self, image: ValidatedImage, summary: DetectionSummary) -> VLMReviewResult:
                return VLMReviewResult(
                    provider=self.provider_name,
                    model_id=self.model_id,
                    duration_ms=1,
                    decisions=[
                        VLMReviewDecision(
                            observation_id=item.observation_id,
                            verdict="confirmed",
                        )
                        for item in summary.detections
                    ],
                )

            def review_crops(self, crops, image, summary, full_image_review):
                raise TimeoutError("crop scan timed out")

        backend = FakeBackend(
            "good",
            [BackendDetection(0, "prohibited_class", 0.9, (10, 10, 30, 40))],
        )
        module = DetectionModule("prohibited", "prohibited", "good", backend)
        with tempfile.TemporaryDirectory() as directory:
            image_path = write_test_image(Path(directory) / "image.jpg")
            response = RuntimePipeline(
                make_config(("prohibited",)),
                [module],
                review_provider=CropTimeoutProvider(),
            ).process(image_path)

        self.assertEqual(response.status, "partial_success")
        self.assertEqual(len(response.observations), 1)
        self.assertEqual(response.review.passes[0].status, "completed")
        self.assertEqual(response.review.passes[1].status, "failed")
        self.assertTrue(
            any(error.code == "crop_scan_review_failure" for error in response.errors)
        )

    def test_crop_scan_still_runs_when_full_image_review_fails(self) -> None:
        class FullImageFailureProvider(ReviewProvider):
            supports_crop_scan = True
            provider_name = "fake_vlm"
            model_id = "fake-vl"

            def __init__(self) -> None:
                self.crop_calls = 0

            def review(self, image: ValidatedImage, summary: DetectionSummary) -> VLMReviewResult:
                raise TimeoutError("full-image review timed out")

            def review_crops(self, crops, image, summary, full_image_review):
                self.crop_calls += 1
                return VLMReviewResult(
                    provider=self.provider_name,
                    model_id=self.model_id,
                    duration_ms=2,
                    review_pass="crop_scan",
                    findings=[
                        VLMFinding(
                            id="vlm-crop-0001",
                            task_group="garbage",
                            class_id=0,
                            class_name="paper",
                            confidence=0.8,
                            bbox_normalized_xyxy=(0.1, 0.1, 0.7, 0.7),
                            crop_id=crops[0].crop_id,
                            review_pass="crop_scan",
                            geometry_source="vlm_crop",
                        )
                    ],
                )

        provider = FullImageFailureProvider()
        backend = FakeBackend(
            "good",
            [BackendDetection(0, "prohibited_class", 0.9, (10, 10, 30, 40))],
        )
        module = DetectionModule("prohibited", "prohibited", "good", backend)
        with tempfile.TemporaryDirectory() as directory:
            image_path = write_test_image(Path(directory) / "image.jpg")
            response = RuntimePipeline(
                make_config(("prohibited",)),
                [module],
                review_provider=provider,
            ).process(image_path)

        self.assertEqual(provider.crop_calls, 1)
        self.assertEqual(response.status, "partial_success")
        self.assertEqual(
            [item.status for item in response.review.passes],
            ["failed", "completed"],
        )
        self.assertTrue(
            any(
                item.metadata.get("geometry_source") == "vlm_crop"
                for item in response.observations
            )
        )

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
