from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wrc_park_vision.runtime.backends.base import BackendDetection
from wrc_park_vision.runtime.competition import build_competition_response
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
    ReviewIssue,
    VLMFinding,
    VLMRequestMetrics,
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
                        "visual_class_guidance": {
                            "crumpled_paper_ball": {
                                "visual_description": "可见纸质折皱和不规则团状结构。",
                                "distinguishing_rules": ["草地纹理不是纸团。"],
                            }
                        },
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
                },
                "garbage": {
                    "crumpled_paper_ball": {
                        "visual": "可见纸质折皱和不规则团状结构。",
                        "distinguish": ["草地纹理不是纸团。"],
                    }
                },
            },
        )

    def test_accuracy_policy_rejects_high_confidence_objects_in_one_request(self) -> None:
        class RejectingProvider(ReviewProvider):
            def __init__(self) -> None:
                self.calls = 0
                self.required_task_groups = set()

            def review(self, image, summary):
                raise AssertionError("V3 pipeline must use review_multi_image")

            def review_multi_image(self, request):
                self.calls += 1
                required_ids = set(request.required_review_observation_ids)
                self.required_task_groups = {
                    detection.task_group
                    for detection in request.summary.detections
                    if detection.observation_id in required_ids
                }
                return VLMReviewResult(
                    provider="fake_vlm",
                    model_id="qwen-vl",
                    duration_ms=1,
                    review_pass="multi_image",
                    decisions=[
                        VLMReviewDecision(
                            observation_id=observation_id,
                            verdict="rejected",
                            confidence=0.99,
                        )
                        for observation_id in request.required_review_observation_ids
                    ],
                )

        config = make_config(
            ("garbage", "prohibited_items", "uncivilized_behavior")
        )
        config.review.candidate_selection.review_all_task_groups = [
            "garbage",
            "prohibited_items",
        ]
        provider = RejectingProvider()
        modules = [
            DetectionModule(
                "garbage",
                "garbage",
                "garbage",
                FakeBackend(
                    "garbage",
                    [
                        BackendDetection(
                            0,
                            "crumpled_paper_ball",
                            0.97,
                            (10, 10, 40, 40),
                        )
                    ],
                ),
            ),
            DetectionModule(
                "prohibited_items",
                "prohibited_items",
                "world",
                FakeBackend(
                    "world",
                    [BackendDetection(0, "spray_can", 0.96, (50, 10, 80, 40))],
                ),
            ),
            DetectionModule(
                "uncivilized_behavior",
                "uncivilized_behavior",
                "world",
                FakeBackend(
                    "world",
                    [BackendDetection(0, "person", 0.99, (10, 45, 90, 75))],
                ),
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            response = RuntimePipeline(
                config,
                modules,
                review_provider=provider,
            ).process(write_test_image(Path(directory) / "image.jpg"))

        self.assertEqual(provider.calls, 1)
        self.assertEqual(
            provider.required_task_groups,
            {"garbage", "prohibited_items"},
        )
        self.assertEqual(
            [(item.task_group, item.class_name) for item in response.observations],
            [("uncivilized_behavior", "person")],
        )
        self.assertEqual(response.fusion.decisions[0].action, "reject_yolo")
        self.assertEqual(response.fusion.decisions[1].action, "reject_yolo")
        self.assertEqual(response.fusion.decisions[2].action, "keep_detector_result")
        competition = build_competition_response(response).model_dump()
        self.assertEqual(competition["schema_version"], "1.0")

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
        self.assertTrue(
            any(
                error.stage == "multi_image_review"
                and error.code == "multi_image_review_failure"
                for error in response.errors
            )
        )
        self.assertFalse(any(error.code == "fusion_failure" for error in response.errors))

    def test_failed_compact_fallback_uses_review_failure_policy(self) -> None:
        class FailedFallbackProvider(ReviewProvider):
            def review(self, image: ValidatedImage, summary: DetectionSummary) -> VLMReviewResult:
                error = RuntimeError("compact fallback failed")
                error.metrics = VLMRequestMetrics(
                    request_count=2,
                    fallback_attempted=True,
                    fallback_reason="response_truncated",
                    fallback_max_tokens=1800,
                )
                error.raw_response_debug = "primary and fallback raw responses"
                raise error

        module = DetectionModule(
            "prohibited",
            "prohibited_items",
            "good",
            FakeBackend(
                "good",
                [BackendDetection(0, "spray_can", 0.3, (10, 10, 30, 40))],
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            response = RuntimePipeline(
                make_config(("prohibited",)),
                [module],
                review_provider=FailedFallbackProvider(),
            ).process(write_test_image(Path(directory) / "image.jpg"))

        self.assertEqual(response.status, "partial_success")
        self.assertEqual(response.fusion.decisions[0].action, "keep_review_failed")
        self.assertEqual(
            [(item.mode, item.status) for item in response.review.passes],
            [("primary", "failed"), ("compact_fallback", "failed")],
        )
        self.assertEqual(
            response.review.raw_response_debug,
            "primary and fallback raw responses",
        )

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

    def test_multi_image_review_runs_once_with_original_and_important_crops(self) -> None:
        class MultiImageProvider(ReviewProvider):
            provider_name = "fake_vlm"
            model_id = "fake-vl"

            def __init__(self) -> None:
                self.calls = 0
                self.request = None

            def review(self, image: ValidatedImage, summary: DetectionSummary) -> VLMReviewResult:
                raise AssertionError("V3 pipeline must use review_multi_image")

            def review_multi_image(self, request):
                self.calls += 1
                self.request = request
                return VLMReviewResult(
                    provider=self.provider_name,
                    model_id=self.model_id,
                    duration_ms=3,
                    review_pass="multi_image",
                    decisions=[
                        VLMReviewDecision(
                            observation_id=item.observation_id,
                            verdict="confirmed",
                        )
                        for item in request.summary.detections
                    ],
                    findings=[
                        VLMFinding(
                            id="vlm-multi-0001",
                            task_group="garbage",
                            class_id=0,
                            class_name="paper",
                            confidence=0.8,
                            bbox_normalized_xyxy=(0.7, 0.1, 0.9, 0.3),
                            crop_id=(
                                request.crops[0].crop_id if request.crops else None
                            ),
                            review_pass="multi_image",
                            geometry_source="vlm_multi_image",
                        )
                    ],
                )

        provider = MultiImageProvider()
        backend = FakeBackend(
            "good",
            [BackendDetection(0, "prohibited_class", 0.3, (10, 10, 30, 40))],
        )
        module = DetectionModule("prohibited", "prohibited", "good", backend)
        with tempfile.TemporaryDirectory() as directory:
            image_path = write_test_image(Path(directory) / "image.jpg", size=(100, 80))
            response = RuntimePipeline(
                make_config(("prohibited",)),
                [module],
                review_provider=provider,
            ).process(image_path)

        self.assertEqual(provider.calls, 1)
        self.assertIsNotNone(provider.request)
        self.assertEqual(provider.request.image.image.size, (100, 80))
        self.assertEqual(len(provider.request.candidates), 1)
        self.assertEqual(len(provider.request.crops), 1)
        self.assertEqual(
            [item.pass_id for item in response.review.passes],
            ["multi_image"],
        )
        finding = next(
            item for item in response.observations if item.source.module_id == "vlm_review"
        )
        self.assertEqual(finding.metadata["geometry_source"], "vlm_multi_image")
        self.assertEqual(finding.geometry.bbox_xyxy, (70.0, 8.0, 90.0, 24.0))
        self.assertIsNotNone(response.timing_ms.candidate_selection)
        self.assertIsNotNone(response.timing_ms.crop_generation)
        self.assertIsNotNone(response.timing_ms.multi_image_review)

    def test_multi_image_timeout_returns_partial_success(self) -> None:
        class MultiImageTimeoutProvider(ReviewProvider):
            def review(self, image: ValidatedImage, summary: DetectionSummary) -> VLMReviewResult:
                raise AssertionError("V3 pipeline must use review_multi_image")

            def review_multi_image(self, request):
                raise TimeoutError("multi-image review timed out")

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
                review_provider=MultiImageTimeoutProvider(),
            ).process(image_path)

        self.assertEqual(response.status, "partial_success")
        self.assertEqual(len(response.observations), 1)
        self.assertEqual(response.review.passes[0].status, "failed")
        self.assertTrue(
            any(error.code == "multi_image_review_failure" for error in response.errors)
        )
        self.assertTrue(build_competition_response(response).degraded)

    def test_exhausted_budget_skips_vlm_and_returns_fused_result(self) -> None:
        class CountingProvider(ReviewProvider):
            def __init__(self) -> None:
                self.calls = 0

            def review(self, image, summary):
                raise AssertionError("V3 pipeline must use review_multi_image")

            def review_multi_image(self, request):
                self.calls += 1
                raise AssertionError("provider must not be called")

        config = make_config(("prohibited",))
        config.runtime.total_timeout_seconds = 0.01
        config.review.reserve_seconds_for_fusion_and_output = 0.02
        provider = CountingProvider()
        module = DetectionModule(
            "prohibited",
            "prohibited",
            "good",
            FakeBackend(
                "good",
                [BackendDetection(0, "prohibited_class", 0.3, (10, 10, 30, 40))],
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            response = RuntimePipeline(
                config,
                [module],
                review_provider=provider,
            ).process(write_test_image(Path(directory) / "image.jpg"))

        self.assertEqual(provider.calls, 0)
        self.assertEqual(response.status, "partial_success")
        self.assertEqual(len(response.observations), 1)
        self.assertEqual(response.review.status, "failed")
        self.assertEqual(
            response.fusion.decisions[0].action,
            "keep_review_failed",
        )
        self.assertTrue(
            any(error.code == "review_deadline_exhausted" for error in response.errors)
        )

    def test_vlm_timeout_is_capped_by_remaining_runtime_budget(self) -> None:
        class TimeoutCapturingProvider(ReviewProvider):
            def __init__(self) -> None:
                self.timeout_seconds = None

            def review(self, image, summary):
                raise AssertionError("V3 pipeline must use review_multi_image")

            def review_multi_image(self, request):
                self.timeout_seconds = request.timeout_seconds
                return VLMReviewResult(
                    provider="fake_vlm",
                    model_id="fake-vl",
                    duration_ms=1,
                    review_pass="multi_image",
                    decisions=[
                        VLMReviewDecision(
                            observation_id=request.required_review_observation_ids[0],
                            verdict="confirmed",
                        )
                    ],
                )

        config = make_config(("prohibited",))
        config.runtime.total_timeout_seconds = 1.0
        config.review.reserve_seconds_for_fusion_and_output = 0.2
        provider = TimeoutCapturingProvider()
        module = DetectionModule(
            "prohibited",
            "prohibited",
            "good",
            FakeBackend(
                "good",
                [BackendDetection(0, "prohibited_class", 0.3, (10, 10, 30, 40))],
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            response = RuntimePipeline(
                config,
                [module],
                review_provider=provider,
            ).process(write_test_image(Path(directory) / "image.jpg"))

        self.assertIsNotNone(provider.timeout_seconds)
        self.assertGreater(provider.timeout_seconds, 0.0)
        self.assertLessEqual(provider.timeout_seconds, 0.8)
        self.assertEqual(response.review.status, "completed")
        self.assertEqual(response.status, "success")

    def test_missing_required_review_item_returns_partial_success(self) -> None:
        class MissingItemProvider(ReviewProvider):
            def review(self, image, summary):
                raise AssertionError("V3 pipeline must use review_multi_image")

            def review_multi_image(self, request):
                observation_id = request.required_review_observation_ids[0]
                return VLMReviewResult(
                    provider="fake_vlm",
                    model_id="fake-vl",
                    duration_ms=1,
                    review_pass="multi_image",
                    issues=[
                        ReviewIssue(
                            section="yolo_reviews",
                            code="missing_observation_review",
                            message=f"missing review for {observation_id}",
                            observation_id=observation_id,
                            review_pass="multi_image",
                        )
                    ],
                )

        module = DetectionModule(
            "prohibited",
            "prohibited",
            "good",
            FakeBackend(
                "good",
                [BackendDetection(0, "prohibited_class", 0.3, (10, 10, 30, 40))],
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            response = RuntimePipeline(
                make_config(("prohibited",)),
                [module],
                review_provider=MissingItemProvider(),
            ).process(write_test_image(Path(directory) / "image.jpg"))

        self.assertEqual(response.status, "partial_success")
        self.assertEqual(
            response.fusion.decisions[0].action,
            "keep_review_failed",
        )
        self.assertTrue(
            any(
                error.code == "required_review_items_missing"
                for error in response.errors
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
