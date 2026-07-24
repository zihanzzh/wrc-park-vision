from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wrc_park_vision.runtime.config import ConfigError, load_runtime_config


class ConfigTests(unittest.TestCase):
    def test_environment_expansion_and_disabled_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.pt"
            model.touch()
            config_path = root / "runtime.yaml"
            config_path.write_text(
                """
modules:
  - id: active
    enabled: true
    type: detection
    task_group: garbage
    backend: ultralytics
    model_path: ${ACTIVE_MODEL}
    model_id: active_model
    expected_class_names: [bottle]
  - id: disabled
    enabled: false
    type: detection
    task_group: disabled
    backend: ultralytics
    model_path: ${UNSET_DISABLED_MODEL}
    model_id: disabled_model
""",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"ACTIVE_MODEL": str(model)}, clear=False):
                config = load_runtime_config(config_path)
            self.assertEqual(config.modules[0].model_path, model)
            self.assertFalse(config.modules[1].enabled)

    def test_missing_enabled_environment_variable_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "runtime.yaml"
            config_path.write_text(
                """
modules:
  - id: active
    enabled: true
    type: detection
    task_group: garbage
    backend: ultralytics
    model_path: ${DEFINITELY_MISSING_MODEL}
    model_id: active_model
    expected_class_names: [bottle]
""",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ConfigError, "DEFINITELY_MISSING_MODEL"):
                    load_runtime_config(config_path)

    def test_missing_model_file_fails_startup_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "runtime.yaml"
            config_path.write_text(
                """
modules:
  - id: active
    enabled: true
    type: detection
    task_group: garbage
    backend: ultralytics
    model_path: missing.pt
    model_id: active_model
    expected_class_names: [bottle]
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "does not exist"):
                load_runtime_config(config_path)

    def test_enabled_detection_requires_expected_class_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "runtime.yaml"
            config_path.write_text(
                """
modules:
  - id: active
    enabled: true
    type: detection
    task_group: garbage
    backend: ultralytics
    model_path: model.pt
    model_id: active_model
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "expected_class_names"):
                load_runtime_config(config_path, validate_model_paths=False)

    def test_expected_class_names_reject_blank_and_duplicate_names(self) -> None:
        for names, message in (("['']", "blank"), ("[bottle, bottle]", "duplicates")):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                config_path = Path(directory) / "runtime.yaml"
                config_path.write_text(
                    f"""
modules:
  - id: active
    enabled: true
    type: detection
    task_group: garbage
    backend: ultralytics
    model_path: model.pt
    model_id: active_model
    expected_class_names: {names}
""",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ConfigError, message):
                    load_runtime_config(config_path, validate_model_paths=False)

    def test_enabled_behavior_does_not_require_expected_class_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "runtime.yaml"
            config_path.write_text(
                """
modules:
  - id: behavior
    enabled: true
    type: behavior
    task_group: uncivilized_behavior
    backend: ultralytics
    model_id: behavior_future
""",
                encoding="utf-8",
            )
            config = load_runtime_config(config_path, validate_model_paths=False)
        self.assertIsNone(config.modules[0].expected_class_names)

    def test_yolo_world_requires_grouped_open_vocabulary_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "runtime.yaml"
            config_path.write_text(
                """
modules:
  - id: world
    enabled: true
    type: detection
    task_group: object_detection
    backend: yolo_world
    model_path: world.pt
    model_id: world_model
    expected_class_names: [spray_can, plastic_drink_bottle]
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "open_vocabulary_classes instead"):
                load_runtime_config(config_path, validate_model_paths=False)

    def test_yolo_world_validates_group_local_class_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "runtime.yaml"
            config_path.write_text(
                """
modules:
  - id: world
    enabled: true
    type: detection
    task_group: object_detection
    backend: yolo_world
    model_path: world.pt
    model_id: world_model
    open_vocabulary_classes:
      - task_group: prohibited_items
        class_id: 1
        class_name: spray_can
        prompts: [spray can]
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "continuous from 0"):
                load_runtime_config(config_path, validate_model_paths=False)

    def test_yolo_world_rejects_behavior_actions_as_object_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "runtime.yaml"
            config_path.write_text(
                """
modules:
  - id: world
    enabled: true
    type: detection
    task_group: object_detection
    backend: yolo_world
    model_path: world.pt
    model_id: world_model
    open_vocabulary_classes:
      - task_group: uncivilized_behavior
        class_id: 0
        class_name: smoking
        prompts: [smoking]
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "Behavior Pipeline"):
                load_runtime_config(config_path, validate_model_paths=False)

    def test_yolo_world_accepts_prohibited_and_behavior_object_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "runtime.yaml"
            config_path.write_text(
                """
modules:
  - id: world
    enabled: true
    type: detection
    task_group: object_detection
    backend: yolo_world
    model_path: world.pt
    model_id: world_model
    open_vocabulary_classes:
      - task_group: prohibited_items
        class_id: 0
        class_name: spray_can
        prompts: [spray can, aerosol can]
      - task_group: uncivilized_behavior
        class_id: 0
        class_name: person
        prompts: [person]
""",
                encoding="utf-8",
            )
            config = load_runtime_config(config_path, validate_model_paths=False)

        classes = config.modules[0].open_vocabulary_classes
        self.assertIsNotNone(classes)
        self.assertEqual([item.task_group for item in classes or []], [
            "prohibited_items",
            "uncivilized_behavior",
        ])
        self.assertTrue(all(item.visual_description is None for item in classes or []))
        self.assertTrue(all(item.distinguishing_rules == [] for item in classes or []))

    def test_yolo_world_rejects_garbage_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "runtime.yaml"
            config_path.write_text(
                """
modules:
  - id: world
    enabled: true
    type: detection
    task_group: object_detection
    backend: yolo_world
    model_path: world.pt
    model_id: world_model
    open_vocabulary_classes:
      - task_group: garbage
        class_id: 0
        class_name: plastic_drink_bottle
        prompts: [plastic drink bottle]
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "must not provide garbage detections"):
                load_runtime_config(config_path, validate_model_paths=False)

    def test_yolo_world_loads_optional_visual_class_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "runtime.yaml"
            config_path.write_text(
                """
modules:
  - id: world
    enabled: true
    type: detection
    task_group: object_detection
    backend: yolo_world
    model_path: world.pt
    model_id: world_model
    open_vocabulary_classes:
      - task_group: prohibited_items
        class_id: 0
        class_name: skateboard
        prompts: [skateboard]
        visual_description: "平板式板面下方有轮子。"
        distinguishing_rules:
          - "没有直立转向杆或车把。"
""",
                encoding="utf-8",
            )
            config = load_runtime_config(config_path, validate_model_paths=False)

        item = config.modules[0].open_vocabulary_classes[0]  # type: ignore[index]
        self.assertEqual(item.visual_description, "平板式板面下方有轮子。")
        self.assertEqual(item.distinguishing_rules, ["没有直立转向杆或车把。"])

    def test_yolo_world_example_keeps_canonical_class_mapping(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WRC_YOLO_WORLD_MODEL_PATH": "weights/yolov8s-worldv2.pt",
                "WRC_GARBAGE_MODEL_PATH": "weights/garbage_best.pt",
            },
            clear=False,
        ):
            config = load_runtime_config(
                Path("configs/runtime.yolo-world.example.yaml"),
                validate_model_paths=False,
            )

        world = next(module for module in config.modules if module.id == "yolo_world_objects")
        garbage = next(module for module in config.modules if module.id == "garbage")
        classes = world.open_vocabulary_classes or []
        grouped = {
            task_group: [
                (item.class_id, item.class_name)
                for item in classes
                if item.task_group == task_group
            ]
            for task_group in ("prohibited_items", "uncivilized_behavior")
        }
        self.assertEqual(
            grouped,
            {
                "prohibited_items": [
                    (0, "spray_can"),
                    (1, "portable_gas_stove"),
                    (2, "megaphone"),
                    (3, "skateboard"),
                    (4, "kick_scooter"),
                    (5, "speaker"),
                    (6, "roller_skates"),
                    (7, "barbecue_grill"),
                ],
                "uncivilized_behavior": [
                    (0, "person"),
                    (1, "bench"),
                    (2, "grass"),
                    (3, "cigarette"),
                    (4, "vehicle"),
                ],
            },
        )
        self.assertEqual(garbage.backend, "ultralytics")
        self.assertEqual(garbage.task_group, "garbage")
        self.assertEqual(
            garbage.expected_class_names,
            [
                "crumpled_paper_ball",
                "disposable_food_container",
                "empty_cigarette_box",
                "plastic_drink_bottle",
                "plastic_food_wrapper",
                "rigid_takeout_bag",
            ],
        )

    def test_runtime_example_uses_split_detection_modules(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WRC_YOLO_WORLD_MODEL_PATH": "weights/yolov8s-worldv2.pt",
                "WRC_GARBAGE_MODEL_PATH": "weights/garbage_best.pt",
            },
            clear=False,
        ):
            config = load_runtime_config(
                Path("configs/runtime.example.yaml"),
                validate_model_paths=False,
            )

        self.assertEqual(
            [(module.id, module.backend, module.task_group) for module in config.modules],
            [
                ("yolo_world_objects", "yolo_world", "object_detection"),
                ("garbage", "ultralytics", "garbage"),
            ],
        )

    def test_legacy_single_ultralytics_module_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "runtime.yaml"
            config_path.write_text(
                """
modules:
  - id: prohibited
    enabled: true
    type: detection
    task_group: prohibited_items
    backend: ultralytics
    model_path: prohibited.pt
    model_id: prohibited_yolo11m
    expected_class_names: [spray_can]
""",
                encoding="utf-8",
            )
            config = load_runtime_config(config_path, validate_model_paths=False)

        self.assertEqual(len(config.modules), 1)
        self.assertEqual(config.modules[0].expected_class_names, ["spray_can"])

    def test_enabled_review_provider_requires_endpoint_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "runtime.yaml"
            config_path.write_text(
                """
modules:
  - id: active
    enabled: true
    type: detection
    task_group: garbage
    backend: ultralytics
    model_path: model.pt
    model_id: active_model
    expected_class_names: [bottle]
review:
  provider:
    enabled: true
    type: qwen2_5_vl
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "endpoint and model_id"):
                load_runtime_config(config_path, validate_model_paths=False)

    def test_review_failure_and_uncertain_default_to_keep_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "runtime.yaml"
            config_path.write_text(
                """
modules:
  - id: active
    enabled: true
    type: detection
    task_group: garbage
    backend: ultralytics
    model_path: model.pt
    model_id: active_model
    expected_class_names: [bottle]
""",
                encoding="utf-8",
            )
            config = load_runtime_config(config_path, validate_model_paths=False)

        self.assertEqual(config.review.uncertain_policy, "keep_flagged")
        self.assertEqual(config.review.review_failure_policy, "keep_flagged")
        self.assertEqual(config.fusion.uncertain_policy, "keep_flagged")
        self.assertEqual(config.fusion.review_failure_policy, "keep_flagged")
        self.assertTrue(config.review.full_image.enabled)
        self.assertTrue(config.review.crop_scan.enabled)
        self.assertEqual(config.review.crop_scan.overlap, 0.20)
        self.assertEqual(config.fusion.finding_iou_threshold, 0.65)

    def test_legacy_review_fusion_policies_are_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "runtime.yaml"
            config_path.write_text(
                """
modules:
  - id: active
    enabled: true
    type: detection
    task_group: garbage
    backend: ultralytics
    model_path: model.pt
    model_id: active_model
    expected_class_names: [bottle]
review:
  uncertain_policy: drop
  review_failure_policy: drop_review_required
""",
                encoding="utf-8",
            )
            config = load_runtime_config(config_path, validate_model_paths=False)

        self.assertEqual(config.fusion.uncertain_policy, "drop")
        self.assertEqual(
            config.fusion.review_failure_policy,
            "drop_review_required",
        )

    def test_invalid_crop_scan_configuration_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "runtime.yaml"
            config_path.write_text(
                """
modules:
  - id: active
    enabled: true
    type: detection
    task_group: garbage
    backend: ultralytics
    model_path: model.pt
    model_id: active_model
    expected_class_names: [bottle]
review:
  crop_scan:
    enabled: true
    overlap: 1.0
    aspect_ratio_threshold: 1.0
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "invalid runtime config"):
                load_runtime_config(config_path, validate_model_paths=False)

    def test_examples_enable_dual_review_passes_and_fusion_dedup(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WRC_YOLO_WORLD_MODEL_PATH": "weights/yolov8s-worldv2.pt",
                "WRC_GARBAGE_MODEL_PATH": "weights/garbage_best.pt",
            },
            clear=False,
        ):
            for path in (
                Path("configs/runtime.example.yaml"),
                Path("configs/runtime.yolo-world.example.yaml"),
            ):
                with self.subTest(path=path):
                    config = load_runtime_config(
                        path,
                        validate_model_paths=False,
                    )
                    self.assertTrue(config.review.require_finding_bbox)
                    self.assertEqual(config.review.full_image.timeout_seconds, 5)
                    self.assertEqual(config.review.full_image.max_tokens, 1200)
                    self.assertEqual(config.review.crop_scan.timeout_seconds, 5)
                    self.assertEqual(config.review.crop_scan.max_tokens, 700)
                    self.assertEqual(config.review.crop_scan.square_rows, 2)
                    self.assertEqual(config.review.crop_scan.square_columns, 2)
                    self.assertEqual(config.review.crop_scan.wide_columns, 3)
                    self.assertEqual(config.review.crop_scan.tall_rows, 3)
                    self.assertEqual(config.fusion.finding_iou_threshold, 0.65)

    def test_behavior_pipeline_requires_four_canonical_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "runtime.yaml"
            config_path.write_text(
                """
modules:
  - id: active
    enabled: true
    type: detection
    task_group: garbage
    backend: ultralytics
    model_path: model.pt
    model_id: active_model
    expected_class_names: [bottle]
behavior:
  enabled: true
  classes:
    - class_id: 0
      class_name: trampling_grass
      required_object_classes: [person, grass]
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "four canonical classes"):
                load_runtime_config(config_path, validate_model_paths=False)


if __name__ == "__main__":
    unittest.main()
