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


if __name__ == "__main__":
    unittest.main()
