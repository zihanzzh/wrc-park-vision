from __future__ import annotations

import unittest

from wrc_park_vision.runtime.backends.ultralytics import validate_model_class_names


class ModelClassMappingTests(unittest.TestCase):
    def assert_mapping_error(self, names: object, expected: list[str], message: str) -> None:
        with self.assertRaisesRegex(ValueError, message) as raised:
            validate_model_class_names(
                names,
                expected,
                module_id="garbage",
                model_id="garbage_yolo11m",
            )
        error = str(raised.exception)
        self.assertIn("module=garbage", error)
        self.assertIn("model=garbage_yolo11m", error)
        self.assertIn("expected=", error)
        self.assertIn("actual=", error)

    def test_dict_names_match(self) -> None:
        actual = validate_model_class_names(
            {0: "paper", 1: "bottle"},
            ["paper", "bottle"],
            module_id="garbage",
            model_id="garbage_yolo11m",
        )
        self.assertEqual(actual, ["paper", "bottle"])

    def test_list_names_match(self) -> None:
        actual = validate_model_class_names(
            ["paper", "bottle"],
            ["paper", "bottle"],
            module_id="garbage",
            model_id="garbage_yolo11m",
        )
        self.assertEqual(actual, ["paper", "bottle"])

    def test_tuple_names_match(self) -> None:
        actual = validate_model_class_names(
            ("paper", "bottle"),
            ["paper", "bottle"],
            module_id="garbage",
            model_id="garbage_yolo11m",
        )
        self.assertEqual(actual, ["paper", "bottle"])

    def test_class_count_mismatch(self) -> None:
        self.assert_mapping_error(["paper"], ["paper", "bottle"], "class mapping mismatch")

    def test_class_name_mismatch(self) -> None:
        self.assert_mapping_error(["paper", "can"], ["paper", "bottle"], "class mapping mismatch")

    def test_class_order_mismatch(self) -> None:
        self.assert_mapping_error(["bottle", "paper"], ["paper", "bottle"], "class mapping mismatch")

    def test_non_continuous_class_ids(self) -> None:
        self.assert_mapping_error({0: "paper", 2: "bottle"}, ["paper", "bottle"], "continuous from 0")


if __name__ == "__main__":
    unittest.main()
