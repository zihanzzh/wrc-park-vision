from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from PIL import Image

from wrc_park_vision.runtime.config import ReviewProviderSettings
from wrc_park_vision.runtime.schemas import DetectionSummary, DetectionSummaryItem, ValidatedImage
from wrc_park_vision.runtime.vlm.parser import ReviewResponseError, parse_review_response
from wrc_park_vision.runtime.vlm.prompt import build_review_prompt
from wrc_park_vision.runtime.vlm.qwen25 import Qwen25VLProvider


CATALOG = {
    "prohibited_items": ["spray_can", "portable_gas_stove"],
    "garbage": ["plastic_drink_bottle"],
}


def make_summary() -> DetectionSummary:
    return DetectionSummary(
        total_detections=1,
        counts_by_task_group={"prohibited_items": 1},
        detections=[
            DetectionSummaryItem(
                observation_id="obs-0001",
                task_group="prohibited_items",
                class_id=0,
                class_name="spray_can",
                confidence=0.72,
                bbox_xyxy=(1, 2, 30, 40),
                bbox_normalized_xyxy=(0.01, 0.025, 0.3, 0.5),
            )
        ],
    )


class VLMReviewTests(unittest.TestCase):
    def test_prompt_requires_full_image_review_without_localization(self) -> None:
        prompt = build_review_prompt(make_summary(), CATALOG)
        self.assertIn("完整原始图片", prompt)
        self.assertIn("只是 YOLO 检测上下文", prompt)
        self.assertIn("被 YOLO 完全漏掉", prompt)
        self.assertIn("不要输出 bbox", prompt)

    def test_parser_preserves_correction_and_vlm_only_finding(self) -> None:
        content = json.dumps(
            {
                "yolo_reviews": [
                    {
                        "observation_id": "obs-0001",
                        "verdict": "corrected",
                        "corrected_task_group": "prohibited_items",
                        "corrected_class_name": "portable_gas_stove",
                        "confidence": 0.9,
                        "reasoning": "semantic correction",
                    }
                ],
                "new_findings": [
                    {
                        "task_group": "garbage",
                        "class_name": "plastic_drink_bottle",
                        "confidence": 0.8,
                        "reasoning": "missed by YOLO",
                    }
                ],
            }
        )
        decisions, findings = parse_review_response(content, make_summary(), CATALOG)
        self.assertEqual(decisions[0].verdict, "corrected")
        self.assertEqual(findings[0].id, "vlm-0001")
        self.assertIsNone(findings[0].geometry)

    def test_parser_rejects_localization_and_incomplete_coverage(self) -> None:
        with self.assertRaises(ReviewResponseError):
            parse_review_response(
                json.dumps(
                    {
                        "yolo_reviews": [
                            {
                                "observation_id": "obs-0001",
                                "verdict": "confirmed",
                                "bbox": [1, 2, 3, 4],
                            }
                        ],
                        "new_findings": [],
                    }
                ),
                make_summary(),
                CATALOG,
            )
        with self.assertRaisesRegex(ReviewResponseError, "coverage mismatch"):
            parse_review_response('{"yolo_reviews": [], "new_findings": []}', make_summary(), CATALOG)

    def test_qwen_provider_sends_full_image_and_parses_response(self) -> None:
        settings = ReviewProviderSettings(
            enabled=True,
            endpoint="http://localhost:8000/v1/chat/completions",
            model_id="Qwen2.5-VL",
        )
        provider = Qwen25VLProvider(settings, CATALOG)
        image = ValidatedImage("image.jpg", Image.new("RGB", (100, 80), "white"), 100, 80)
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "yolo_reviews": [
                                    {"observation_id": "obs-0001", "verdict": "confirmed"}
                                ],
                                "new_findings": [],
                            }
                        )
                    }
                }
            ]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return None

            def read(self) -> bytes:
                return json.dumps(response_payload).encode("utf-8")

        captured = {}

        def fake_urlopen(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = provider.review(image, make_summary())

        content = captured["body"]["messages"][0]["content"]
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertIn("完整原始图片", content[1]["text"])
        self.assertEqual(captured["timeout"], 10.0)
        self.assertEqual(result.decisions[0].verdict, "confirmed")


if __name__ == "__main__":
    unittest.main()
