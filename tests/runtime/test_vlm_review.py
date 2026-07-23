from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from PIL import Image

from wrc_park_vision.runtime.config import ReviewProviderSettings
from wrc_park_vision.runtime.schemas import (
    BehaviorCandidate,
    BehaviorClassSummary,
    DetectionSummary,
    DetectionSummaryItem,
    ValidatedImage,
)
from wrc_park_vision.runtime.vlm.parser import ReviewResponseError, parse_review_response
from wrc_park_vision.runtime.vlm.prompt import build_review_prompt
from wrc_park_vision.runtime.vlm.qwen25 import Qwen25VLProvider


CATALOG = {
    "prohibited_items": ["spray_can", "portable_gas_stove", "speaker"],
    "garbage": ["plastic_drink_bottle"],
}

VISUAL_TEST_CATALOG = {
    "prohibited_items": [
        "portable_gas_stove",
        "skateboard",
        "kick_scooter",
        "barbecue_grill",
    ],
    "garbage": [
        "crumpled_paper_ball",
        "plastic_drink_bottle",
        "plastic_food_wrapper",
    ],
}

VISUAL_TEST_GUIDE = {
    "prohibited_items": {
        "portable_gas_stove": {
            "visual": "小型桌面炉具，可见燃烧器、锅架或气罐舱。",
            "distinguish": ["不是带烤网的开放式 barbecue_grill。"],
        },
        "skateboard": {
            "visual": "平板式板面下方有轮子。",
            "distinguish": ["没有直立转向杆或车把。"],
        },
        "kick_scooter": {
            "visual": "站立踏板连接直立转向杆，顶部有车把。",
            "distinguish": ["不是无车把的平板 skateboard。"],
        },
        "barbecue_grill": {
            "visual": "开放式烧烤炉体，可见烤网或炭火结构。",
            "distinguish": ["不是小型桌面 portable_gas_stove。"],
        },
        "roller_skates": {
            "visual": "穿在脚上的带轮鞋。",
        },
    },
    "garbage": {
        "crumpled_paper_ball": {
            "visual": "明显揉皱成团的纸。",
            "distinguish": ["不是压扁塑料瓶或薄塑料包装。"],
        },
        "plastic_drink_bottle": {
            "visual": "塑料饮料瓶，可见瓶口、瓶盖、瓶身或标签结构。",
            "distinguish": ["压扁后仍按瓶体结构识别，不是 crumpled_paper_ball。"],
        },
        "plastic_food_wrapper": {
            "visual": "薄而柔软的食品塑料包装，常见封边或印刷包装形态。",
            "distinguish": ["不是纸团。"],
        },
    },
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
    def test_prompt_uses_real_enums_without_json_placeholders(self) -> None:
        prompt = build_review_prompt(make_summary(), CATALOG)
        self.assertIn("检查完整图片", prompt)
        self.assertIn("prohibited_items, garbage, uncivilized_behavior", prompt)
        self.assertIn("trampling_grass, smoking, blocking_fire_lane, standing_or_lying_on_bench", prompt)
        self.assertIn('"observation_id":"obs-0001"', prompt)
        self.assertIn('"verdict":"uncertain"', prompt)
        self.assertIn('"speaker"', prompt)
        self.assertIn("behavior_candidates", prompt)
        self.assertNotIn("允许的 task_group", prompt)
        self.assertNotIn("允许的 class_name", prompt)
        self.assertNotIn("合法类别", prompt)
        self.assertNotIn("类别名称", prompt)
        self.assertNotIn("填写", prompt)
        self.assertIn("只输出一个 JSON object", prompt)

    def test_prompt_uses_compact_visual_guide_for_enabled_classes(self) -> None:
        prompt = build_review_prompt(
            make_summary(),
            VISUAL_TEST_CATALOG,
            visual_class_guide=VISUAL_TEST_GUIDE,
        )

        self.assertIn("平板式板面下方有轮子", prompt)
        self.assertIn("没有直立转向杆或车把", prompt)
        self.assertIn("站立踏板连接直立转向杆，顶部有车把", prompt)
        self.assertIn("燃烧器、锅架或气罐舱", prompt)
        self.assertIn("开放式烧烤炉体", prompt)
        self.assertIn("明显揉皱成团的纸", prompt)
        self.assertIn("瓶口、瓶盖、瓶身或标签结构", prompt)
        self.assertIn("薄而柔软的食品塑料包装", prompt)
        self.assertNotIn("roller_skates", prompt)
        self.assertNotIn("允许的 task_group", prompt)
        self.assertNotIn("允许的 class_name", prompt)

    def test_parser_accepts_minimal_7b_response_and_empty_reasoning(self) -> None:
        content = json.dumps(
            {
                "yolo_reviews": [
                    {
                        "observation_id": " obs-0001 ",
                        "verdict": "confirmed",
                        "confidence": None,
                        "reasoning": "",
                    }
                ],
                "new_findings": [
                    {
                        "task_group": " garbage ",
                        "class_name": " plastic_drink_bottle ",
                        "confidence": 0.7,
                        "reasoning": None,
                    }
                ],
                "behavior_reviews": [],
            }
        )

        decisions, findings, behaviors = parse_review_response(content, make_summary(), CATALOG)

        self.assertEqual(decisions[0].observation_id, "obs-0001")
        self.assertEqual(decisions[0].reasoning, "")
        self.assertEqual(findings[0].task_group, "garbage")
        self.assertEqual(findings[0].class_name, "plastic_drink_bottle")
        self.assertIsNone(findings[0].reasoning)
        self.assertEqual(behaviors, [])

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
        decisions, findings, behaviors = parse_review_response(content, make_summary(), CATALOG)
        self.assertEqual(decisions[0].verdict, "corrected")
        self.assertEqual(findings[0].id, "vlm-0001")
        self.assertIsNone(findings[0].geometry)
        self.assertEqual(behaviors, [])

    def test_parser_handles_candidate_review_and_full_image_behavior(self) -> None:
        summary = make_summary().model_copy(
            update={
                "behavior_classes": [
                    BehaviorClassSummary(
                        class_id=0,
                        class_name="trampling_grass",
                        required_object_classes=["person", "grass"],
                    ),
                    BehaviorClassSummary(
                        class_id=2,
                        class_name="blocking_fire_lane",
                        required_object_classes=["vehicle"],
                    ),
                ],
                "behavior_candidates": [
                    BehaviorCandidate(
                        id="behavior-candidate-0001",
                        class_id=0,
                        class_name="trampling_grass",
                        evidence_observation_ids=["obs-0001"],
                        evidence_class_names=["person", "grass"],
                    )
                ],
            }
        )
        content = json.dumps(
            {
                "yolo_reviews": [{"observation_id": "obs-0001", "verdict": "confirmed"}],
                "new_findings": [],
                "behavior_reviews": [
                    {
                        "candidate_id": "behavior-candidate-0001",
                        "class_name": "trampling_grass",
                        "verdict": "rejected",
                        "reasoning": "person is beside the grass",
                    },
                    {
                        "candidate_id": None,
                        "class_name": "blocking_fire_lane",
                        "verdict": "confirmed",
                        "confidence": 0.82,
                        "reasoning": "vehicle blocks the marked fire lane",
                    },
                ],
            }
        )

        _, _, behaviors = parse_review_response(content, summary, CATALOG)

        self.assertEqual([item.verdict for item in behaviors], ["rejected", "confirmed"])
        self.assertEqual(behaviors[0].evidence_observation_ids, ["obs-0001"])
        self.assertIsNone(behaviors[1].candidate_id)
        self.assertEqual(behaviors[1].class_id, 2)

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

    def test_parser_still_rejects_invalid_task_group(self) -> None:
        content = json.dumps(
            {
                "yolo_reviews": [{"observation_id": "obs-0001", "verdict": "confirmed"}],
                "new_findings": [
                    {
                        "task_group": "允许的 task_group",
                        "class_name": "speaker",
                        "reasoning": None,
                    }
                ],
                "behavior_reviews": [],
            }
        )

        with self.assertRaisesRegex(ReviewResponseError, "unknown VLM task_group"):
            parse_review_response(content, make_summary(), CATALOG)

    def test_qwen_provider_sends_full_image_and_parses_response(self) -> None:
        settings = ReviewProviderSettings(
            enabled=True,
            endpoint="http://localhost:8000/v1/chat/completions",
            model_id="Qwen2.5-VL",
        )
        provider = Qwen25VLProvider(
            settings,
            CATALOG,
            visual_class_guide={
                "prohibited_items": {
                    "spray_can": {
                        "visual": "带喷嘴或喷头的加压气雾罐。",
                    }
                }
            },
        )
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
            captured["calls"] = captured.get("calls", 0) + 1
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = provider.review(image, make_summary())

        content = captured["body"]["messages"][0]["content"]
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertIn("检查完整图片", content[1]["text"])
        self.assertIn("带喷嘴或喷头的加压气雾罐", content[1]["text"])
        self.assertEqual(captured["calls"], 1)
        self.assertEqual(captured["timeout"], 10.0)
        self.assertEqual(result.decisions[0].verdict, "confirmed")

    def test_qwen_parser_error_contains_truncated_raw_response(self) -> None:
        settings = ReviewProviderSettings(
            enabled=True,
            endpoint="http://localhost:8000/v1/chat/completions",
            model_id="Qwen2.5-VL-7B-Instruct-AWQ",
        )
        provider = Qwen25VLProvider(settings, CATALOG)
        image = ValidatedImage("image.jpg", Image.new("RGB", (100, 80), "white"), 100, 80)
        raw_content = "not-json-start " + ("x" * 1000) + " hidden-tail"
        response_payload = {
            "choices": [{"message": {"content": raw_content}}],
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return None

            def read(self) -> bytes:
                return json.dumps(response_payload).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            with self.assertRaises(ReviewResponseError) as raised:
                provider.review(image, make_summary())

        message = str(raised.exception)
        self.assertIn("raw_response_excerpt='not-json-start", message)
        self.assertIn("...", message)
        self.assertNotIn("hidden-tail", message)
        self.assertLess(len(message), 800)


if __name__ == "__main__":
    unittest.main()
