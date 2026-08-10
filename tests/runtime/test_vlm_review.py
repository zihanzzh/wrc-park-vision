from __future__ import annotations

import base64
import json
import unittest
from io import BytesIO
from unittest.mock import patch

from PIL import Image

from wrc_park_vision.runtime.config import (
    ImportantCropSettings,
    ReviewProviderSettings,
    ReviewSettings,
)
from wrc_park_vision.runtime.review import (
    MultiImageReviewRequest,
    ReviewCandidate,
    generate_important_crops,
)
from wrc_park_vision.runtime.schemas import (
    BehaviorCandidate,
    BehaviorClassSummary,
    DetectionSummary,
    DetectionSummaryItem,
    ValidatedImage,
)
from wrc_park_vision.runtime.vlm.parser import ReviewResponseError, parse_review_response
from wrc_park_vision.runtime.vlm.prompt import (
    build_multi_image_prompt,
    build_review_prompt,
)
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


def make_two_detection_summary() -> DetectionSummary:
    return make_summary().model_copy(
        update={
            "total_detections": 2,
            "detections": [
                *make_summary().detections,
                DetectionSummaryItem(
                    observation_id="obs-0002",
                    task_group="prohibited_items",
                    class_id=2,
                    class_name="speaker",
                    confidence=0.91,
                    bbox_xyxy=(40, 5, 80, 50),
                    bbox_normalized_xyxy=(0.4, 0.0625, 0.8, 0.625),
                ),
            ],
        }
    )


def prompt_output_template(prompt: str) -> dict[str, object]:
    template = prompt.split("最小输出模板：", 1)[1].splitlines()[0]
    return json.loads(template)


class VLMReviewTests(unittest.TestCase):
    def test_prompt_uses_real_enums_without_json_placeholders(self) -> None:
        prompt = build_review_prompt(make_summary(), CATALOG)
        self.assertIn("original_image 是坐标唯一基准", prompt)
        self.assertIn('"prohibited_items","garbage","uncivilized_behavior"', prompt)
        self.assertIn('"trampling_grass","smoking","blocking_fire_lane"', prompt)
        self.assertIn('"id":"obs-0001"', prompt)
        self.assertIn('"verdict":"confirmed"', prompt)
        self.assertIn('"speaker"', prompt)
        self.assertNotIn("允许的 task_group", prompt)
        self.assertNotIn("允许的 class_name", prompt)
        self.assertNotIn("合法类别", prompt)
        self.assertNotIn("类别名称", prompt)
        self.assertNotIn("填写", prompt)
        self.assertIn("只输出一个 JSON object", prompt)
        self.assertIn("uncertain 只用于确实看不清", prompt)
        self.assertIn("普通塑料瓶不是 spray_can", prompt)
        self.assertIn("final visual reasoning judge", prompt)
        self.assertIn("reject or correct", prompt)
        self.assertIn("无论有无 candidate，都主动检查四类行为", prompt)
        self.assertNotIn('"reasoning":null', prompt)
        self.assertNotIn("review_pass=", prompt)
        self.assertNotIn("geometry_source=", prompt)

    def test_prompt_uses_compact_visual_guide_for_enabled_classes(self) -> None:
        summary = DetectionSummary(
            total_detections=4,
            detections=[
                DetectionSummaryItem(
                    observation_id=f"obs-{index:04d}",
                    task_group="prohibited_items",
                    class_id=index - 1,
                    class_name=class_name,
                    confidence=0.5,
                    bbox_xyxy=(10, 10, 30, 30),
                    bbox_normalized_xyxy=(0.1, 0.1, 0.3, 0.3),
                )
                for index, class_name in enumerate(
                    (
                        "portable_gas_stove",
                        "skateboard",
                        "kick_scooter",
                        "barbecue_grill",
                    ),
                    1,
                )
            ],
        )
        prompt = build_review_prompt(
            summary,
            VISUAL_TEST_CATALOG,
            visual_class_guide=VISUAL_TEST_GUIDE,
        )

        self.assertIn("平板式板面下方有轮子", prompt)
        self.assertIn("没有直立转向杆或车把", prompt)
        self.assertIn("站立踏板连接直立转向杆，顶部有车把", prompt)
        self.assertIn("燃烧器、锅架或气罐舱", prompt)
        self.assertIn("开放式烧烤炉体", prompt)
        self.assertIn('"garbage":["crumpled_paper_ball"', prompt)
        self.assertNotIn("明显揉皱成团的纸", prompt)
        self.assertNotIn("roller_skates", prompt)
        self.assertNotIn("允许的 task_group", prompt)
        self.assertNotIn("允许的 class_name", prompt)

    def test_visual_guide_includes_full_required_task_group_for_corrections(
        self,
    ) -> None:
        summary = DetectionSummary(
            total_detections=3,
            detections=[
                DetectionSummaryItem(
                    observation_id="obs-0001",
                    task_group="prohibited_items",
                    class_id=1,
                    class_name="skateboard",
                    confidence=0.3,
                    bbox_xyxy=(10, 10, 30, 30),
                    bbox_normalized_xyxy=(0.1, 0.1, 0.3, 0.3),
                    conflict_observation_ids=["obs-0002"],
                ),
                DetectionSummaryItem(
                    observation_id="obs-0002",
                    task_group="prohibited_items",
                    class_id=2,
                    class_name="kick_scooter",
                    confidence=0.4,
                    bbox_xyxy=(10, 10, 30, 30),
                    bbox_normalized_xyxy=(0.1, 0.1, 0.3, 0.3),
                ),
                DetectionSummaryItem(
                    observation_id="obs-0003",
                    task_group="prohibited_items",
                    class_id=0,
                    class_name="portable_gas_stove",
                    confidence=0.9,
                    bbox_xyxy=(40, 10, 60, 30),
                    bbox_normalized_xyxy=(0.4, 0.1, 0.6, 0.3),
                ),
            ],
        )
        candidate = ReviewCandidate(
            candidate_id="review-candidate-0001",
            bbox_normalized_xyxy=(0.1, 0.1, 0.3, 0.3),
            reasons=("cross_model_conflict",),
            observation_ids=("obs-0001",),
        )
        prompt = build_multi_image_prompt(
            MultiImageReviewRequest(
                image=ValidatedImage(
                    "image.jpg",
                    Image.new("RGB", (100, 80), "white"),
                    100,
                    80,
                ),
                summary=summary,
                candidates=(candidate,),
            ),
            VISUAL_TEST_CATALOG,
            VISUAL_TEST_GUIDE,
        )
        visual_section = prompt.split("视觉指南：", 1)[1].split(
            "\n审核输入：",
            1,
        )[0]

        self.assertIn("平板式板面下方有轮子", visual_section)
        self.assertIn("站立踏板连接直立转向杆", visual_section)
        self.assertIn("燃烧器、锅架或气罐舱", visual_section)
        self.assertIn("开放式烧烤炉体", visual_section)
        self.assertNotIn("明显揉皱成团的纸", visual_section)

    def test_multi_image_prompt_only_templates_candidate_observations(self) -> None:
        summary = make_two_detection_summary()
        candidate = ReviewCandidate(
            candidate_id="review-candidate-0001",
            bbox_normalized_xyxy=(0.01, 0.025, 0.3, 0.5),
            reasons=("low_confidence",),
            observation_ids=("obs-0001",),
        )

        prompt = build_multi_image_prompt(
            MultiImageReviewRequest(
                image=ValidatedImage(
                    "image.jpg",
                    Image.new("RGB", (100, 80), "white"),
                    100,
                    80,
                ),
                summary=summary,
                candidates=(candidate,),
            ),
            CATALOG,
        )
        template = prompt_output_template(prompt)

        self.assertEqual(
            [item["id"] for item in template["yolo_reviews"]],
            ["obs-0001"],
        )
        self.assertNotIn('"id":"obs-0002"', prompt)
        self.assertIn(
            "每个 detection id 在 yolo_reviews 恰好出现一次",
            prompt,
        )

    def test_required_review_observation_ids_are_deduplicated(self) -> None:
        summary = make_two_detection_summary()
        candidates = (
            ReviewCandidate(
                candidate_id="review-candidate-0001",
                bbox_normalized_xyxy=(0.01, 0.025, 0.3, 0.5),
                reasons=("low_confidence",),
                observation_ids=("obs-0001",),
            ),
            ReviewCandidate(
                candidate_id="review-candidate-0002",
                bbox_normalized_xyxy=(0.0, 0.0, 0.5, 0.6),
                reasons=("behavior_candidate",),
                observation_ids=("obs-0001",),
            ),
        )
        request = MultiImageReviewRequest(
            image=ValidatedImage(
                "image.jpg",
                Image.new("RGB", (100, 80), "white"),
                100,
                80,
            ),
            summary=summary,
            candidates=candidates,
        )

        self.assertEqual(request.required_review_observation_ids, ("obs-0001",))
        template = prompt_output_template(build_multi_image_prompt(request, CATALOG))
        self.assertEqual(len(template["yolo_reviews"]), 1)

    def test_behavior_only_prompt_allows_empty_yolo_reviews(self) -> None:
        summary = make_two_detection_summary().model_copy(
            update={
                "behavior_classes": [
                    BehaviorClassSummary(
                        class_id=2,
                        class_name="blocking_fire_lane",
                        required_object_classes=["vehicle"],
                        decision_rules=[
                            "vehicle must occupy a marked fire lane",
                        ],
                    )
                ]
            }
        )
        request = MultiImageReviewRequest(
            image=ValidatedImage(
                "image.jpg",
                Image.new("RGB", (100, 80), "white"),
                100,
                80,
            ),
            summary=summary,
        )

        prompt = build_multi_image_prompt(request, CATALOG)
        template = prompt_output_template(prompt)

        self.assertEqual(request.required_review_observation_ids, ())
        self.assertEqual(template["yolo_reviews"], [])
        self.assertIn("vehicle must occupy a marked fire lane", prompt)
        self.assertIn("bbox_normalized_xyxy", prompt)

    def test_prompt_templates_every_behavior_candidate_and_full_confirmed_shape(
        self,
    ) -> None:
        summary = make_summary().model_copy(
            update={
                "behavior_classes": [
                    BehaviorClassSummary(
                        class_id=0,
                        class_name="trampling_grass",
                        required_object_classes=["person", "grass"],
                    )
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

        prompt = build_review_prompt(summary, CATALOG)
        template = prompt_output_template(prompt)

        self.assertEqual(
            template["behavior_reviews"][0]["candidate_id"],
            "behavior-candidate-0001",
        )
        self.assertIn("每个 candidate_id 必须在 behavior_reviews 中恰好出现一次", prompt)
        self.assertIn('"bbox_normalized_xyxy":[0.15,0.08,0.84,0.95]', prompt)
        self.assertIn("Detector class labels are proposals, not ground truth", prompt)
        self.assertIn("detector confidence 不是语义正确的证据", prompt)
        self.assertIn("不得为已有 detector detection 输出 bbox", prompt)
        self.assertIn("塑料饮料瓶", prompt)

    def test_prompt_includes_required_garbage_visual_rules(self) -> None:
        summary = DetectionSummary(
            total_detections=1,
            counts_by_task_group={"garbage": 1},
            detections=[
                DetectionSummaryItem(
                    observation_id="obs-0001",
                    task_group="garbage",
                    class_id=0,
                    class_name="crumpled_paper_ball",
                    confidence=0.97,
                    bbox_xyxy=(1, 2, 30, 40),
                    bbox_normalized_xyxy=(0.01, 0.025, 0.3, 0.5),
                )
            ],
        )
        prompt = build_review_prompt(
            summary,
            {
                **CATALOG,
                "garbage": ["crumpled_paper_ball", "plastic_drink_bottle"],
            },
            visual_class_guide={
                "garbage": {
                    "crumpled_paper_ball": {
                        "visual": "纸质材料的不规则皱缩团状结构。",
                        "distinguish": ["草地纹理、石头、阴影和背景斑块不是纸团。"],
                    },
                    "plastic_drink_bottle": {
                        "visual": "可见瓶身、瓶颈和瓶盖的塑料瓶。",
                    },
                }
            },
        )

        self.assertIn("纸质材料的不规则皱缩团状结构", prompt)
        self.assertIn("草地纹理、石头、阴影和背景斑块不是纸团", prompt)
        self.assertIn("可见瓶身、瓶颈和瓶盖的塑料瓶", prompt)

    def test_empty_required_reviews_still_parse_findings_and_behaviors(self) -> None:
        summary = make_summary().model_copy(
            update={
                "behavior_classes": [
                    BehaviorClassSummary(
                        class_id=2,
                        class_name="blocking_fire_lane",
                        required_object_classes=["vehicle"],
                    )
                ]
            }
        )
        parsed = parse_review_response(
            json.dumps(
                {
                    "yolo_reviews": [],
                    "new_findings": [
                        {
                            "task_group": "garbage",
                            "class_name": "plastic_drink_bottle",
                            "confidence": 0.76,
                            "bbox_normalized_xyxy": [0.2, 0.2, 0.4, 0.5],
                            "review_pass": "multi_image",
                            "geometry_source": "vlm_multi_image",
                        }
                    ],
                    "behavior_reviews": [
                        {
                            "candidate_id": None,
                            "class_name": "blocking_fire_lane",
                            "verdict": "confirmed",
                            "confidence": 0.8,
                        }
                    ],
                }
            ),
            summary,
            CATALOG,
            review_pass="multi_image",
            required_review_observation_ids=(),
        )

        self.assertEqual(parsed.decisions, [])
        self.assertEqual(
            [finding.class_name for finding in parsed.findings],
            ["plastic_drink_bottle"],
        )
        self.assertEqual(
            [behavior.class_name for behavior in parsed.behaviors],
            ["blocking_fire_lane"],
        )
        self.assertEqual(parsed.issues, [])

    def test_parser_accepts_minimal_7b_response_and_empty_reasoning(self) -> None:
        content = json.dumps(
            {
                "yolo_reviews": [
                    {
                        "observation_id": " obs-0001 ",
                        "verdict": "confirmed",
                        "corrected_task_group": None,
                        "corrected_class_name": None,
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

        parsed = parse_review_response(content, make_summary(), CATALOG)

        self.assertEqual(parsed.decisions[0].observation_id, "obs-0001")
        self.assertEqual(parsed.decisions[0].reasoning, "")
        self.assertEqual(parsed.findings[0].task_group, "garbage")
        self.assertEqual(parsed.findings[0].class_name, "plastic_drink_bottle")
        self.assertIsNone(parsed.findings[0].reasoning)
        self.assertEqual(parsed.behaviors, [])
        self.assertEqual(parsed.issues, [])

    def test_parser_accepts_compact_review_fields(self) -> None:
        parsed = parse_review_response(
            json.dumps(
                {
                    "yolo_reviews": [
                        {
                            "id": "obs-0001",
                            "verdict": "corrected",
                            "task_group": "prohibited_items",
                            "class_name": "speaker",
                        }
                    ],
                    "new_findings": [],
                    "behavior_reviews": [],
                }
            ),
            make_summary(),
            CATALOG,
        )

        self.assertEqual(parsed.decisions[0].verdict, "corrected")
        self.assertEqual(parsed.decisions[0].corrected_class_name, "speaker")

    def test_invalid_yolo_review_does_not_drop_valid_sibling(self) -> None:
        summary = make_summary().model_copy(
            update={
                "total_detections": 2,
                "detections": [
                    *make_summary().detections,
                    DetectionSummaryItem(
                        observation_id="obs-0002",
                        task_group="prohibited_items",
                        class_id=2,
                        class_name="speaker",
                        confidence=0.6,
                        bbox_xyxy=(40, 5, 80, 50),
                        bbox_normalized_xyxy=(0.4, 0.0625, 0.8, 0.625),
                    ),
                ],
            }
        )
        content = json.dumps(
            {
                "yolo_reviews": [
                    {"observation_id": "obs-0001", "verdict": "confirmed"},
                    {"observation_id": "obs-0002", "verdict": "not_a_verdict"},
                ],
                "new_findings": [],
                "behavior_reviews": [],
            }
        )

        parsed = parse_review_response(content, summary, CATALOG)

        self.assertEqual([item.observation_id for item in parsed.decisions], ["obs-0001"])
        self.assertEqual(
            [issue.code for issue in parsed.issues],
            ["invalid_item", "missing_observation_review"],
        )

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
        parsed = parse_review_response(content, make_summary(), CATALOG)
        self.assertEqual(parsed.decisions[0].verdict, "corrected")
        self.assertEqual(parsed.decisions[0].corrected_class_id, 1)
        self.assertEqual(parsed.findings[0].id, "vlm-full-0001")
        self.assertIsNone(parsed.findings[0].geometry)
        self.assertEqual(parsed.behaviors, [])

    def test_invalid_finding_bbox_only_drops_that_item(self) -> None:
        content = json.dumps(
            {
                "yolo_reviews": [{"observation_id": "obs-0001", "verdict": "confirmed"}],
                "new_findings": [
                    {
                        "task_group": "garbage",
                        "class_name": "plastic_drink_bottle",
                        "confidence": 0.8,
                        "bbox_normalized_xyxy": [0.2, 0.2, 0.1, 0.5],
                        "review_pass": "full_image",
                        "geometry_source": "vlm_full_image",
                    },
                    {
                        "task_group": "garbage",
                        "class_name": "plastic_drink_bottle",
                        "confidence": 0.7,
                        "bbox_normalized_xyxy": [-0.1, 0.1, 1.1, 0.9],
                        "review_pass": "full_image",
                        "geometry_source": "vlm_full_image",
                    },
                ],
                "behavior_reviews": [],
            }
        )

        parsed = parse_review_response(
            content,
            make_summary(),
            CATALOG,
            require_finding_bbox=True,
        )

        self.assertEqual(len(parsed.findings), 1)
        self.assertEqual(parsed.findings[0].bbox_normalized_xyxy, (0.0, 0.1, 1.0, 0.9))
        self.assertEqual(parsed.findings[0].review_pass, "full_image")
        self.assertEqual(parsed.issues[0].section, "new_findings")

    def test_multi_image_finding_uses_original_coordinates_and_known_crop_id(self) -> None:
        content = json.dumps(
            {
                "yolo_reviews": [
                    {"observation_id": "obs-0001", "verdict": "confirmed"}
                ],
                "new_findings": [
                    {
                        "task_group": "prohibited_items",
                        "class_name": "speaker",
                        "confidence": 0.65,
                        "bbox_normalized_xyxy": [0.1, 0.2, 0.8, 0.9],
                        "crop_id": "important-crop-01",
                        "review_pass": "multi_image",
                        "geometry_source": "vlm_multi_image",
                    }
                ],
                "behavior_reviews": [],
            }
        )

        parsed = parse_review_response(
            content,
            make_summary(),
            CATALOG,
            review_pass="multi_image",
            require_finding_bbox=True,
            valid_crop_ids={"important-crop-01"},
        )

        self.assertEqual(len(parsed.decisions), 1)
        self.assertEqual(parsed.findings[0].crop_id, "important-crop-01")
        self.assertEqual(parsed.findings[0].geometry_source, "vlm_multi_image")
        self.assertEqual(
            parsed.findings[0].bbox_normalized_xyxy,
            (0.1, 0.2, 0.8, 0.9),
        )
        self.assertEqual(parsed.issues, [])

    def test_parser_handles_candidate_review_and_full_image_behavior(self) -> None:
        summary = make_summary().model_copy(
            update={
                "behavior_classes": [
                    BehaviorClassSummary(
                        class_id=0,
                        class_name="trampling_grass",
                        required_object_classes=["person", "grass"],
                        decision_rules=[
                            "person feet must actually be on the grass",
                        ],
                    ),
                    BehaviorClassSummary(
                        class_id=2,
                        class_name="blocking_fire_lane",
                        required_object_classes=["vehicle"],
                        decision_rules=[
                            "vehicle must occupy a marked fire lane",
                        ],
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
                        "bbox_normalized_xyxy": [-0.1, 0.2, 1.2, 0.9],
                    },
                ],
            }
        )

        parsed = parse_review_response(content, summary, CATALOG)

        self.assertEqual([item.verdict for item in parsed.behaviors], ["rejected", "confirmed"])
        self.assertEqual(parsed.behaviors[0].evidence_observation_ids, ["obs-0001"])
        self.assertIsNone(parsed.behaviors[1].candidate_id)
        self.assertEqual(parsed.behaviors[1].class_id, 2)
        self.assertEqual(
            parsed.behaviors[1].bbox_normalized_xyxy,
            (0.0, 0.2, 1.0, 0.9),
        )

    def test_invalid_behavior_bbox_does_not_drop_valid_behavior(self) -> None:
        summary = make_summary().model_copy(
            update={
                "behavior_classes": [
                    BehaviorClassSummary(
                        class_id=0,
                        class_name="trampling_grass",
                    ),
                    BehaviorClassSummary(
                        class_id=1,
                        class_name="smoking",
                    ),
                ]
            }
        )
        parsed = parse_review_response(
            json.dumps(
                {
                    "yolo_reviews": [
                        {
                            "observation_id": "obs-0001",
                            "verdict": "confirmed",
                        }
                    ],
                    "new_findings": [],
                    "behavior_reviews": [
                        {
                            "class_name": "trampling_grass",
                            "verdict": "confirmed",
                            "confidence": 0.8,
                            "bbox_normalized_xyxy": [0.8, 0.2, 0.2, 0.9],
                        },
                        {
                            "class_name": "smoking",
                            "verdict": "confirmed",
                            "confidence": 0.9,
                            "bbox_normalized_xyxy": [0.1, 0.1, 0.4, 0.8],
                        },
                    ],
                }
            ),
            summary,
            CATALOG,
        )

        self.assertEqual(
            [item.class_name for item in parsed.behaviors],
            ["smoking"],
        )
        self.assertEqual(parsed.issues[0].section, "behavior_reviews")
        self.assertEqual(parsed.issues[0].code, "invalid_item")

    def test_parser_skips_localization_and_reports_incomplete_coverage(self) -> None:
        localized = parse_review_response(
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
        self.assertEqual(localized.decisions, [])
        self.assertEqual(
            [issue.code for issue in localized.issues],
            ["invalid_item", "missing_observation_review"],
        )

        incomplete = parse_review_response(
            '{"yolo_reviews": [], "new_findings": []}',
            make_summary(),
            CATALOG,
        )
        self.assertEqual(incomplete.decisions, [])
        self.assertEqual(incomplete.issues[0].code, "missing_observation_review")

    def test_duplicate_required_object_review_is_reported(self) -> None:
        decision = {"observation_id": "obs-0001", "verdict": "confirmed"}

        parsed = parse_review_response(
            json.dumps(
                {
                    "yolo_reviews": [decision, decision],
                    "new_findings": [],
                    "behavior_reviews": [],
                }
            ),
            make_summary(),
            CATALOG,
            required_review_observation_ids=("obs-0001",),
        )

        self.assertEqual(len(parsed.decisions), 1)
        self.assertIn("duplicate_observation", [issue.code for issue in parsed.issues])

    def test_parser_skips_invalid_task_group_but_keeps_valid_items(self) -> None:
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

        parsed = parse_review_response(content, make_summary(), CATALOG)

        self.assertEqual([item.observation_id for item in parsed.decisions], ["obs-0001"])
        self.assertEqual(parsed.findings, [])
        self.assertEqual(parsed.issues[0].section, "new_findings")
        self.assertEqual(parsed.issues[0].code, "invalid_item")
        self.assertIn("unknown VLM task_group", parsed.issues[0].message)

    def test_rejected_with_legal_correction_is_normalized(self) -> None:
        content = json.dumps(
            {
                "yolo_reviews": [
                    {
                        "observation_id": "obs-0001",
                        "verdict": "rejected",
                        "corrected_task_group": None,
                        "corrected_class_name": "speaker",
                        "confidence": 0.83,
                    }
                ],
                "new_findings": [],
                "behavior_reviews": [],
            }
        )

        parsed = parse_review_response(content, make_summary(), CATALOG)

        self.assertEqual(parsed.decisions[0].verdict, "corrected")
        self.assertEqual(parsed.decisions[0].corrected_task_group, "prohibited_items")
        self.assertEqual(parsed.decisions[0].corrected_class_id, 2)
        self.assertEqual(parsed.decisions[0].corrected_class_name, "speaker")
        self.assertEqual(parsed.issues, [])

    def test_empty_behavior_reviews_reports_missing_candidate_decision(self) -> None:
        summary = make_summary().model_copy(
            update={
                "behavior_classes": [
                    BehaviorClassSummary(
                        class_id=0,
                        class_name="trampling_grass",
                        required_object_classes=["person", "grass"],
                    )
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
                "behavior_reviews": [],
            }
        )

        parsed = parse_review_response(content, summary, CATALOG)

        self.assertEqual(parsed.behaviors, [])
        self.assertEqual(
            [issue.code for issue in parsed.issues],
            ["missing_candidate_decision"],
        )
        self.assertEqual(
            parsed.issues[0].candidate_id,
            "behavior-candidate-0001",
        )

    def test_missing_behavior_section_reports_missing_section(self) -> None:
        summary = make_summary().model_copy(
            update={
                "behavior_classes": [
                    BehaviorClassSummary(class_id=0, class_name="trampling_grass")
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

        parsed = parse_review_response(
            json.dumps(
                {
                    "yolo_reviews": [
                        {"observation_id": "obs-0001", "verdict": "confirmed"}
                    ],
                    "new_findings": [],
                }
            ),
            summary,
            CATALOG,
        )

        self.assertIn("missing_section", [issue.code for issue in parsed.issues])

    def test_partial_and_duplicate_behavior_candidate_decisions_are_reported(
        self,
    ) -> None:
        candidates = [
            BehaviorCandidate(
                id=f"behavior-candidate-{index:04d}",
                class_id=index - 1,
                class_name=class_name,
                evidence_observation_ids=["obs-0001"],
                evidence_class_names=["person"],
            )
            for index, class_name in enumerate(
                ("trampling_grass", "smoking"),
                1,
            )
        ]
        summary = make_summary().model_copy(
            update={
                "behavior_classes": [
                    BehaviorClassSummary(class_id=0, class_name="trampling_grass"),
                    BehaviorClassSummary(class_id=1, class_name="smoking"),
                ],
                "behavior_candidates": candidates,
            }
        )
        decision = {
            "candidate_id": "behavior-candidate-0001",
            "class_name": "trampling_grass",
            "verdict": "rejected",
            "reasoning": "person is beside the grass",
        }

        parsed = parse_review_response(
            json.dumps(
                {
                    "yolo_reviews": [
                        {"observation_id": "obs-0001", "verdict": "confirmed"}
                    ],
                    "new_findings": [],
                    "behavior_reviews": [decision, decision],
                }
            ),
            summary,
            CATALOG,
        )

        codes = [issue.code for issue in parsed.issues]
        self.assertIn("duplicate_candidate_decision", codes)
        self.assertIn("missing_candidate_decision", codes)
        missing = next(
            issue for issue in parsed.issues if issue.code == "missing_candidate_decision"
        )
        self.assertEqual(missing.candidate_id, "behavior-candidate-0002")

    def test_active_scan_parses_each_canonical_behavior_without_candidates(self) -> None:
        classes = (
            "trampling_grass",
            "smoking",
            "blocking_fire_lane",
            "standing_or_lying_on_bench",
        )
        summary = make_summary().model_copy(
            update={
                "behavior_classes": [
                    BehaviorClassSummary(class_id=index, class_name=class_name)
                    for index, class_name in enumerate(classes)
                ]
            }
        )

        for class_name in classes:
            with self.subTest(class_name=class_name):
                parsed = parse_review_response(
                    json.dumps(
                        {
                            "yolo_reviews": [
                                {
                                    "observation_id": "obs-0001",
                                    "verdict": "confirmed",
                                }
                            ],
                            "new_findings": [],
                            "behavior_reviews": [
                                {
                                    "class_name": class_name,
                                    "verdict": "confirmed",
                                    "confidence": 0.9,
                                    "bbox_normalized_xyxy": [0.1, 0.1, 0.8, 0.9],
                                    "reasoning": "clear visual evidence",
                                }
                            ],
                        }
                    ),
                    summary,
                    CATALOG,
                )
                self.assertEqual(parsed.behaviors[0].class_name, class_name)
                self.assertEqual(parsed.issues, [])

    def test_qwen_provider_sends_original_image_and_parses_response(self) -> None:
        settings = ReviewProviderSettings(
            enabled=True,
            endpoint="http://localhost:8000/v1/chat/completions",
            model_id="Qwen2.5-VL",
            image_max_side=64,
            response_format_json_object=True,
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
                    "finish_reason": "stop",
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
            ],
            "usage": {
                "prompt_tokens": 123,
                "completion_tokens": 45,
                "total_tokens": 168,
            },
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
        self.assertIn("一次完成", content[0]["text"])
        self.assertEqual(content[1]["text"], "image_id=original_image")
        self.assertTrue(content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        encoded = content[2]["image_url"]["url"].split(",", 1)[1]
        with Image.open(BytesIO(base64.b64decode(encoded))) as sent_image:
            self.assertEqual(sent_image.size, (64, 51))
        self.assertIn("带喷嘴或喷头的加压气雾罐", content[0]["text"])
        self.assertEqual(captured["calls"], 1)
        self.assertEqual(captured["timeout"], 8.0)
        self.assertEqual(
            captured["body"]["response_format"],
            {"type": "json_object"},
        )
        self.assertEqual(result.review_pass, "multi_image")
        self.assertEqual(result.decisions[0].verdict, "confirmed")
        self.assertEqual(result.metrics.finish_reason, "stop")
        self.assertEqual(result.metrics.prompt_tokens, 123)
        self.assertEqual(result.metrics.completion_tokens, 45)
        self.assertEqual(result.metrics.total_tokens, 168)
        self.assertEqual(result.metrics.image_count, 1)
        self.assertGreater(result.metrics.request_payload_bytes, 0)
        self.assertGreater(result.metrics.encoded_image_bytes_total, 0)

    def test_qwen_sends_original_and_all_important_crops_in_one_request(self) -> None:
        settings = ReviewProviderSettings(
            enabled=True,
            endpoint="http://localhost:8000/v1/chat/completions",
            model_id="Qwen2.5-VL",
        )
        review_settings = ReviewSettings()
        provider = Qwen25VLProvider(
            settings,
            CATALOG,
            review_settings=review_settings,
        )
        image = ValidatedImage(
            "image.jpg",
            Image.new("RGB", (100, 100), "white"),
            100,
            100,
        )
        candidate = ReviewCandidate(
            candidate_id="review-candidate-0001",
            bbox_normalized_xyxy=(0.1, 0.1, 0.3, 0.3),
            reasons=("low_confidence",),
            observation_ids=("obs-0001",),
            priority=2,
        )
        crops = generate_important_crops(
            image,
            [candidate],
            ImportantCropSettings(),
        )
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "yolo_reviews": [
                                    {
                                        "observation_id": "obs-0001",
                                        "verdict": "confirmed",
                                    }
                                ],
                                "new_findings": [
                                    {
                                        "task_group": "garbage",
                                        "class_name": "plastic_drink_bottle",
                                        "confidence": 0.7,
                                        "bbox_normalized_xyxy": [0.5, 0.5, 0.8, 0.8],
                                        "crop_id": crops[0].crop_id,
                                        "review_pass": "multi_image",
                                        "geometry_source": "vlm_multi_image",
                                    }
                                ],
                                "behavior_reviews": [],
                            }
                        )
                    }
                }
            ]
        }
        captured_bodies = []

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return None

            def read(self) -> bytes:
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured_bodies.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse(response_payload)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = provider.review_multi_image(
                MultiImageReviewRequest(
                    image=image,
                    summary=make_summary(),
                    candidates=(candidate,),
                    crops=tuple(crops),
                )
            )

        self.assertEqual(len(captured_bodies), 1)
        content = captured_bodies[0]["messages"][0]["content"]
        images = [item for item in content if item["type"] == "image_url"]
        self.assertEqual(len(images), 1 + len(crops))
        self.assertIn("坐标唯一基准", content[0]["text"])
        self.assertIn(crops[0].crop_id, content[0]["text"])
        self.assertEqual(result.review_pass, "multi_image")
        self.assertEqual(result.findings[0].crop_id, crops[0].crop_id)
        self.assertEqual(result.findings[0].geometry_source, "vlm_multi_image")

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
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": raw_content},
                }
            ],
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
        self.assertIn("finish_reason='length'", message)
        self.assertIn("...", message)
        self.assertNotIn("hidden-tail", message)
        self.assertLess(len(message), 800)


if __name__ == "__main__":
    unittest.main()
