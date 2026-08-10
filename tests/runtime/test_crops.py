from __future__ import annotations

import unittest

from PIL import Image

from wrc_park_vision.runtime.config import (
    CandidateSelectionSettings,
    ImportantCropSettings,
)
from wrc_park_vision.runtime.review import (
    ReviewCandidate,
    generate_important_crops,
    select_review_candidates,
)
from wrc_park_vision.runtime.schemas import (
    BehaviorCandidate,
    Conflict,
    ValidatedImage,
)

from .helpers import make_observation


def make_image(width: int = 200, height: int = 100) -> ValidatedImage:
    return ValidatedImage(
        "image.jpg",
        Image.new("RGB", (width, height), "white"),
        width,
        height,
    )


class ReviewCropTests(unittest.TestCase):
    def test_crop_covering_most_of_original_image_is_skipped(self) -> None:
        candidate = ReviewCandidate(
            candidate_id="candidate-large",
            bbox_normalized_xyxy=(0.05, 0.05, 0.95, 0.95),
            reasons=("behavior_candidate",),
            observation_ids=("obs-0001",),
            priority=3,
        )

        crops = generate_important_crops(
            make_image(100, 100),
            [candidate],
            ImportantCropSettings(skip_crop_area_ratio=0.8),
        )

        self.assertEqual(crops, [])

    def test_selects_low_confidence_conflict_small_and_behavior_candidates(self) -> None:
        low = make_observation(
            "garbage", "garbage", 0, "bottle", 0.3, (10, 10, 40, 40), (200, 100)
        )
        low.id = "obs-0001"
        conflict = make_observation(
            "prohibited_items",
            "world",
            0,
            "spray_can",
            0.9,
            (80, 10, 130, 60),
            (200, 100),
        )
        conflict.id = "obs-0002"
        conflict.conflicts = [Conflict(observation_id="obs-0003")]
        small = make_observation(
            "garbage", "garbage", 1, "paper", 0.9, (180, 80, 190, 90), (200, 100)
        )
        small.id = "obs-0003"
        behavior = BehaviorCandidate(
            id="behavior-candidate-0001",
            class_id=0,
            class_name="trampling_grass",
            evidence_observation_ids=["obs-0001", "obs-0002"],
            evidence_class_names=["person", "grass"],
        )

        selected = select_review_candidates(
            [low, conflict, small],
            [behavior],
            CandidateSelectionSettings(),
            low_confidence_threshold=0.45,
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(selected[0].reasons, ("low_confidence",))
        self.assertEqual(selected[1].reasons, ("cross_model_conflict",))
        self.assertEqual(selected[2].reasons, ("small_object",))
        self.assertEqual(selected[3].reasons, ("behavior_candidate",))
        self.assertEqual(
            selected[3].behavior_candidate_ids,
            ("behavior-candidate-0001",),
        )

    def test_review_all_task_groups_selects_high_confidence_objects_only(self) -> None:
        garbage = make_observation(
            "garbage",
            "garbage",
            0,
            "crumpled_paper_ball",
            0.97,
            (10, 10, 80, 70),
            (200, 100),
        )
        prohibited = make_observation(
            "prohibited_items",
            "world",
            0,
            "spray_can",
            0.96,
            (90, 10, 160, 70),
            (200, 100),
        )
        behavior_object = make_observation(
            "uncivilized_behavior",
            "world",
            0,
            "person",
            0.99,
            (20, 5, 180, 95),
            (200, 100),
        )
        for index, observation in enumerate(
            (garbage, prohibited, behavior_object),
            1,
        ):
            observation.id = f"obs-{index:04d}"

        selected = select_review_candidates(
            [garbage, prohibited, behavior_object],
            [],
            CandidateSelectionSettings(
                review_all_task_groups=["garbage", "prohibited_items"]
            ),
            low_confidence_threshold=0.45,
        )

        self.assertEqual(
            [candidate.observation_ids for candidate in selected],
            [("obs-0001",), ("obs-0002",)],
        )
        self.assertTrue(
            all("task_group_required" in candidate.reasons for candidate in selected)
        )

    def test_important_crops_are_candidate_driven_not_fixed_grid(self) -> None:
        candidate = ReviewCandidate(
            candidate_id="review-candidate-0001",
            bbox_normalized_xyxy=(0.05, 0.1, 0.15, 0.3),
            reasons=("low_confidence",),
            observation_ids=("obs-0001",),
            priority=2,
        )

        crops = generate_important_crops(
            make_image(),
            [candidate],
            ImportantCropSettings(),
        )

        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0].observation_ids, ("obs-0001",))
        self.assertLess(crops[0].bbox_xyxy[2], 100)
        self.assertEqual(crops[0].image.size, (crops[0].width, crops[0].height))

    def test_overlapping_candidates_are_merged(self) -> None:
        candidates = [
            ReviewCandidate(
                candidate_id=f"review-candidate-{index:04d}",
                bbox_normalized_xyxy=bbox,
                reasons=("low_confidence",),
                observation_ids=(f"obs-{index:04d}",),
                priority=2,
            )
            for index, bbox in enumerate(
                ((0.2, 0.2, 0.4, 0.5), (0.3, 0.25, 0.5, 0.55)),
                1,
            )
        ]

        crops = generate_important_crops(
            make_image(),
            candidates,
            ImportantCropSettings(context_scale=1.5, merge_iou_threshold=0.1),
        )

        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0].observation_ids, ("obs-0001", "obs-0002"))

    def test_crop_count_is_capped_and_all_regions_stay_in_bounds(self) -> None:
        candidates = [
            ReviewCandidate(
                candidate_id=f"review-candidate-{index:04d}",
                bbox_normalized_xyxy=(
                    index * 0.15,
                    0.1,
                    index * 0.15 + 0.05,
                    0.2,
                ),
                reasons=("small_object",),
                observation_ids=(f"obs-{index:04d}",),
                priority=1,
            )
            for index in range(6)
        ]

        crops = generate_important_crops(
            make_image(),
            candidates,
            ImportantCropSettings(
                context_scale=1.0,
                min_crop_size_ratio=0.05,
                merge_iou_threshold=1.0,
                max_crops=3,
            ),
        )

        self.assertEqual(len(crops), 3)
        self.assertTrue(
            all(
                0 <= x1 < x2 <= 200 and 0 <= y1 < y2 <= 100
                for x1, y1, x2, y2 in (crop.bbox_xyxy for crop in crops)
            )
        )

    def test_disabled_or_empty_candidate_set_creates_no_crops(self) -> None:
        image = make_image()
        self.assertEqual(
            generate_important_crops(image, [], ImportantCropSettings()),
            [],
        )
        candidate = ReviewCandidate(
            candidate_id="review-candidate-0001",
            bbox_normalized_xyxy=(0.1, 0.1, 0.2, 0.2),
            reasons=("small_object",),
            observation_ids=("obs-0001",),
            priority=1,
        )
        self.assertEqual(
            generate_important_crops(
                image,
                [candidate],
                ImportantCropSettings(enabled=False),
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
