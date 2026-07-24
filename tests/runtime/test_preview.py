from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from wrc_park_vision.runtime.config import PreviewSettings
from wrc_park_vision.runtime.fusion import fuse_review_results, merge_and_mark_conflicts
from wrc_park_vision.runtime.preview import _layout_label, render_preview
from wrc_park_vision.runtime.schemas import (
    BBoxGeometry,
    Observation,
    ObservationReview,
    ObservationSource,
    ReviewSummary,
    VLMFinding,
    VLMReviewDecision,
)

from .helpers import make_observation, make_response, write_test_image


class PreviewTests(unittest.TestCase):
    def test_preview_uses_pipeline_response_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = write_test_image(root / "image.jpg", (100, 80))
            observations = merge_and_mark_conflicts(
                [make_observation("prohibited_items", "prohibited", 0, "spray_can", 0.9, (10, 20, 50, 60))],
                0.75,
            )
            response = make_response(observations, image_path)
            preview_path = root / "preview.jpg"
            render_preview(image_path, response, preview_path, PreviewSettings())
            with Image.open(preview_path) as image:
                preview = image.convert("RGB")
                edge_pixel = preview.getpixel((10, 40))
                untouched_pixel = preview.getpixel((80, 70))
        self.assertGreater(edge_pixel[0], edge_pixel[1] * 2)
        self.assertTrue(all(channel > 220 for channel in untouched_pixel))

    def test_preview_lists_vlm_only_findings_without_creating_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = write_test_image(root / "source.png")
            observation = make_observation(
                "prohibited_items",
                "prohibited",
                0,
                "spray_can",
                0.9,
                (10, 10, 40, 40),
            )
            observation.id = "obs-0001"
            review = ReviewSummary(
                attempted=True,
                status="completed",
                decisions=[VLMReviewDecision(observation_id="obs-0001", verdict="confirmed")],
                findings=[VLMFinding(id="vlm-0001", task_group="garbage", class_name="bottle")],
            )
            observations, fusion = fuse_review_results([observation], review)
            response = make_response(observations, image_path)
            response.review = review
            response.fusion = fusion
            preview_path = root / "preview.jpg"

            render_preview(image_path, response, preview_path, PreviewSettings())

            with Image.open(preview_path) as rendered:
                self.assertEqual(rendered.size, (100, 80))
            self.assertIsNone(response.review.findings[0].geometry)

    def test_preview_lists_behavior_without_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = write_test_image(root / "source.png")
            response = make_response([], image_path)
            response.observations.append(
                Observation(
                    id="behavior-0001",
                    kind="behavior",
                    task_group="uncivilized_behavior",
                    class_id=0,
                    class_name="trampling_grass",
                    confidence=0.81,
                    source=ObservationSource(
                        module_id="behavior_pipeline",
                        backend="fake_qwen",
                        model_id="fake-vl",
                    ),
                    evidence_observation_ids=[],
                    reasoning="confirmed from the full image",
                )
            )
            preview_path = root / "preview.jpg"

            render_preview(image_path, response, preview_path, PreviewSettings())

            with Image.open(preview_path) as rendered:
                self.assertEqual(rendered.size, (100, 80))

    def test_preview_draws_vlm_finding_from_final_observations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = write_test_image(root / "source.png", (100, 80))
            finding = VLMFinding(
                id="vlm-full-0001",
                task_group="garbage",
                class_id=0,
                class_name="plastic_drink_bottle",
                confidence=0.82,
                bbox_normalized_xyxy=(0.6, 0.2, 0.9, 0.7),
                geometry=BBoxGeometry.from_xyxy((60, 16, 90, 56), 100, 80),
            )
            review = ReviewSummary(
                attempted=True,
                status="completed",
                provider="fake_qwen",
                model_id="fake-vl",
                findings=[finding],
            )
            observations, fusion = fuse_review_results([], review)
            response = make_response(observations, image_path)
            response.review = review
            response.fusion = fusion
            preview_path = root / "preview.jpg"

            render_preview(image_path, response, preview_path, PreviewSettings())

            with Image.open(preview_path) as rendered:
                preview = rendered.convert("RGB")
                edge_pixel = preview.getpixel((60, 40))
                self.assertEqual(rendered.size, (100, 80))
            self.assertGreater(edge_pixel[1], edge_pixel[0])
            self.assertEqual(
                response.observations[0].metadata["geometry_source"],
                "vlm_full_image",
            )

    def test_corrected_and_review_failed_do_not_expand_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = write_test_image(root / "source.png", (200, 120))
            corrected = make_observation(
                "prohibited_items",
                "prohibited",
                0,
                "kick_scooter",
                0.78,
                (20, 20, 80, 80),
                image_size=(200, 120),
            )
            corrected.metadata.update(
                review_source="vlm_corrected",
                geometry_source="yolo",
            )
            review_failed = make_observation(
                "prohibited_items",
                "prohibited",
                1,
                "portable_gas_stove",
                0.51,
                (110, 20, 190, 90),
                image_size=(200, 120),
            )
            review_failed.review = ObservationReview(
                required=True,
                status="pending",
                reasons=["review_item_missing_or_failed"],
            )
            response = make_response([corrected, review_failed], image_path)
            preview_path = root / "preview.jpg"

            render_preview(image_path, response, preview_path, PreviewSettings())

            with Image.open(preview_path) as rendered:
                self.assertEqual(rendered.size, (200, 120))

    def test_label_layout_stays_inside_right_and_top_edges(self) -> None:
        image = Image.new("RGB", (200, 120), "white")
        draw = ImageDraw.Draw(image, "RGBA")
        font = ImageFont.load_default()
        class_name = "standing_or_lying_on_bench"
        status_text = "0.83 | review-failed"

        layout = _layout_label(
            draw,
            font,
            class_name,
            0.83,
            "review-failed",
            (185, 0, 199, 30),
            image.size,
        )

        left, top, right, bottom = layout.box_xyxy
        self.assertTrue(0 <= left < right <= 200)
        self.assertTrue(0 <= top < bottom <= 120)
        self.assertEqual("".join(layout.lines), class_name + status_text)
        for line, origin in zip(layout.lines, layout.text_origins):
            text_box = draw.textbbox(origin, line, font=font, anchor="lt")
            self.assertTrue(0 <= text_box[0] <= text_box[2] <= 200)
            self.assertTrue(0 <= text_box[1] <= text_box[3] <= 120)

    def test_label_layout_handles_tiny_image_without_out_of_bounds(self) -> None:
        image = Image.new("RGB", (8, 8), "white")
        draw = ImageDraw.Draw(image, "RGBA")
        layout = _layout_label(
            draw,
            ImageFont.load_default(),
            "spray_can",
            0.90,
            "confirmed",
            (0, 0, 8, 8),
            image.size,
        )

        left, top, right, bottom = layout.box_xyxy
        self.assertTrue(0 <= left <= right <= 8)
        self.assertTrue(0 <= top <= bottom <= 8)
        for line, origin in zip(layout.lines, layout.text_origins):
            text_box = draw.textbbox(
                origin,
                line,
                font=ImageFont.load_default(),
                anchor="lt",
            )
            self.assertTrue(0 <= text_box[0] <= text_box[2] <= 8)
            self.assertTrue(0 <= text_box[1] <= text_box[3] <= 8)

    def test_label_layout_avoids_existing_label_when_space_is_available(self) -> None:
        image = Image.new("RGB", (200, 120), "white")
        draw = ImageDraw.Draw(image, "RGBA")
        font = ImageFont.load_default()
        first = _layout_label(
            draw,
            font,
            "spray_can",
            0.90,
            "confirmed",
            (10, 10, 35, 45),
            image.size,
        )
        second = _layout_label(
            draw,
            font,
            "plastic_drink_bottle",
            0.88,
            "vlm-full",
            (10, 10, 35, 45),
            image.size,
            occupied_boxes=[first.box_xyxy],
        )

        first_left, first_top, first_right, first_bottom = first.box_xyxy
        second_left, second_top, second_right, second_bottom = second.box_xyxy
        overlap = (
            max(0, min(first_right, second_right) - max(first_left, second_left))
            * max(0, min(first_bottom, second_bottom) - max(first_top, second_top))
        )
        self.assertEqual(overlap, 0)

    def test_rejected_observation_is_not_drawn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = write_test_image(root / "source.png", (100, 80))
            observation = make_observation(
                "prohibited_items",
                "prohibited",
                0,
                "speaker",
                0.66,
                (10, 10, 50, 50),
            )
            observation.id = "obs-0001"
            review = ReviewSummary(
                attempted=True,
                status="completed",
                decisions=[
                    VLMReviewDecision(
                        observation_id="obs-0001",
                        verdict="rejected",
                    )
                ],
            )
            observations, fusion = fuse_review_results([observation], review)
            response = make_response(observations, image_path)
            response.review = review
            response.fusion = fusion
            preview_path = root / "preview.jpg"

            render_preview(image_path, response, preview_path, PreviewSettings())

            self.assertEqual(observations, [])
            with Image.open(preview_path) as rendered:
                self.assertEqual(rendered.size, (100, 80))
                self.assertTrue(
                    all(channel > 245 for channel in rendered.convert("RGB").getpixel((10, 30)))
                )

    def test_all_final_observation_statuses_render_inside_original_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = write_test_image(root / "source.png", (300, 200))
            observations = []
            specs = [
                ("spray_can", "confirmed", (5, 30, 40, 70)),
                ("kick_scooter", "corrected", (55, 30, 95, 70)),
                ("plastic_drink_bottle", "vlm-full", (105, 30, 145, 70)),
                ("empty_cigarette_box", "vlm-crop", (155, 30, 195, 70)),
                ("portable_gas_stove", "uncertain", (205, 30, 245, 70)),
                ("megaphone", "review-failed", (255, 30, 295, 70)),
            ]
            for index, (class_name, status, bbox) in enumerate(specs):
                observation = make_observation(
                    "prohibited_items",
                    "prohibited",
                    index,
                    class_name,
                    0.5 + index * 0.05,
                    bbox,
                    image_size=(300, 200),
                )
                if status == "confirmed":
                    observation.review = ObservationReview(
                        required=True,
                        status="confirmed",
                    )
                elif status == "corrected":
                    observation.metadata["review_source"] = "vlm_corrected"
                    observation.metadata["geometry_source"] = "yolo"
                elif status == "vlm-full":
                    observation.metadata["geometry_source"] = "vlm_full_image"
                elif status == "vlm-crop":
                    observation.metadata["geometry_source"] = "vlm_crop"
                elif status == "uncertain":
                    observation.review = ObservationReview(
                        required=True,
                        status="pending",
                        reasons=["vlm_uncertain"],
                    )
                else:
                    observation.review = ObservationReview(
                        required=True,
                        status="pending",
                        reasons=["review_item_missing_or_failed"],
                    )
                observations.append(observation)
            response = make_response(observations, image_path)
            preview_path = root / "preview.jpg"

            render_preview(image_path, response, preview_path, PreviewSettings())

            with Image.open(preview_path) as rendered:
                preview = rendered.convert("RGB")
                self.assertEqual(rendered.size, (300, 200))
                for _, _, (x1, y1, _, y2) in specs:
                    pixel = preview.getpixel((x1, (y1 + y2) // 2))
                    self.assertFalse(all(channel > 245 for channel in pixel))


if __name__ == "__main__":
    unittest.main()
