from __future__ import annotations

import unittest

from PIL import Image

from wrc_park_vision.runtime.config import CropScanSettings
from wrc_park_vision.runtime.crops import (
    generate_crops,
    map_crop_bbox_to_image,
    map_full_image_bbox,
)
from wrc_park_vision.runtime.schemas import ValidatedImage


def make_image(width: int, height: int) -> ValidatedImage:
    return ValidatedImage(
        "image.jpg",
        Image.new("RGB", (width, height), "white"),
        width,
        height,
    )


class CropTests(unittest.TestCase):
    def test_square_image_generates_overlapping_two_by_two_crops(self) -> None:
        crops = generate_crops(make_image(100, 100), CropScanSettings())

        self.assertEqual(len(crops), 4)
        self.assertEqual(crops[0].bbox_xyxy, (0, 0, 56, 56))
        self.assertEqual(crops[-1].bbox_xyxy, (44, 44, 100, 100))
        self.assertGreater(crops[0].bbox_xyxy[2] - crops[1].bbox_xyxy[0], 0)
        self.assertGreater(crops[0].bbox_xyxy[3] - crops[2].bbox_xyxy[1], 0)

    def test_wide_image_generates_three_horizontal_crops(self) -> None:
        crops = generate_crops(make_image(300, 100), CropScanSettings())

        self.assertEqual(len(crops), 3)
        self.assertTrue(all(crop.bbox_xyxy[1] == 0 for crop in crops))
        self.assertEqual(crops[0].bbox_xyxy[0], 0)
        self.assertEqual(crops[-1].bbox_xyxy[2], 300)

    def test_tall_image_generates_three_vertical_crops(self) -> None:
        crops = generate_crops(make_image(100, 300), CropScanSettings())

        self.assertEqual(len(crops), 3)
        self.assertTrue(all(crop.bbox_xyxy[0] == 0 for crop in crops))
        self.assertEqual(crops[0].bbox_xyxy[1], 0)
        self.assertEqual(crops[-1].bbox_xyxy[3], 300)

    def test_crops_cover_image_without_out_of_bounds_or_empty_regions(self) -> None:
        for width, height in ((100, 100), (300, 100), (100, 300), (1, 1), (2, 1)):
            with self.subTest(size=(width, height)):
                crops = generate_crops(make_image(width, height), CropScanSettings())
                self.assertTrue(crops)
                self.assertTrue(
                    all(
                        0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
                        for x1, y1, x2, y2 in (crop.bbox_xyxy for crop in crops)
                    )
                )
                covered = {
                    (x, y)
                    for crop in crops
                    for x in range(crop.bbox_xyxy[0], crop.bbox_xyxy[2])
                    for y in range(crop.bbox_xyxy[1], crop.bbox_xyxy[3])
                }
                self.assertEqual(len(covered), width * height)

    def test_crop_bbox_mapping_handles_top_left_overlap_bottom_right_and_clipping(self) -> None:
        image = make_image(200, 100)
        crops = generate_crops(image, CropScanSettings())
        left, middle, right = crops

        self.assertEqual(
            map_crop_bbox_to_image(left, (0.0, 0.0, 0.5, 0.5), 200, 100).bbox_xyxy,
            (0.0, 0.0, left.width * 0.5, 50.0),
        )
        middle_geometry = map_crop_bbox_to_image(
            middle,
            (0.25, 0.2, 0.75, 0.8),
            200,
            100,
        )
        self.assertGreater(middle_geometry.bbox_xyxy[0], middle.bbox_xyxy[0])
        self.assertLess(middle_geometry.bbox_xyxy[2], middle.bbox_xyxy[2])
        self.assertEqual(
            map_crop_bbox_to_image(right, (-0.2, 0.5, 1.2, 1.1), 200, 100).bbox_xyxy,
            (float(right.bbox_xyxy[0]), 50.0, 200.0, 100.0),
        )

    def test_full_image_bbox_mapping_supports_non_square_images(self) -> None:
        geometry = map_full_image_bbox((0.1, 0.2, 0.9, 0.8), 300, 100)

        self.assertEqual(geometry.bbox_xyxy, (30.0, 20.0, 270.0, 80.0))
        self.assertEqual(geometry.bbox_normalized_xyxy, (0.1, 0.2, 0.9, 0.8))


if __name__ == "__main__":
    unittest.main()
