"""Provider contract for semantic full-image review."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..crops import ImageCrop
from ..schemas import DetectionSummary, ReviewSummary, VLMReviewResult, ValidatedImage


class ReviewProvider(ABC):
    """Inspect a full image using detection output only as context."""

    supports_crop_scan = False

    @abstractmethod
    def review(self, image: ValidatedImage, summary: DetectionSummary) -> VLMReviewResult:
        """Run the full-image review pass."""

    def review_crops(
        self,
        crops: list[ImageCrop],
        image: ValidatedImage,
        summary: DetectionSummary,
        full_image_review: ReviewSummary,
    ) -> VLMReviewResult:
        """Run one crop-scan request containing all crops."""
        raise NotImplementedError("review provider does not support crop scan")

    def close(self) -> None:
        """Release provider resources when needed."""
