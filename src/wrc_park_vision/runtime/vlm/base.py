"""Provider contract for Runtime V3 semantic review."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..schemas import DetectionSummary, VLMReviewResult, ValidatedImage

if TYPE_CHECKING:
    from ..review.multi_image_request import MultiImageReviewRequest


class ReviewProvider(ABC):
    """Inspect the original image, with optional focused crops, in one request."""

    @abstractmethod
    def review(self, image: ValidatedImage, summary: DetectionSummary) -> VLMReviewResult:
        """Backward-compatible original-image-only entry point."""

    def review_multi_image(
        self,
        request: "MultiImageReviewRequest",
    ) -> VLMReviewResult:
        """Run one request; legacy providers safely receive the original image."""
        return self.review(request.image, request.summary)

    def close(self) -> None:
        """Release provider resources when needed."""
