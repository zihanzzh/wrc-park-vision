"""Provider contract for semantic full-image review."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import DetectionSummary, VLMReviewResult, ValidatedImage


class ReviewProvider(ABC):
    """Inspect a full image using detection output only as context."""

    @abstractmethod
    def review(self, image: ValidatedImage, summary: DetectionSummary) -> VLMReviewResult:
        """Return semantic decisions and optional non-localized missed-object findings."""

    def close(self) -> None:
        """Release provider resources when needed."""
