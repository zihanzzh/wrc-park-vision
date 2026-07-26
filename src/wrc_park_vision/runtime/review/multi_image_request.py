"""Typed input for the single Runtime V3 VLM call."""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas import DetectionSummary, ValidatedImage
from .candidate_selector import ReviewCandidate
from .crop_generator import ReviewCrop


@dataclass(frozen=True)
class MultiImageReviewRequest:
    image: ValidatedImage
    summary: DetectionSummary
    candidates: tuple[ReviewCandidate, ...] = ()
    crops: tuple[ReviewCrop, ...] = ()
    timeout_seconds: float | None = None

    @property
    def required_review_observation_ids(self) -> tuple[str, ...]:
        """Return candidate-linked observation IDs once, in summary order."""
        candidate_ids = {
            observation_id
            for candidate in self.candidates
            for observation_id in candidate.observation_ids
        }
        return tuple(
            detection.observation_id
            for detection in self.summary.detections
            if detection.observation_id in candidate_ids
        )
