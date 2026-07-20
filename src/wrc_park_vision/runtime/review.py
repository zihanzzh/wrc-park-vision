"""VLM review decisions without performing VLM inference."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .config import ReviewSettings
from .schemas import ModuleSummary, Observation, ObservationReview, ReviewSummary, ValidatedImage


REASON_ORDER = ("low_confidence", "cross_model_overlap", "module_failure")


class ReviewPolicy:
    def __init__(self, settings: ReviewSettings) -> None:
        self.settings = settings

    def apply(
        self,
        observations: list[Observation],
        modules: list[ModuleSummary],
    ) -> tuple[list[Observation], ReviewSummary]:
        reviewed = [observation.model_copy(deep=True) for observation in observations]
        top_reasons: set[str] = set()
        for observation in reviewed:
            reasons: list[str] = []
            if observation.confidence < self.settings.low_confidence_threshold:
                reasons.append("low_confidence")
            if self.settings.review_cross_task_overlap and observation.conflicts:
                reasons.append("cross_model_overlap")
            if reasons:
                observation.review = ObservationReview(required=True, status="pending", reasons=reasons)
                top_reasons.update(reasons)
            else:
                observation.review = ObservationReview()

        if self.settings.review_module_failure and any(module.status == "failure" for module in modules):
            top_reasons.add("module_failure")
        ordered = [reason for reason in REASON_ORDER if reason in top_reasons]
        return reviewed, ReviewSummary(required=bool(ordered), reasons=ordered)


class ReviewProvider(ABC):
    """Future review inference contract; no concrete VLM provider exists yet."""

    @abstractmethod
    def review(self, image: ValidatedImage, observations: list[Observation]) -> list[Observation]:
        """Return reviewed observations, including confirmed/rejected status when supported."""


class ReviewCoordinator:
    """Keep policy decisions and future VLM inference behind one pipeline call."""

    def __init__(self, settings: ReviewSettings, provider: ReviewProvider | None = None) -> None:
        self.policy = ReviewPolicy(settings)
        self.provider = provider

    def apply(
        self,
        image: ValidatedImage,
        observations: list[Observation],
        modules: list[ModuleSummary],
    ) -> tuple[list[Observation], ReviewSummary]:
        reviewed, summary = self.policy.apply(observations, modules)
        if self.provider is None or not any(item.review.required for item in reviewed):
            return reviewed, summary
        return self.provider.review(image, reviewed), summary
