"""Review policy and full-image VLM review coordination."""

from __future__ import annotations

from .config import ReviewSettings
from .schemas import DetectionSummary, ModuleSummary, Observation, ObservationReview, ReviewSummary, ValidatedImage
from .vlm.base import ReviewProvider


REASON_ORDER = (
    "low_confidence",
    "cross_model_overlap",
    "module_failure",
    "behavior_candidate",
    "behavior_full_image_scan",
)


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
        status = "pending" if ordered else "not_required"
        return reviewed, ReviewSummary(required=bool(ordered), reasons=ordered, status=status)


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
        detection_summary: DetectionSummary,
    ) -> tuple[list[Observation], ReviewSummary]:
        reviewed, summary = self.policy.apply(observations, modules)
        behavior_reasons: list[str] = []
        if detection_summary.behavior_candidates:
            behavior_reasons.append("behavior_candidate")
        if detection_summary.behavior_classes:
            behavior_reasons.append("behavior_full_image_scan")
        if behavior_reasons:
            summary.required = True
            summary.status = "pending"
            summary.reasons = [
                reason
                for reason in REASON_ORDER
                if reason in {*summary.reasons, *behavior_reasons}
            ]
        if self.provider is None:
            return reviewed, summary

        result = self.provider.review(image, detection_summary)
        decisions_by_id = {decision.observation_id: decision for decision in result.decisions}
        for observation in reviewed:
            decision = decisions_by_id.get(observation.id)
            if decision is None:
                continue
            if decision.verdict == "confirmed":
                observation.review.status = "confirmed"
            elif decision.verdict == "rejected":
                observation.review.status = "rejected"
            elif decision.verdict == "corrected":
                observation.review.status = "confirmed"
            else:
                observation.review.status = "pending"
            observation.review.required = True
            if "full_image_vlm_review" not in observation.review.reasons:
                observation.review.reasons.append("full_image_vlm_review")

        return reviewed, ReviewSummary(
            required=True,
            reasons=summary.reasons,
            attempted=True,
            status="completed",
            provider=result.provider,
            model_id=result.model_id,
            duration_ms=result.duration_ms,
            decisions=result.decisions,
            findings=result.findings,
            behaviors=result.behaviors,
        )
