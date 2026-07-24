"""Review policy and full-image VLM review coordination."""

from __future__ import annotations

from .config import ReviewSettings
from .schemas import (
    DetectionSummary,
    ModuleSummary,
    Observation,
    ObservationReview,
    ReviewPassSummary,
    ReviewSummary,
    VLMReviewResult,
    ValidatedImage,
)
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
        return reviewed, ReviewSummary(
            required=bool(ordered),
            reasons=ordered,
            status=status,
            uncertain_policy=self.settings.uncertain_policy,
            review_failure_policy=self.settings.review_failure_policy,
        )


class ReviewCoordinator:
    """Keep policy decisions and future VLM inference behind one pipeline call."""

    def __init__(self, settings: ReviewSettings, provider: ReviewProvider | None = None) -> None:
        self.policy = ReviewPolicy(settings)
        self.provider = provider

    def prepare(
        self,
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
        return reviewed, summary

    def apply_result(
        self,
        reviewed: list[Observation],
        summary: ReviewSummary,
        result: VLMReviewResult,
        *,
        include_object_reviews: bool,
        include_behavior_reviews: bool,
    ) -> tuple[list[Observation], ReviewSummary]:
        merged = summary.model_copy(deep=True)
        if include_object_reviews:
            decisions_by_id = {
                decision.observation_id: decision
                for decision in result.decisions
            }
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
                reason = f"{result.review_pass}_vlm_review"
                if reason not in observation.review.reasons:
                    observation.review.reasons.append(reason)

        merged.required = True
        merged.attempted = True
        merged.status = "completed"
        merged.provider = result.provider
        merged.model_id = result.model_id
        merged.duration_ms = (merged.duration_ms or 0.0) + result.duration_ms
        if include_object_reviews:
            merged.decisions.extend(result.decisions)
        merged.findings.extend(result.findings)
        if include_behavior_reviews:
            merged.behaviors.extend(result.behaviors)
        merged.issues.extend(result.issues)
        merged.passes.append(
            ReviewPassSummary(
                pass_id=result.review_pass,
                attempted=True,
                status="completed",
                duration_ms=result.duration_ms,
                finding_count=len(result.findings),
                issue_count=len(result.issues),
            )
        )
        return reviewed, merged

    def apply(
        self,
        image: ValidatedImage,
        observations: list[Observation],
        modules: list[ModuleSummary],
        detection_summary: DetectionSummary,
    ) -> tuple[list[Observation], ReviewSummary]:
        """Backward-compatible single full-image review entry point."""
        reviewed, summary = self.prepare(observations, modules, detection_summary)
        if self.provider is None:
            return reviewed, summary

        result = self.provider.review(image, detection_summary)
        return self.apply_result(
            reviewed,
            summary,
            result,
            include_object_reviews=True,
            include_behavior_reviews=True,
        )
