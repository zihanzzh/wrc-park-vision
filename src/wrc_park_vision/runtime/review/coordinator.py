"""Review policy and unified VLM result coordination."""

from __future__ import annotations

from ..config import ReviewSettings
from ..schemas import (
    BBoxGeometry,
    DetectionSummary,
    ModuleSummary,
    Observation,
    ObservationReview,
    ReviewPassSummary,
    ReviewSummary,
    VLMReviewResult,
    ValidatedImage,
)
from ..vlm.base import ReviewProvider
from .candidate_selector import normalized_bbox_area


REASON_ORDER = (
    "task_group_required",
    "low_confidence",
    "cross_model_overlap",
    "small_object",
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
        selection = self.settings.candidate_selection
        for observation in reviewed:
            reasons: list[str] = []
            if observation.task_group in selection.review_all_task_groups:
                reasons.append("task_group_required")
            if (
                selection.include_low_confidence
                and observation.confidence < self.settings.low_confidence_threshold
            ):
                reasons.append("low_confidence")
            if (
                selection.include_cross_model_conflicts
                and self.settings.review_cross_task_overlap
                and observation.conflicts
            ):
                reasons.append("cross_model_overlap")
            if (
                selection.include_small_objects
                and isinstance(observation.geometry, BBoxGeometry)
                and normalized_bbox_area(observation.geometry)
                <= selection.small_object_area_ratio
            ):
                reasons.append("small_object")
            if reasons:
                observation.review = ObservationReview(
                    required=True,
                    status="pending",
                    reasons=reasons,
                )
                top_reasons.update(reasons)
            else:
                observation.review = ObservationReview()

        if self.settings.review_module_failure and any(
            module.status == "failure" for module in modules
        ):
            top_reasons.add("module_failure")
        ordered = [reason for reason in REASON_ORDER if reason in top_reasons]
        return reviewed, ReviewSummary(
            required=bool(ordered),
            reasons=ordered,
            status="pending" if ordered else "not_required",
            uncertain_policy=self.settings.uncertain_policy,
            review_failure_policy=self.settings.review_failure_policy,
        )


class ReviewCoordinator:
    def __init__(
        self,
        settings: ReviewSettings,
        provider: ReviewProvider | None = None,
    ) -> None:
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
    ) -> tuple[list[Observation], ReviewSummary]:
        merged = summary.model_copy(deep=True)
        decisions_by_id = {
            decision.observation_id: decision for decision in result.decisions
        }
        for observation in reviewed:
            decision = decisions_by_id.get(observation.id)
            if decision is None:
                continue
            if decision.verdict in {"confirmed", "corrected"}:
                observation.review.status = "confirmed"
            elif decision.verdict == "rejected":
                observation.review.status = "rejected"
            else:
                observation.review.status = "pending"
            observation.review.required = True
            if "multi_image_vlm_review" not in observation.review.reasons:
                observation.review.reasons.append("multi_image_vlm_review")

        merged.required = True
        merged.attempted = True
        merged.status = "completed"
        merged.provider = result.provider
        merged.model_id = result.model_id
        merged.duration_ms = result.duration_ms
        merged.decisions.extend(result.decisions)
        merged.findings.extend(result.findings)
        merged.behaviors.extend(result.behaviors)
        merged.issues.extend(result.issues)
        merged.metrics = result.metrics
        merged.raw_response_debug = result.raw_response_debug
        if result.metrics is not None and result.metrics.fallback_attempted:
            merged.passes.extend(
                [
                    ReviewPassSummary(
                        pass_id="multi_image",
                        attempted=True,
                        status="failed",
                        error="response_truncated",
                        mode="primary",
                        fallback_reason="response_truncated",
                    ),
                    ReviewPassSummary(
                        pass_id="multi_image",
                        attempted=True,
                        status="completed",
                        duration_ms=result.duration_ms,
                        finding_count=0,
                        issue_count=len(result.issues),
                        mode="compact_fallback",
                        fallback_reason="response_truncated",
                    ),
                ]
            )
        else:
            merged.passes.append(
                ReviewPassSummary(
                    pass_id="multi_image",
                    attempted=True,
                    status="completed",
                    duration_ms=result.duration_ms,
                    finding_count=len(result.findings),
                    issue_count=len(result.issues),
                )
            )
        return reviewed, merged

    @staticmethod
    def mark_required_observations(
        observations: list[Observation],
        required_observation_ids: tuple[str, ...],
    ) -> None:
        required_ids = set(required_observation_ids)
        for observation in observations:
            if observation.id not in required_ids:
                continue
            observation.review.required = True
            observation.review.status = "pending"
            if "review_candidate" not in observation.review.reasons:
                observation.review.reasons.append("review_candidate")

    def apply(
        self,
        image: ValidatedImage,
        observations: list[Observation],
        modules: list[ModuleSummary],
        detection_summary: DetectionSummary,
    ) -> tuple[list[Observation], ReviewSummary]:
        """Compatibility entry point for original-image-only provider callers."""
        reviewed, summary = self.prepare(observations, modules, detection_summary)
        if self.provider is None:
            return reviewed, summary
        return self.apply_result(
            reviewed,
            summary,
            self.provider.review(image, detection_summary),
        )
