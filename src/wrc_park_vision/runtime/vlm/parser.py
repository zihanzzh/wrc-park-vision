"""Strict parsing of semantic VLM review responses."""

from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..schemas import DetectionSummary, VLMBehaviorDecision, VLMFinding, VLMReviewDecision


class ReviewResponseError(ValueError):
    """Raised when a VLM response cannot be safely fused."""


class _RawDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str
    verdict: Literal["confirmed", "rejected", "corrected", "uncertain"]
    corrected_task_group: Optional[str] = None
    corrected_class_name: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reasoning: Optional[str] = None

    @model_validator(mode="after")
    def validate_correction(self) -> "_RawDecision":
        has_correction = self.corrected_task_group is not None or self.corrected_class_name is not None
        if self.verdict == "corrected":
            if self.corrected_task_group is None or self.corrected_class_name is None:
                raise ValueError("corrected verdict requires both corrected fields")
        elif has_correction:
            raise ValueError("corrected fields are only allowed for corrected verdict")
        return self


class _RawFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_group: str
    class_name: str
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reasoning: Optional[str] = None


class _RawBehaviorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: Optional[str] = None
    class_name: str
    verdict: Literal["confirmed", "rejected", "uncertain"]
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    evidence_observation_ids: list[str] = Field(default_factory=list)
    reasoning: Optional[str] = None


class _RawResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    yolo_reviews: list[_RawDecision]
    new_findings: list[_RawFinding]
    behavior_reviews: list[_RawBehaviorDecision] = Field(default_factory=list)


def _strip_code_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        raise ReviewResponseError("VLM response has an incomplete Markdown code fence")
    return "\n".join(lines[1:-1]).strip()


def _validate_class(task_group: str, class_name: str, class_catalog: dict[str, list[str]]) -> None:
    if task_group not in class_catalog:
        raise ReviewResponseError(f"unknown VLM task_group: {task_group}")
    if class_name not in class_catalog[task_group]:
        raise ReviewResponseError(f"unknown VLM class for {task_group}: {class_name}")


def parse_review_response(
    content: str,
    summary: DetectionSummary,
    class_catalog: dict[str, list[str]],
) -> tuple[list[VLMReviewDecision], list[VLMFinding], list[VLMBehaviorDecision]]:
    """Parse one complete response without inventing or dropping review records."""
    try:
        payload = json.loads(_strip_code_fence(content))
        raw = _RawResponse.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ReviewResponseError(f"invalid VLM review response: {exc}") from exc

    expected_ids = [item.observation_id for item in summary.detections]
    actual_ids = [item.observation_id for item in raw.yolo_reviews]
    if len(actual_ids) != len(set(actual_ids)):
        raise ReviewResponseError("VLM response contains duplicate observation_id values")
    if set(actual_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(actual_ids))
        unknown = sorted(set(actual_ids) - set(expected_ids))
        raise ReviewResponseError(f"VLM review coverage mismatch: missing={missing}, unknown={unknown}")

    raw_by_id = {item.observation_id: item for item in raw.yolo_reviews}
    decisions: list[VLMReviewDecision] = []
    for observation_id in expected_ids:
        item = raw_by_id[observation_id]
        if item.verdict == "corrected":
            assert item.corrected_task_group is not None
            assert item.corrected_class_name is not None
            _validate_class(item.corrected_task_group, item.corrected_class_name, class_catalog)
        decisions.append(VLMReviewDecision(**item.model_dump()))

    findings: list[VLMFinding] = []
    for index, item in enumerate(raw.new_findings, 1):
        _validate_class(item.task_group, item.class_name, class_catalog)
        findings.append(VLMFinding(id=f"vlm-{index:04d}", **item.model_dump()))

    known_observation_ids = set(expected_ids)
    behavior_classes = {item.class_name: item for item in summary.behavior_classes}
    candidates = {item.id: item for item in summary.behavior_candidates}
    candidate_ids = [item.candidate_id for item in raw.behavior_reviews if item.candidate_id is not None]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ReviewResponseError("VLM response contains duplicate behavior candidate_id values")
    if set(candidate_ids) != set(candidates):
        missing = sorted(set(candidates) - set(candidate_ids))
        unknown = sorted(set(candidate_ids) - set(candidates))
        raise ReviewResponseError(
            f"VLM behavior candidate coverage mismatch: missing={missing}, unknown={unknown}"
        )

    behaviors: list[VLMBehaviorDecision] = []
    confirmed_classes: set[str] = set()
    for index, item in enumerate(raw.behavior_reviews, 1):
        behavior_class = behavior_classes.get(item.class_name)
        if behavior_class is None:
            raise ReviewResponseError(f"unknown VLM behavior class: {item.class_name}")
        candidate = candidates.get(item.candidate_id) if item.candidate_id is not None else None
        if candidate is not None and candidate.class_name != item.class_name:
            raise ReviewResponseError(
                f"behavior candidate {candidate.id} expects {candidate.class_name}, got {item.class_name}"
            )
        if item.candidate_id is None and item.verdict != "confirmed":
            raise ReviewResponseError("full-image behavior findings without a candidate must be confirmed")

        evidence_ids = list(item.evidence_observation_ids)
        unknown_evidence = sorted(set(evidence_ids) - known_observation_ids)
        if unknown_evidence:
            raise ReviewResponseError(f"unknown behavior evidence observation IDs: {unknown_evidence}")
        if candidate is not None:
            if not evidence_ids:
                evidence_ids = list(candidate.evidence_observation_ids)
            unexpected_evidence = sorted(set(evidence_ids) - set(candidate.evidence_observation_ids))
            if unexpected_evidence:
                raise ReviewResponseError(
                    f"behavior candidate {candidate.id} contains unrelated evidence IDs: {unexpected_evidence}"
                )
        if item.verdict == "confirmed":
            if item.class_name in confirmed_classes:
                raise ReviewResponseError(f"duplicate confirmed behavior class: {item.class_name}")
            confirmed_classes.add(item.class_name)

        behaviors.append(
            VLMBehaviorDecision(
                id=f"behavior-review-{index:04d}",
                candidate_id=item.candidate_id,
                class_id=behavior_class.class_id,
                class_name=item.class_name,
                verdict=item.verdict,
                confidence=item.confidence,
                evidence_observation_ids=evidence_ids,
                reasoning=item.reasoning,
            )
        )
    return decisions, findings, behaviors
