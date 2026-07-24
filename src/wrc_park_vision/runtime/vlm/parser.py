"""Fault-tolerant parsing of semantic VLM review responses."""

from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..schemas import (
    DetectionSummary,
    ParsedReviewResponse,
    ReviewIssue,
    VLMBehaviorDecision,
    VLMFinding,
    VLMReviewDecision,
)


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

    @field_validator("observation_id", mode="before")
    @classmethod
    def strip_observation_id(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("corrected_task_group", "corrected_class_name", mode="before")
    @classmethod
    def strip_optional_identifiers(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        return stripped or None


class _RawFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_group: str
    class_name: str
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reasoning: Optional[str] = None

    @field_validator("task_group", "class_name", mode="before")
    @classmethod
    def strip_identifiers(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class _RawBehaviorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: Optional[str] = None
    class_name: str
    verdict: Literal["confirmed", "rejected", "uncertain"]
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    evidence_observation_ids: list[str] = Field(default_factory=list)
    reasoning: Optional[str] = None

    @field_validator("candidate_id", "class_name", mode="before")
    @classmethod
    def strip_identifiers(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("evidence_observation_ids", mode="before")
    @classmethod
    def strip_evidence_ids(cls, value: object) -> object:
        if isinstance(value, list):
            return [item.strip() if isinstance(item, str) else item for item in value]
        return value


def _strip_code_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        raise ReviewResponseError("VLM response has an incomplete Markdown code fence")
    return "\n".join(lines[1:-1]).strip()


def _class_id(task_group: str, class_name: str, class_catalog: dict[str, list[str]]) -> int:
    if task_group not in class_catalog:
        raise ReviewResponseError(f"unknown VLM task_group: {task_group}")
    if class_name not in class_catalog[task_group]:
        raise ReviewResponseError(f"unknown VLM class for {task_group}: {class_name}")
    return class_catalog[task_group].index(class_name)


def _section_items(
    payload: dict[str, object],
    section: str,
    issues: list[ReviewIssue],
) -> list[object]:
    value = payload.get(section, [])
    if isinstance(value, list):
        return value
    issues.append(
        ReviewIssue(
            section=section,  # type: ignore[arg-type]
            code="invalid_section",
            message=f"{section} must be an array",
        )
    )
    return []


def _item_issue(
    section: str,
    index: int,
    code: str,
    message: str,
    *,
    observation_id: str | None = None,
    candidate_id: str | None = None,
) -> ReviewIssue:
    return ReviewIssue(
        section=section,  # type: ignore[arg-type]
        item_index=index,
        code=code,
        message=message,
        observation_id=observation_id,
        candidate_id=candidate_id,
    )


def _infer_correction_task_group(
    class_name: str,
    original_task_group: str,
    class_catalog: dict[str, list[str]],
) -> str:
    if class_name in class_catalog.get(original_task_group, []):
        return original_task_group
    matching_groups = [
        task_group
        for task_group, class_names in class_catalog.items()
        if class_name in class_names
    ]
    if len(matching_groups) == 1:
        return matching_groups[0]
    raise ReviewResponseError(
        f"cannot deterministically infer corrected task_group for class: {class_name}"
    )


def parse_review_response(
    content: str,
    summary: DetectionSummary,
    class_catalog: dict[str, list[str]],
) -> ParsedReviewResponse:
    """Keep valid items and report invalid siblings without inventing semantics."""
    try:
        payload = json.loads(_strip_code_fence(content))
    except json.JSONDecodeError as exc:
        raise ReviewResponseError(f"invalid VLM review response: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReviewResponseError("invalid VLM review response: top-level value must be an object")

    expected_ids = [item.observation_id for item in summary.detections]
    summary_by_id = {item.observation_id: item for item in summary.detections}
    issues: list[ReviewIssue] = []
    decisions_by_id: dict[str, VLMReviewDecision] = {}
    for index, value in enumerate(_section_items(payload, "yolo_reviews", issues)):
        try:
            item = _RawDecision.model_validate(value)
        except ValidationError as exc:
            issues.append(_item_issue("yolo_reviews", index, "invalid_item", str(exc)))
            continue
        observation_id = item.observation_id
        if observation_id not in summary_by_id:
            issues.append(
                _item_issue(
                    "yolo_reviews",
                    index,
                    "unknown_observation",
                    f"unknown observation_id: {observation_id}",
                    observation_id=observation_id,
                )
            )
            continue
        if observation_id in decisions_by_id:
            issues.append(
                _item_issue(
                    "yolo_reviews",
                    index,
                    "duplicate_observation",
                    f"duplicate observation_id: {observation_id}",
                    observation_id=observation_id,
                )
            )
            continue

        verdict = item.verdict
        corrected_task_group = item.corrected_task_group
        corrected_class_name = item.corrected_class_name
        corrected_class_id = None
        try:
            if verdict == "rejected" and corrected_class_name is not None:
                corrected_task_group = corrected_task_group or _infer_correction_task_group(
                    corrected_class_name,
                    summary_by_id[observation_id].task_group,
                    class_catalog,
                )
                corrected_class_id = _class_id(
                    corrected_task_group,
                    corrected_class_name,
                    class_catalog,
                )
                verdict = "corrected"
            elif verdict == "corrected":
                if corrected_class_name is None:
                    raise ReviewResponseError(
                        "corrected verdict requires corrected_class_name"
                    )
                corrected_task_group = corrected_task_group or _infer_correction_task_group(
                    corrected_class_name,
                    summary_by_id[observation_id].task_group,
                    class_catalog,
                )
                corrected_class_id = _class_id(
                    corrected_task_group,
                    corrected_class_name,
                    class_catalog,
                )
            elif corrected_task_group is not None or corrected_class_name is not None:
                raise ReviewResponseError(
                    f"{verdict} verdict must not contain corrected fields"
                )
        except ReviewResponseError as exc:
            issues.append(
                _item_issue(
                    "yolo_reviews",
                    index,
                    "invalid_correction",
                    str(exc),
                    observation_id=observation_id,
                )
            )
            continue

        decisions_by_id[observation_id] = VLMReviewDecision(
            observation_id=observation_id,
            verdict=verdict,
            corrected_task_group=corrected_task_group if verdict == "corrected" else None,
            corrected_class_id=corrected_class_id if verdict == "corrected" else None,
            corrected_class_name=corrected_class_name if verdict == "corrected" else None,
            confidence=item.confidence,
            reasoning=item.reasoning,
        )

    for observation_id in expected_ids:
        if observation_id not in decisions_by_id:
            issues.append(
                ReviewIssue(
                    section="yolo_reviews",
                    code="missing_observation_review",
                    message=f"missing review for observation_id: {observation_id}",
                    observation_id=observation_id,
                )
            )
    decisions = [
        decisions_by_id[observation_id]
        for observation_id in expected_ids
        if observation_id in decisions_by_id
    ]

    findings: list[VLMFinding] = []
    for index, value in enumerate(_section_items(payload, "new_findings", issues)):
        try:
            item = _RawFinding.model_validate(value)
            _class_id(item.task_group, item.class_name, class_catalog)
        except (ValidationError, ReviewResponseError) as exc:
            issues.append(_item_issue("new_findings", index, "invalid_item", str(exc)))
            continue
        findings.append(
            VLMFinding(
                id=f"vlm-{len(findings) + 1:04d}",
                **item.model_dump(),
            )
        )

    known_observation_ids = set(expected_ids)
    behavior_classes = {item.class_name: item for item in summary.behavior_classes}
    candidates = {item.id: item for item in summary.behavior_candidates}
    behaviors: list[VLMBehaviorDecision] = []
    reviewed_candidate_ids: set[str] = set()
    confirmed_classes: set[str] = set()
    for index, value in enumerate(_section_items(payload, "behavior_reviews", issues)):
        try:
            item = _RawBehaviorDecision.model_validate(value)
        except ValidationError as exc:
            issues.append(_item_issue("behavior_reviews", index, "invalid_item", str(exc)))
            continue
        behavior_class = behavior_classes.get(item.class_name)
        if behavior_class is None:
            issues.append(
                _item_issue(
                    "behavior_reviews",
                    index,
                    "unknown_behavior_class",
                    f"unknown VLM behavior class: {item.class_name}",
                    candidate_id=item.candidate_id,
                )
            )
            continue
        candidate = candidates.get(item.candidate_id) if item.candidate_id is not None else None
        if item.candidate_id is not None and candidate is None:
            issues.append(
                _item_issue(
                    "behavior_reviews",
                    index,
                    "unknown_candidate",
                    f"unknown behavior candidate_id: {item.candidate_id}",
                    candidate_id=item.candidate_id,
                )
            )
            continue
        if item.candidate_id is not None and item.candidate_id in reviewed_candidate_ids:
            issues.append(
                _item_issue(
                    "behavior_reviews",
                    index,
                    "duplicate_candidate",
                    f"duplicate behavior candidate_id: {item.candidate_id}",
                    candidate_id=item.candidate_id,
                )
            )
            continue
        if candidate is not None and candidate.class_name != item.class_name:
            issues.append(
                _item_issue(
                    "behavior_reviews",
                    index,
                    "candidate_class_mismatch",
                    f"behavior candidate {candidate.id} expects "
                    f"{candidate.class_name}, got {item.class_name}",
                    candidate_id=candidate.id,
                )
            )
            continue
        if item.candidate_id is None and item.verdict != "confirmed":
            issues.append(
                _item_issue(
                    "behavior_reviews",
                    index,
                    "unconfirmed_full_image_behavior",
                    "behavior without a candidate must be confirmed",
                )
            )
            continue

        evidence_ids = list(item.evidence_observation_ids)
        unknown_evidence = sorted(set(evidence_ids) - known_observation_ids)
        if unknown_evidence:
            issues.append(
                _item_issue(
                    "behavior_reviews",
                    index,
                    "unknown_behavior_evidence",
                    f"unknown behavior evidence observation IDs: {unknown_evidence}",
                    candidate_id=item.candidate_id,
                )
            )
            continue
        if candidate is not None:
            if not evidence_ids:
                evidence_ids = list(candidate.evidence_observation_ids)
            unexpected_evidence = sorted(set(evidence_ids) - set(candidate.evidence_observation_ids))
            if unexpected_evidence:
                issues.append(
                    _item_issue(
                        "behavior_reviews",
                        index,
                        "unrelated_behavior_evidence",
                        f"behavior candidate {candidate.id} contains unrelated evidence IDs: "
                        f"{unexpected_evidence}",
                        candidate_id=candidate.id,
                    )
                )
                continue
        if item.verdict == "confirmed":
            if item.class_name in confirmed_classes:
                issues.append(
                    _item_issue(
                        "behavior_reviews",
                        index,
                        "duplicate_confirmed_behavior",
                        f"duplicate confirmed behavior class: {item.class_name}",
                        candidate_id=item.candidate_id,
                    )
                )
                continue
            confirmed_classes.add(item.class_name)
        if item.candidate_id is not None:
            reviewed_candidate_ids.add(item.candidate_id)

        behaviors.append(
            VLMBehaviorDecision(
                id=f"behavior-review-{len(behaviors) + 1:04d}",
                candidate_id=item.candidate_id,
                class_id=behavior_class.class_id,
                class_name=item.class_name,
                verdict=item.verdict,
                confidence=item.confidence,
                evidence_observation_ids=evidence_ids,
                reasoning=item.reasoning,
            )
        )
    return ParsedReviewResponse(
        decisions=decisions,
        findings=findings,
        behaviors=behaviors,
        issues=issues,
    )
