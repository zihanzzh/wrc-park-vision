"""Strict parsing of semantic VLM review responses."""

from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..schemas import DetectionSummary, VLMFinding, VLMReviewDecision


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


class _RawResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    yolo_reviews: list[_RawDecision]
    new_findings: list[_RawFinding]


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
) -> tuple[list[VLMReviewDecision], list[VLMFinding]]:
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
    return decisions, findings
