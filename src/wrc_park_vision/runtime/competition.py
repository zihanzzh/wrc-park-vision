"""Competition-facing response adapter kept separate from internal Runtime schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .schemas import BBoxGeometry, Observation, PipelineResponse


CompetitionStatus = Literal["success", "partial_success", "failed"]
CompetitionReviewStatus = Literal[
    "not_required",
    "confirmed",
    "corrected",
    "uncertain",
    "review_failed",
]
CompetitionObjectSource = Literal[
    "detector",
    "detector_vlm_fused",
    "vlm_finding",
]


class CompetitionFrame(BaseModel):
    frame_id: Optional[int] = Field(default=None, ge=0)
    timestamp: Optional[datetime] = None
    width: int = Field(ge=0)
    height: int = Field(ge=0)


class CompetitionObject(BaseModel):
    object_id: str
    task_group: str
    class_id: int = Field(ge=0)
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    bbox_normalized_xyxy: tuple[float, float, float, float]
    confidence: float = Field(ge=0.0, le=1.0)
    review_status: CompetitionReviewStatus
    source: CompetitionObjectSource


class CompetitionBehavior(BaseModel):
    class_id: int = Field(ge=0)
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_object_ids: list[str] = Field(default_factory=list)


class CompetitionResponse(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str
    frame: CompetitionFrame
    status: CompetitionStatus
    objects: list[CompetitionObject] = Field(default_factory=list)
    behaviors: list[CompetitionBehavior] = Field(default_factory=list)
    processing_time_ms: float = Field(ge=0.0)
    degraded: bool = False


def _review_status(observation: Observation) -> CompetitionReviewStatus:
    if observation.metadata.get("review_source") == "vlm_corrected":
        return "corrected"
    if observation.source.module_id == "vlm_review":
        return "confirmed"
    if "review_item_missing_or_failed" in observation.review.reasons:
        return "review_failed"
    if "vlm_uncertain" in observation.review.reasons:
        return "uncertain"
    if observation.review.status == "confirmed":
        return "confirmed"
    return "not_required"


def _object_source(
    observation: Observation,
    review_status: CompetitionReviewStatus,
) -> CompetitionObjectSource:
    if observation.source.module_id == "vlm_review":
        return "vlm_finding"
    if review_status in {"confirmed", "corrected", "uncertain"}:
        return "detector_vlm_fused"
    return "detector"


def _final_confidence(
    observation: Observation,
    review_status: CompetitionReviewStatus,
) -> float:
    if review_status in {"confirmed", "corrected"}:
        vlm_confidence = observation.metadata.get("vlm_review_confidence")
        if isinstance(vlm_confidence, (int, float)):
            return min(max(float(vlm_confidence), 0.0), 1.0)
    return observation.confidence


def build_competition_response(
    response: PipelineResponse,
    *,
    adapter_duration_ms: float = 0.0,
) -> CompetitionResponse:
    """Map final fused observations to the provisional competition SDK V1 schema."""
    objects: list[CompetitionObject] = []
    behaviors: list[CompetitionBehavior] = []
    for observation in response.observations:
        if observation.kind == "behavior":
            if observation.class_id is None:
                continue
            behaviors.append(
                CompetitionBehavior(
                    class_id=observation.class_id,
                    class_name=observation.class_name,
                    confidence=observation.confidence,
                    evidence_object_ids=list(
                        observation.evidence_observation_ids
                    ),
                )
            )
            continue
        if (
            observation.class_id is None
            or not isinstance(observation.geometry, BBoxGeometry)
        ):
            continue
        review_status = _review_status(observation)
        objects.append(
            CompetitionObject(
                object_id=observation.id,
                task_group=observation.task_group,
                class_id=observation.class_id,
                class_name=observation.class_name,
                bbox_xyxy=observation.geometry.bbox_xyxy,
                bbox_normalized_xyxy=(
                    observation.geometry.bbox_normalized_xyxy
                ),
                confidence=_final_confidence(observation, review_status),
                review_status=review_status,
                source=_object_source(observation, review_status),
            )
        )

    status = "failed" if response.status in {"failed", "failure"} else response.status
    degraded = status != "success" or bool(response.errors)
    return CompetitionResponse(
        request_id=response.request_id,
        frame=CompetitionFrame(
            frame_id=response.input.context.frame_index,
            timestamp=response.input.context.timestamp,
            width=response.input.width,
            height=response.input.height,
        ),
        status=status,
        objects=objects,
        behaviors=behaviors,
        processing_time_ms=response.timing_ms.total + adapter_duration_ms,
        degraded=degraded,
    )
