"""Stable runtime input and output schemas."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


Float4 = tuple[float, float, float, float]
PipelineStatus = Literal["success", "partial_success", "failed", "failure"]
ModuleStatus = Literal["success", "failure"]
ReviewStatus = Literal["not_required", "pending", "confirmed", "rejected"]
ReviewExecutionStatus = Literal["not_required", "pending", "completed", "failed"]
ReviewVerdict = Literal["confirmed", "rejected", "corrected", "uncertain"]
BehaviorVerdict = Literal["confirmed", "rejected", "uncertain"]
ReviewPassId = Literal["full_image", "crop_scan"]
FindingGeometrySource = Literal["vlm_full_image", "vlm_crop"]
FusionStatus = Literal["not_run", "completed", "fallback"]
FusionAction = Literal[
    "keep_yolo",
    "keep_uncertain",
    "drop_uncertain",
    "keep_review_failed",
    "drop_review_failed",
    "reject_yolo",
    "correct_yolo",
    "add_vlm_finding",
    "merge_duplicate",
    "mark_cross_source_conflict",
    "add_behavior",
]


class BBoxGeometry(BaseModel):
    """Canonical pixel bbox plus normalized coordinates derived from it."""

    type: Literal["bbox"] = "bbox"
    bbox_xyxy: Float4
    bbox_normalized_xyxy: Float4

    @model_validator(mode="after")
    def validate_coordinates(self) -> "BBoxGeometry":
        x1, y1, x2, y2 = self.bbox_xyxy
        nx1, ny1, nx2, ny2 = self.bbox_normalized_xyxy
        if not all(math.isfinite(value) for value in (*self.bbox_xyxy, *self.bbox_normalized_xyxy)):
            raise ValueError("bbox coordinates must be finite")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy must have positive width and height")
        if not all(0.0 <= value <= 1.0 for value in self.bbox_normalized_xyxy):
            raise ValueError("normalized bbox coordinates must be in [0, 1]")
        if nx2 <= nx1 or ny2 <= ny1:
            raise ValueError("bbox_normalized_xyxy must have positive width and height")
        return self

    @classmethod
    def from_xyxy(cls, xyxy: list[float] | tuple[float, ...], image_width: int, image_height: int) -> "BBoxGeometry":
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if len(xyxy) != 4:
            raise ValueError("bbox requires exactly four xyxy coordinates")
        values = tuple(float(value) for value in xyxy)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("bbox coordinates must be finite")
        x1, y1, x2, y2 = values
        clipped = (
            min(max(x1, 0.0), float(image_width)),
            min(max(y1, 0.0), float(image_height)),
            min(max(x2, 0.0), float(image_width)),
            min(max(y2, 0.0), float(image_height)),
        )
        cx1, cy1, cx2, cy2 = clipped
        if cx2 <= cx1 or cy2 <= cy1:
            raise ValueError("bbox is invalid after clipping to image bounds")
        normalized = (
            cx1 / image_width,
            cy1 / image_height,
            cx2 / image_width,
            cy2 / image_height,
        )
        return cls(bbox_xyxy=clipped, bbox_normalized_xyxy=normalized)


class MaskGeometry(BaseModel):
    type: Literal["mask"] = "mask"
    encoding: Literal["polygon", "rle", "external"]
    data: Any
    bbox_xyxy: Optional[Float4] = None


class PoseGeometry(BaseModel):
    type: Literal["pose"] = "pose"
    keypoints: list[tuple[float, float, float]]
    bbox_xyxy: Optional[Float4] = None


class RegionGeometry(BaseModel):
    type: Literal["region"] = "region"
    polygon_xy: list[tuple[float, float]]


class RelationGeometry(BaseModel):
    type: Literal["relation"] = "relation"
    subject_id: str
    predicate: str
    object_id: Optional[str] = None
    region: Optional[RegionGeometry] = None


Geometry = Annotated[
    Union[BBoxGeometry, MaskGeometry, PoseGeometry, RegionGeometry, RelationGeometry],
    Field(discriminator="type"),
]


class ObservationSource(BaseModel):
    module_id: str
    backend: str
    model_id: str


class ObservationReview(BaseModel):
    required: bool = False
    status: ReviewStatus = "not_required"
    reasons: list[str] = Field(default_factory=list)


class Conflict(BaseModel):
    observation_id: str
    type: Literal["cross_model_overlap"] = "cross_model_overlap"


class Observation(BaseModel):
    id: str = ""
    kind: Literal["detection", "segmentation", "pose", "region", "relation", "behavior"]
    task_group: str
    class_id: Optional[int] = None
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: ObservationSource
    track_id: Optional[str] = None
    geometry: Optional[Geometry] = None
    evidence_observation_ids: list[str] = Field(default_factory=list)
    reasoning: Optional[str] = None
    review: ObservationReview = Field(default_factory=ObservationReview)
    conflicts: list[Conflict] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_detection(self) -> "Observation":
        if self.kind == "detection":
            if self.class_id is None:
                raise ValueError("detection observations require class_id")
            if not isinstance(self.geometry, BBoxGeometry):
                raise ValueError("detection observations require bbox geometry")
        elif self.kind != "behavior" and self.geometry is None:
            raise ValueError(f"{self.kind} observations require geometry")
        return self


class DetectionSummaryItem(BaseModel):
    observation_id: str
    task_group: str
    class_id: Optional[int] = None
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: Float4
    bbox_normalized_xyxy: Float4
    conflict_observation_ids: list[str] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)


class DetectionSummary(BaseModel):
    total_detections: int = Field(ge=0)
    counts_by_task_group: dict[str, int] = Field(default_factory=dict)
    detections: list[DetectionSummaryItem] = Field(default_factory=list)
    behavior_classes: list["BehaviorClassSummary"] = Field(default_factory=list)
    behavior_candidates: list["BehaviorCandidate"] = Field(default_factory=list)


class BehaviorClassSummary(BaseModel):
    class_id: int = Field(ge=0)
    class_name: str
    required_object_classes: list[str] = Field(default_factory=list)


class BehaviorCandidate(BaseModel):
    id: str
    task_group: Literal["uncivilized_behavior"] = "uncivilized_behavior"
    class_id: int = Field(ge=0)
    class_name: str
    evidence_observation_ids: list[str] = Field(min_length=1)
    evidence_class_names: list[str] = Field(min_length=1)


class VLMReviewDecision(BaseModel):
    observation_id: str
    verdict: ReviewVerdict
    corrected_task_group: Optional[str] = None
    corrected_class_id: Optional[int] = Field(default=None, ge=0)
    corrected_class_name: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reasoning: Optional[str] = None

    @model_validator(mode="after")
    def validate_correction(self) -> "VLMReviewDecision":
        has_correction = (
            self.corrected_task_group is not None
            or self.corrected_class_id is not None
            or self.corrected_class_name is not None
        )
        if self.verdict == "corrected":
            if (
                self.corrected_task_group is None
                or self.corrected_class_id is None
                or self.corrected_class_name is None
            ):
                raise ValueError(
                    "corrected verdict requires corrected_task_group, corrected_class_id, "
                    "and corrected_class_name"
                )
        elif has_correction:
            raise ValueError("corrected fields are only allowed for corrected verdict")
        return self


class VLMFinding(BaseModel):
    id: str
    task_group: str
    class_id: Optional[int] = Field(default=None, ge=0)
    class_name: str
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    bbox_normalized_xyxy: Optional[Float4] = None
    crop_id: Optional[str] = None
    review_pass: ReviewPassId = "full_image"
    geometry_source: FindingGeometrySource = "vlm_full_image"
    geometry: Optional[BBoxGeometry] = None

    @model_validator(mode="after")
    def validate_finding_geometry(self) -> "VLMFinding":
        if self.bbox_normalized_xyxy is not None:
            values = tuple(float(value) for value in self.bbox_normalized_xyxy)
            if not all(math.isfinite(value) for value in values):
                raise ValueError("finding bbox coordinates must be finite")
            clipped = tuple(min(max(value, 0.0), 1.0) for value in values)
            x1, y1, x2, y2 = clipped
            if x2 <= x1 or y2 <= y1:
                raise ValueError("finding bbox must have positive width and height after clipping")
            self.bbox_normalized_xyxy = clipped
        if self.review_pass == "full_image":
            if self.crop_id is not None:
                raise ValueError("full-image finding must not include crop_id")
            if self.geometry_source != "vlm_full_image":
                raise ValueError("full-image finding requires geometry_source=vlm_full_image")
        else:
            if not self.crop_id:
                raise ValueError("crop-scan finding requires crop_id")
            if self.geometry_source != "vlm_crop":
                raise ValueError("crop-scan finding requires geometry_source=vlm_crop")
        return self


class VLMBehaviorDecision(BaseModel):
    id: str
    candidate_id: Optional[str] = None
    task_group: Literal["uncivilized_behavior"] = "uncivilized_behavior"
    class_id: int = Field(ge=0)
    class_name: str
    verdict: BehaviorVerdict
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    evidence_observation_ids: list[str] = Field(default_factory=list)
    reasoning: Optional[str] = None


class ReviewIssue(BaseModel):
    section: Literal["yolo_reviews", "new_findings", "behavior_reviews", "response"]
    item_index: Optional[int] = Field(default=None, ge=0)
    code: str
    message: str
    observation_id: Optional[str] = None
    candidate_id: Optional[str] = None
    crop_id: Optional[str] = None
    review_pass: Optional[ReviewPassId] = None


class ParsedReviewResponse(BaseModel):
    decisions: list[VLMReviewDecision] = Field(default_factory=list)
    findings: list[VLMFinding] = Field(default_factory=list)
    behaviors: list[VLMBehaviorDecision] = Field(default_factory=list)
    issues: list[ReviewIssue] = Field(default_factory=list)


class VLMReviewResult(BaseModel):
    provider: str
    model_id: str
    duration_ms: float = Field(ge=0.0)
    review_pass: ReviewPassId = "full_image"
    decisions: list[VLMReviewDecision] = Field(default_factory=list)
    findings: list[VLMFinding] = Field(default_factory=list)
    behaviors: list[VLMBehaviorDecision] = Field(default_factory=list)
    issues: list[ReviewIssue] = Field(default_factory=list)


class FusionDecision(BaseModel):
    id: str
    action: FusionAction
    observation_id: Optional[str] = None
    finding_id: Optional[str] = None
    behavior_review_id: Optional[str] = None
    original_task_group: Optional[str] = None
    original_class_name: Optional[str] = None
    final_task_group: Optional[str] = None
    final_class_name: Optional[str] = None
    geometry_source: Literal["yolo", "vlm_full_image", "vlm_crop", "none"]
    yolo_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    vlm_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    evidence_observation_ids: list[str] = Field(default_factory=list)
    reasoning: Optional[str] = None


class FusionSummary(BaseModel):
    status: FusionStatus = "not_run"
    decisions: list[FusionDecision] = Field(default_factory=list)


class RequestContext(BaseModel):
    camera_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    session_id: Optional[str] = None
    frame_index: Optional[int] = Field(default=None, ge=0)


class InputInfo(BaseModel):
    image_path: str
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    context: RequestContext = Field(default_factory=RequestContext)


class ModuleSummary(BaseModel):
    module_id: str
    task_group: str
    status: ModuleStatus
    duration_ms: float = Field(ge=0.0)
    error: Optional[str] = None


class ReviewSummary(BaseModel):
    required: bool = False
    reasons: list[str] = Field(default_factory=list)
    attempted: bool = False
    status: ReviewExecutionStatus = "not_required"
    provider: Optional[str] = None
    model_id: Optional[str] = None
    duration_ms: Optional[float] = Field(default=None, ge=0.0)
    decisions: list[VLMReviewDecision] = Field(default_factory=list)
    findings: list[VLMFinding] = Field(default_factory=list)
    behaviors: list[VLMBehaviorDecision] = Field(default_factory=list)
    issues: list[ReviewIssue] = Field(default_factory=list)
    passes: list["ReviewPassSummary"] = Field(default_factory=list)
    uncertain_policy: Literal["keep_flagged", "drop"] = "keep_flagged"
    review_failure_policy: Literal["keep_flagged", "drop_review_required"] = "keep_flagged"


class ReviewPassSummary(BaseModel):
    pass_id: ReviewPassId
    enabled: bool = True
    attempted: bool = False
    status: ReviewExecutionStatus = "not_required"
    duration_ms: Optional[float] = Field(default=None, ge=0.0)
    finding_count: int = Field(default=0, ge=0)
    issue_count: int = Field(default=0, ge=0)
    error: Optional[str] = None


class RuntimeErrorInfo(BaseModel):
    stage: Literal[
        "input",
        "module",
        "behavior",
        "detection_summary",
        "fusion",
        "review",
        "full_image_review",
        "crop_generation",
        "crop_scan_review",
        "output",
    ]
    code: str
    message: str
    module_id: Optional[str] = None


class TimingInfo(BaseModel):
    total: float = Field(ge=0.0)
    detection: Optional[float] = Field(default=None, ge=0.0)
    detection_summary: Optional[float] = Field(default=None, ge=0.0)
    review: Optional[float] = Field(default=None, ge=0.0)
    full_image_review: Optional[float] = Field(default=None, ge=0.0)
    crop_generation: Optional[float] = Field(default=None, ge=0.0)
    crop_scan_review: Optional[float] = Field(default=None, ge=0.0)
    fusion: Optional[float] = Field(default=None, ge=0.0)
    preview: Optional[float] = Field(default=None, ge=0.0)


class PipelineResponse(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str
    status: PipelineStatus
    input: InputInfo
    modules: list[ModuleSummary] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    detection_summary: Optional[DetectionSummary] = None
    review: ReviewSummary = Field(default_factory=ReviewSummary)
    fusion: FusionSummary = Field(default_factory=FusionSummary)
    errors: list[RuntimeErrorInfo] = Field(default_factory=list)
    timing_ms: TimingInfo


class ValidatedImage:
    """Decoded image kept outside the serialized response."""

    def __init__(self, path: str, image: Any, width: int, height: int) -> None:
        self.path = path
        self.image = image
        self.width = width
        self.height = height
