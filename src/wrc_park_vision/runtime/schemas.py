"""Stable runtime input and output schemas."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


Float4 = tuple[float, float, float, float]
PipelineStatus = Literal["success", "partial_success", "failure"]
ModuleStatus = Literal["success", "failure"]
ReviewStatus = Literal["not_required", "pending", "confirmed", "rejected"]


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
    kind: Literal["detection", "segmentation", "pose", "region", "relation"]
    task_group: str
    class_id: Optional[int] = None
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: ObservationSource
    track_id: Optional[str] = None
    geometry: Geometry
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
        return self


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


class RuntimeErrorInfo(BaseModel):
    stage: Literal["input", "module", "fusion", "review", "output"]
    code: str
    message: str
    module_id: Optional[str] = None


class TimingInfo(BaseModel):
    total: float = Field(ge=0.0)


class PipelineResponse(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str
    status: PipelineStatus
    input: InputInfo
    modules: list[ModuleSummary] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    review: ReviewSummary = Field(default_factory=ReviewSummary)
    errors: list[RuntimeErrorInfo] = Field(default_factory=list)
    timing_ms: TimingInfo


class ValidatedImage:
    """Decoded image kept outside the serialized response."""

    def __init__(self, path: str, image: Any, width: int, height: int) -> None:
        self.path = path
        self.image = image
        self.width = width
        self.height = height
