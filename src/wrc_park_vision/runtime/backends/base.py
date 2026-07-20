"""Backend contract isolated from task modules and pipeline orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..schemas import ValidatedImage


@dataclass(frozen=True)
class BackendDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendPrediction:
    detections: list[BackendDetection]
    timing_ms: dict[str, float] = field(default_factory=dict)


class InferenceBackend(ABC):
    backend_name: str
    model_id: str

    @abstractmethod
    def load(self) -> None:
        """Load model resources once during pipeline startup."""

    @abstractmethod
    def predict(self, image: ValidatedImage) -> BackendPrediction:
        """Run inference on an already validated and decoded image."""

    def close(self) -> None:
        """Release backend resources when supported."""

