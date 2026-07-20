"""TensorRT backend placeholder with an explicit unsupported state."""

from __future__ import annotations

from pathlib import Path

from .base import BackendPrediction, InferenceBackend
from ..schemas import ValidatedImage


class TensorRTBackend(InferenceBackend):
    backend_name = "tensorrt"

    def __init__(
        self,
        model_path: Path,
        model_id: str,
        device: str,
        confidence: float,
        iou: float,
        imgsz: int,
    ) -> None:
        self.model_path = model_path
        self.model_id = model_id
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz

    def load(self) -> None:
        raise NotImplementedError("TensorRT backend is not implemented; Thor integration is a future phase")

    def predict(self, image: ValidatedImage) -> BackendPrediction:
        raise NotImplementedError("TensorRT backend is not implemented")
