"""Ultralytics YOLO backend with no result-object leakage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BackendDetection, BackendPrediction, InferenceBackend
from ..schemas import ValidatedImage


def normalize_model_class_names(names: Any) -> list[str]:
    """Convert supported Ultralytics class mappings to an ID-ordered list."""
    if isinstance(names, dict):
        if any(not isinstance(class_id, int) or isinstance(class_id, bool) for class_id in names):
            raise ValueError("model class-name dict keys must be integer class IDs")
        class_ids = sorted(names)
        expected_ids = list(range(len(names)))
        if class_ids != expected_ids:
            raise ValueError(f"model class IDs must be continuous from 0: actual_ids={class_ids}")
        ordered = [names[class_id] for class_id in expected_ids]
    elif isinstance(names, (list, tuple)):
        ordered = list(names)
    else:
        raise ValueError(f"unsupported model class names type: {type(names).__name__}")

    if any(not isinstance(name, str) for name in ordered):
        raise ValueError("model class names must all be strings")
    return ordered


def validate_model_class_names(
    names: Any,
    expected_class_names: list[str],
    *,
    module_id: str,
    model_id: str,
) -> list[str]:
    """Strictly validate class IDs, count, names, and order before inference."""
    try:
        actual = normalize_model_class_names(names)
    except ValueError as exc:
        raise ValueError(
            f"class mapping validation failed for module={module_id}, model={model_id}: "
            f"expected={expected_class_names!r}, actual={names!r}; {exc}"
        ) from exc
    if actual != expected_class_names:
        raise ValueError(
            f"class mapping mismatch for module={module_id}, model={model_id}: "
            f"expected={expected_class_names!r}, actual={actual!r}"
        )
    return actual


def resolve_ultralytics_device(device: str) -> str | int:
    if device != "auto":
        if device.isdigit():
            return int(device)
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return 0
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class UltralyticsBackend(InferenceBackend):
    backend_name = "ultralytics"

    def __init__(
        self,
        model_path: Path,
        module_id: str,
        model_id: str,
        expected_class_names: list[str],
        device: str,
        confidence: float,
        iou: float,
        imgsz: int,
    ) -> None:
        self.model_path = model_path
        self.module_id = module_id
        self.model_id = model_id
        self.expected_class_names = list(expected_class_names)
        self.device = resolve_ultralytics_device(device)
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"model file does not exist: {self.model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics backend requires the optional dependency: "
                "pip install -e '.[ultralytics]'"
            ) from exc
        # The path check above prevents Ultralytics from treating a model name as a download request.
        model = YOLO(str(self.model_path))
        validate_model_class_names(
            getattr(model, "names", None),
            self.expected_class_names,
            module_id=self.module_id,
            model_id=self.model_id,
        )
        self._model = model

    def predict(self, image: ValidatedImage) -> BackendPrediction:
        if self._model is None:
            raise RuntimeError(f"backend model is not loaded: {self.model_id}")
        results = self._model.predict(
            source=image.image,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return BackendPrediction(detections=[])
        result = results[0]
        names = result.names
        detections: list[BackendDetection] = []
        boxes = result.boxes
        if boxes is not None:
            xyxy_rows = boxes.xyxy.detach().cpu().tolist()
            class_ids = boxes.cls.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            for xyxy, raw_class_id, confidence in zip(xyxy_rows, class_ids, confidences):
                class_id = int(raw_class_id)
                if isinstance(names, dict):
                    class_name = str(names.get(class_id, class_id))
                else:
                    class_name = str(names[class_id]) if 0 <= class_id < len(names) else str(class_id)
                detections.append(
                    BackendDetection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(confidence),
                        bbox_xyxy=tuple(float(value) for value in xyxy),
                    )
                )
        timing = {
            str(key): float(value)
            for key, value in (getattr(result, "speed", None) or {}).items()
            if isinstance(value, (int, float))
        }
        return BackendPrediction(detections=detections, timing_ms=timing)
