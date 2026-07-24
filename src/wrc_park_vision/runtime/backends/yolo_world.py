"""Ultralytics YOLO-World backend for grouped object-level categories."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import BackendDetection, BackendPrediction, InferenceBackend
from .ultralytics import resolve_ultralytics_device, validate_model_class_names
from ..schemas import ValidatedImage


@dataclass(frozen=True)
class YOLOWorldClassDefinition:
    task_group: str
    class_id: int
    class_name: str
    prompts: tuple[str, ...]


@dataclass(frozen=True)
class _PromptTarget:
    task_group: str
    class_id: int
    class_name: str
    prompt: str


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    if intersection <= 0.0:
        return 0.0
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


class YOLOWorldBackend(InferenceBackend):
    backend_name = "yolo_world"

    def __init__(
        self,
        model_path: Path,
        module_id: str,
        model_id: str,
        classes: list[YOLOWorldClassDefinition],
        device: str,
        confidence: float,
        iou: float,
        imgsz: int,
    ) -> None:
        if not classes:
            raise ValueError("YOLO-World requires at least one object class definition")
        if any(item.task_group == "garbage" for item in classes):
            raise ValueError(
                "YOLO-World must not provide garbage detections; "
                "use the dedicated Ultralytics garbage module"
            )
        self.model_path = model_path
        self.module_id = module_id
        self.model_id = model_id
        self.classes = list(classes)
        self.device = resolve_ultralytics_device(device)
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self._model: Any = None
        self._prompt_targets = [
            _PromptTarget(item.task_group, item.class_id, item.class_name, prompt)
            for item in self.classes
            for prompt in item.prompts
        ]

    @property
    def prompts(self) -> list[str]:
        return [target.prompt for target in self._prompt_targets]

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"model file does not exist: {self.model_path}")
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:
            raise RuntimeError(
                "YOLO-World backend requires the optional dependency: "
                "pip install -e '.[yolo-world]'"
            ) from exc

        model = YOLOWorld(str(self.model_path))
        prompts = self.prompts
        try:
            model.set_classes(list(prompts))
        except ImportError as exc:
            raise RuntimeError(
                "YOLO-World text prompts require the Ultralytics CLIP dependency: "
                "pip install -e '.[yolo-world]'"
            ) from exc
        validate_model_class_names(
            getattr(model, "names", None),
            prompts,
            module_id=self.module_id,
            model_id=self.model_id,
        )

        # Text features are cached in the detector; the CLIP encoder is not needed per image.
        inner_model = getattr(model, "model", None)
        if inner_model is not None and hasattr(inner_model, "clip_model"):
            inner_model.clip_model = None
            gc.collect()
        self._model = model

    def _deduplicate_aliases(self, detections: list[BackendDetection]) -> list[BackendDetection]:
        kept: list[BackendDetection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            duplicate = any(
                existing.task_group == detection.task_group
                and existing.class_id == detection.class_id
                and _bbox_iou(existing.bbox_xyxy, detection.bbox_xyxy) >= self.iou
                for existing in kept
            )
            if not duplicate:
                kept.append(detection)
        return kept

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
        detections: list[BackendDetection] = []
        boxes = result.boxes
        if boxes is not None:
            xyxy_rows = boxes.xyxy.detach().cpu().tolist()
            prompt_ids = boxes.cls.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            for xyxy, raw_prompt_id, confidence in zip(xyxy_rows, prompt_ids, confidences):
                prompt_id = int(raw_prompt_id)
                if not 0 <= prompt_id < len(self._prompt_targets):
                    raise ValueError(
                        f"YOLO-World returned unknown prompt class ID {prompt_id} "
                        f"for module={self.module_id}, model={self.model_id}"
                    )
                target = self._prompt_targets[prompt_id]
                detections.append(
                    BackendDetection(
                        class_id=target.class_id,
                        class_name=target.class_name,
                        confidence=float(confidence),
                        bbox_xyxy=tuple(float(value) for value in xyxy),
                        metadata={
                            "matched_prompt": target.prompt,
                            "raw_prompt_class_id": prompt_id,
                        },
                        task_group=target.task_group,
                    )
                )

        timing = {
            str(key): float(value)
            for key, value in (getattr(result, "speed", None) or {}).items()
            if isinstance(value, (int, float))
        }
        return BackendPrediction(
            detections=self._deduplicate_aliases(detections),
            timing_ms=timing,
        )

    def close(self) -> None:
        self._model = None
        gc.collect()
