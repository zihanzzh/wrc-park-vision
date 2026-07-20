"""Generic object-detection task module."""

from __future__ import annotations

from ..backends.base import InferenceBackend
from ..schemas import BBoxGeometry, Observation, ObservationSource, ValidatedImage
from .base import TaskModule


class DetectionModule(TaskModule):
    def __init__(self, module_id: str, task_group: str, model_id: str, backend: InferenceBackend) -> None:
        self.module_id = module_id
        self.task_group = task_group
        self.model_id = model_id
        self.backend = backend

    def load(self) -> None:
        self.backend.load()

    def run(self, image: ValidatedImage) -> list[Observation]:
        prediction = self.backend.predict(image)
        observations: list[Observation] = []
        for detection in prediction.detections:
            geometry = BBoxGeometry.from_xyxy(detection.bbox_xyxy, image.width, image.height)
            metadata = dict(detection.metadata)
            if prediction.timing_ms:
                metadata["backend_timing_ms"] = prediction.timing_ms
            observations.append(
                Observation(
                    kind="detection",
                    task_group=self.task_group,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    source=ObservationSource(
                        module_id=self.module_id,
                        backend=self.backend.backend_name,
                        model_id=self.model_id,
                    ),
                    geometry=geometry,
                    metadata=metadata,
                )
            )
        return observations

    def close(self) -> None:
        self.backend.close()

