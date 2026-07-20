from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image

from wrc_park_vision.runtime.backends.base import BackendDetection, BackendPrediction, InferenceBackend
from wrc_park_vision.runtime.config import RuntimeConfig
from wrc_park_vision.runtime.schemas import (
    BBoxGeometry,
    InputInfo,
    Observation,
    ObservationSource,
    PipelineResponse,
    RequestContext,
    TimingInfo,
    ValidatedImage,
)


class FakeBackend(InferenceBackend):
    backend_name = "fake"

    def __init__(
        self,
        model_id: str,
        detections: Optional[list[BackendDetection]] = None,
        error: Optional[Exception] = None,
        load_error: Optional[Exception] = None,
        close_error: Optional[Exception] = None,
    ) -> None:
        self.model_id = model_id
        self.detections = detections or []
        self.error = error
        self.load_error = load_error
        self.close_error = close_error
        self.load_calls = 0
        self.predict_calls = 0
        self.close_calls = 0

    def load(self) -> None:
        self.load_calls += 1
        if self.load_error is not None:
            raise self.load_error

    def predict(self, image: ValidatedImage) -> BackendPrediction:
        self.predict_calls += 1
        if self.error is not None:
            raise self.error
        return BackendPrediction(detections=self.detections)

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def make_config(module_ids: tuple[str, ...] = ("prohibited", "garbage")) -> RuntimeConfig:
    modules = []
    for module_id in module_ids:
        modules.append(
            {
                "id": module_id,
                "enabled": True,
                "type": "detection",
                "task_group": module_id,
                "backend": "ultralytics",
                "model_path": Path(f"{module_id}.pt"),
                "model_id": f"{module_id}_model",
                "expected_class_names": [f"{module_id}_class"],
            }
        )
    return RuntimeConfig.model_validate({"modules": modules})


def write_test_image(path: Path, size: tuple[int, int] = (100, 80)) -> Path:
    Image.new("RGB", size, "white").save(path)
    return path


def make_observation(
    task_group: str,
    module_id: str,
    class_id: int,
    class_name: str,
    confidence: float,
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int] = (100, 80),
) -> Observation:
    return Observation(
        kind="detection",
        task_group=task_group,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        source=ObservationSource(module_id=module_id, backend="fake", model_id=f"{module_id}_model"),
        geometry=BBoxGeometry.from_xyxy(bbox, image_size[0], image_size[1]),
    )


def make_response(observations: list[Observation], image_path: Path) -> PipelineResponse:
    with Image.open(image_path) as image:
        width, height = image.size
    return PipelineResponse(
        request_id="request-test",
        status="success",
        input=InputInfo(
            image_path=str(image_path),
            width=width,
            height=height,
            context=RequestContext(),
        ),
        observations=observations,
        timing_ms=TimingInfo(total=1.0),
    )
