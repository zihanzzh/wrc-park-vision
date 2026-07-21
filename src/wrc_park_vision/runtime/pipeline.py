"""Sequential, failure-isolated multi-module runtime orchestration."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError

from .backends import TensorRTBackend, UltralyticsBackend
from .config import ModuleSettings, RuntimeConfig
from .detection_summary import build_detection_summary
from .fusion import fuse_review_results, merge_and_mark_conflicts, prepare_observations
from .modules import BehaviorModule, DetectionModule, TaskModule
from .review import ReviewCoordinator, ReviewProvider
from .schemas import (
    DetectionSummary,
    FusionSummary,
    InputInfo,
    ModuleSummary,
    PipelineResponse,
    RequestContext,
    ReviewSummary,
    RuntimeErrorInfo,
    TimingInfo,
    ValidatedImage,
)
from .vlm import Qwen25VLProvider


def validate_image(image_path: Path) -> ValidatedImage:
    path = image_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"input image does not exist: {path}")
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"input file is not a decodable image: {path}") from exc
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"input image has invalid dimensions: {width}x{height}")
    return ValidatedImage(str(path), image, width, height)


def _build_detection_module(settings: ModuleSettings) -> DetectionModule:
    if settings.model_path is None or not settings.expected_class_names:
        raise ValueError(f"detection module {settings.id} requires model_path and expected_class_names")
    if settings.backend == "ultralytics":
        backend = UltralyticsBackend(
            model_path=settings.model_path,
            module_id=settings.id,
            model_id=settings.model_id,
            expected_class_names=settings.expected_class_names,
            device=settings.device,
            confidence=settings.confidence,
            iou=settings.iou,
            imgsz=settings.imgsz,
        )
    elif settings.backend == "tensorrt":
        backend = TensorRTBackend(
            model_path=settings.model_path,
            model_id=settings.model_id,
            device=settings.device,
            confidence=settings.confidence,
            iou=settings.iou,
            imgsz=settings.imgsz,
        )
    else:
        raise ValueError(f"unsupported backend: {settings.backend}")
    return DetectionModule(settings.id, settings.task_group, settings.model_id, backend)


def build_modules(config: RuntimeConfig) -> list[TaskModule]:
    modules: list[TaskModule] = []
    for settings in config.modules:
        if not settings.enabled:
            continue
        if settings.type == "detection":
            modules.append(_build_detection_module(settings))
        elif settings.type == "behavior":
            modules.append(BehaviorModule(settings.id, settings.task_group))
        else:
            raise ValueError(f"unsupported module type: {settings.type}")
    return modules


def build_class_catalog(config: RuntimeConfig) -> dict[str, list[str]]:
    catalog: dict[str, list[str]] = {}
    for settings in config.modules:
        if not settings.enabled or not settings.expected_class_names:
            continue
        names = catalog.setdefault(settings.task_group, [])
        for class_name in settings.expected_class_names:
            if class_name not in names:
                names.append(class_name)
    return catalog


class RuntimePipeline:
    def __init__(
        self,
        config: RuntimeConfig,
        modules: Optional[list[TaskModule]] = None,
        review_provider: Optional[ReviewProvider] = None,
    ) -> None:
        self.config = config
        self.modules = modules if modules is not None else build_modules(config)
        expected_ids = [settings.id for settings in config.modules if settings.enabled]
        actual_ids = [module.module_id for module in self.modules]
        if actual_ids != expected_ids:
            raise ValueError(f"module order/ids do not match enabled config: expected={expected_ids}, actual={actual_ids}")
        loaded_modules: list[TaskModule] = []
        try:
            for module in self.modules:
                module.load()
                loaded_modules.append(module)
        except Exception:
            for loaded_module in loaded_modules:
                try:
                    loaded_module.close()
                except Exception:
                    pass
            raise
        try:
            class_catalog = build_class_catalog(config)
            if review_provider is None and config.review.provider.enabled:
                review_provider = Qwen25VLProvider(config.review.provider, class_catalog)
            self.review = ReviewCoordinator(config.review, review_provider)
        except Exception:
            for loaded_module in loaded_modules:
                try:
                    loaded_module.close()
                except Exception:
                    pass
            raise

    def process(
        self,
        image_path: Path,
        context: Optional[RequestContext] = None,
        request_id: Optional[str] = None,
    ) -> PipelineResponse:
        started = time.perf_counter()
        resolved_request_id = request_id or str(uuid.uuid4())
        request_context = context or RequestContext()
        try:
            image = validate_image(image_path)
        except Exception as exc:
            return PipelineResponse(
                request_id=resolved_request_id,
                status="failure",
                input=InputInfo(image_path=str(image_path), width=0, height=0, context=request_context),
                review={"required": False, "reasons": []},
                errors=[RuntimeErrorInfo(stage="input", code="invalid_image", message=str(exc))],
                timing_ms=TimingInfo(total=(time.perf_counter() - started) * 1000.0),
            )

        summaries: list[ModuleSummary] = []
        errors: list[RuntimeErrorInfo] = []
        observations = []
        for module in self.modules:
            module_started = time.perf_counter()
            try:
                observations.extend(module.run(image))
                summaries.append(
                    ModuleSummary(
                        module_id=module.module_id,
                        task_group=module.task_group,
                        status="success",
                        duration_ms=(time.perf_counter() - module_started) * 1000.0,
                    )
                )
            except Exception as exc:
                message = str(exc) or exc.__class__.__name__
                summaries.append(
                    ModuleSummary(
                        module_id=module.module_id,
                        task_group=module.task_group,
                        status="failure",
                        duration_ms=(time.perf_counter() - module_started) * 1000.0,
                        error=message,
                    )
                )
                errors.append(
                    RuntimeErrorInfo(
                        stage="module",
                        code="module_failure",
                        message=message,
                        module_id=module.module_id,
                    )
                )

        successful_modules = sum(summary.status == "success" for summary in summaries)
        postprocessing_failed = False
        detection_summary_duration = None
        review_duration = None
        fusion_duration = None
        try:
            current_observations = merge_and_mark_conflicts(
                observations,
                self.config.review.cross_model_iou_threshold,
            )
        except Exception as exc:
            errors.append(RuntimeErrorInfo(stage="fusion", code="fusion_failure", message=str(exc)))
            current_observations = prepare_observations(observations)
            postprocessing_failed = True

        summary_started = time.perf_counter()
        detection_summary: DetectionSummary | None = None
        try:
            detection_summary = build_detection_summary(current_observations, self.config.review)
            detection_summary_duration = (time.perf_counter() - summary_started) * 1000.0
        except Exception as exc:
            errors.append(
                RuntimeErrorInfo(stage="detection_summary", code="detection_summary_failure", message=str(exc))
            )
            detection_summary_duration = (time.perf_counter() - summary_started) * 1000.0
            postprocessing_failed = True

        review_started = time.perf_counter()
        if detection_summary is None:
            reviewed, policy_summary = self.review.policy.apply(current_observations, summaries)
            review_summary = ReviewSummary(
                required=policy_summary.required,
                reasons=[*policy_summary.reasons, "detection_summary_failure"],
                attempted=False,
                status="failed",
            )
        else:
            try:
                reviewed, review_summary = self.review.apply(
                    image,
                    current_observations,
                    summaries,
                    detection_summary,
                )
            except Exception as exc:
                errors.append(RuntimeErrorInfo(stage="review", code="review_failure", message=str(exc)))
                reviewed, policy_summary = self.review.policy.apply(current_observations, summaries)
                provider = self.review.provider
                review_summary = ReviewSummary(
                    required=True,
                    reasons=[*policy_summary.reasons, "review_failure"],
                    attempted=provider is not None,
                    status="failed",
                    provider=getattr(provider, "provider_name", provider.__class__.__name__ if provider else None),
                    model_id=getattr(provider, "model_id", None),
                )
                postprocessing_failed = True
        review_duration = (time.perf_counter() - review_started) * 1000.0

        fusion_started = time.perf_counter()
        try:
            fused_observations, fusion_summary = fuse_review_results(reviewed, review_summary)
        except Exception as exc:
            errors.append(RuntimeErrorInfo(stage="fusion", code="review_fusion_failure", message=str(exc)))
            fused_observations = [observation.model_copy(deep=True) for observation in reviewed]
            fusion_summary = FusionSummary(status="fallback")
            postprocessing_failed = True
        fusion_duration = (time.perf_counter() - fusion_started) * 1000.0

        if successful_modules == 0:
            status = "failure"
        elif successful_modules < len(summaries) or postprocessing_failed:
            status = "partial_success"
        else:
            status = "success"

        return PipelineResponse(
            request_id=resolved_request_id,
            status=status,
            input=InputInfo(
                image_path=image.path,
                width=image.width,
                height=image.height,
                context=request_context,
            ),
            modules=summaries,
            observations=fused_observations,
            detection_summary=detection_summary,
            review=review_summary,
            fusion=fusion_summary,
            errors=errors,
            timing_ms=TimingInfo(
                total=(time.perf_counter() - started) * 1000.0,
                detection_summary=detection_summary_duration,
                review=review_duration,
                fusion=fusion_duration,
            ),
        )

    def close(self) -> None:
        for module in self.modules:
            try:
                module.close()
            except Exception:
                pass
        if self.review.provider is not None:
            try:
                self.review.provider.close()
            except Exception:
                pass

    def __enter__(self) -> "RuntimePipeline":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
