"""Sequential, failure-isolated multi-module runtime orchestration."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError

from .backends import (
    TensorRTBackend,
    UltralyticsBackend,
    YOLOWorldBackend,
    YOLOWorldClassDefinition,
)
from .config import ModuleSettings, RuntimeConfig
from .detection_summary import build_detection_summary
from .fusion import fuse_review_results, merge_and_mark_conflicts, prepare_observations
from .modules import BehaviorModule, BehaviorPipeline, DetectionModule, TaskModule
from .review import (
    MultiImageReviewRequest,
    ReviewCoordinator,
    ReviewProvider,
    generate_important_crops,
    select_review_candidates,
)
from .schemas import (
    BBoxGeometry,
    DetectionSummary,
    FusionSummary,
    InputInfo,
    ModuleSummary,
    PipelineResponse,
    RequestContext,
    ReviewSummary,
    ReviewIssue,
    ReviewPassSummary,
    RuntimeErrorInfo,
    TimingInfo,
    ValidatedImage,
    VLMReviewResult,
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
    if settings.model_path is None:
        raise ValueError(f"detection module {settings.id} requires model_path")
    if settings.backend == "ultralytics":
        if not settings.expected_class_names:
            raise ValueError(f"detection module {settings.id} requires expected_class_names")
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
    elif settings.backend == "yolo_world":
        if not settings.open_vocabulary_classes:
            raise ValueError(f"YOLO-World module {settings.id} requires open_vocabulary_classes")
        backend = YOLOWorldBackend(
            model_path=settings.model_path,
            module_id=settings.id,
            model_id=settings.model_id,
            classes=[
                YOLOWorldClassDefinition(
                    task_group=item.task_group,
                    class_id=item.class_id,
                    class_name=item.class_name,
                    prompts=tuple(item.prompts),
                )
                for item in settings.open_vocabulary_classes
            ],
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
        if not settings.enabled:
            continue
        if settings.backend == "yolo_world" and settings.open_vocabulary_classes:
            grouped: dict[str, list[tuple[int, str]]] = {}
            for item in settings.open_vocabulary_classes:
                grouped.setdefault(item.task_group, []).append((item.class_id, item.class_name))
            for task_group, entries in grouped.items():
                names = catalog.setdefault(task_group, [])
                for _, class_name in sorted(entries):
                    if class_name not in names:
                        names.append(class_name)
        elif settings.expected_class_names:
            names = catalog.setdefault(settings.task_group, [])
            for class_name in settings.expected_class_names:
                if class_name not in names:
                    names.append(class_name)
    return catalog


def build_visual_class_guide(
    config: RuntimeConfig,
) -> dict[str, dict[str, dict[str, object]]]:
    guide: dict[str, dict[str, dict[str, object]]] = {}
    for settings in config.modules:
        if not settings.enabled or not settings.open_vocabulary_classes:
            continue
        for item in settings.open_vocabulary_classes:
            if item.visual_description is None and not item.distinguishing_rules:
                continue
            details: dict[str, object] = {}
            if item.visual_description is not None:
                details["visual"] = item.visual_description
            if item.distinguishing_rules:
                details["distinguish"] = list(item.distinguishing_rules)
            guide.setdefault(item.task_group, {})[item.class_name] = details
    return guide


def _map_finding_geometries(
    result: VLMReviewResult,
    image: ValidatedImage,
    *,
    require_bbox: bool = False,
) -> VLMReviewResult:
    mapped = []
    issues = list(result.issues)
    for index, finding in enumerate(result.findings):
        if finding.bbox_normalized_xyxy is None:
            if require_bbox:
                issues.append(
                    ReviewIssue(
                        section="new_findings",
                        item_index=index,
                        code="missing_finding_geometry",
                        message="finding requires bbox_normalized_xyxy",
                        crop_id=finding.crop_id,
                        review_pass=finding.review_pass,
                    )
                )
                continue
            mapped.append(finding)
            continue
        try:
            x1, y1, x2, y2 = finding.bbox_normalized_xyxy
            geometry = BBoxGeometry.from_xyxy(
                (
                    x1 * image.width,
                    y1 * image.height,
                    x2 * image.width,
                    y2 * image.height,
                ),
                image.width,
                image.height,
            )
        except Exception as exc:
            issues.append(
                ReviewIssue(
                    section="new_findings",
                    item_index=index,
                    code="invalid_finding_geometry",
                    message=str(exc) or exc.__class__.__name__,
                    crop_id=finding.crop_id,
                    review_pass=finding.review_pass,
                )
            )
            continue
        mapped.append(finding.model_copy(update={"geometry": geometry}))
    return result.model_copy(update={"findings": mapped, "issues": issues})


class RuntimePipeline:
    def __init__(
        self,
        config: RuntimeConfig,
        modules: Optional[list[TaskModule]] = None,
        review_provider: Optional[ReviewProvider] = None,
    ) -> None:
        self.config = config
        self.behavior = BehaviorPipeline(config.behavior)
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
            visual_class_guide = build_visual_class_guide(config)
            if review_provider is None and config.review.provider.enabled:
                review_provider = Qwen25VLProvider(
                    config.review.provider,
                    class_catalog,
                    visual_class_guide=visual_class_guide,
                    review_settings=config.review,
                )
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
                status="failed",
                input=InputInfo(image_path=str(image_path), width=0, height=0, context=request_context),
                review={"required": False, "reasons": []},
                errors=[RuntimeErrorInfo(stage="input", code="invalid_image", message=str(exc))],
                timing_ms=TimingInfo(total=(time.perf_counter() - started) * 1000.0),
            )

        summaries: list[ModuleSummary] = []
        errors: list[RuntimeErrorInfo] = []
        observations = []
        detection_started = time.perf_counter()
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

        detection_duration = (time.perf_counter() - detection_started) * 1000.0
        successful_modules = sum(summary.status == "success" for summary in summaries)
        postprocessing_failed = False
        detection_summary_duration = None
        candidate_selection_duration = None
        review_duration = None
        crop_generation_duration = None
        multi_image_review_duration = None
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

        behavior_candidates = []
        try:
            behavior_candidates = self.behavior.generate_candidates(current_observations)
        except Exception as exc:
            errors.append(
                RuntimeErrorInfo(
                    stage="behavior",
                    code="behavior_candidate_failure",
                    message=str(exc) or exc.__class__.__name__,
                )
            )
            postprocessing_failed = True

        summary_started = time.perf_counter()
        detection_summary: DetectionSummary | None = None
        try:
            detection_summary = build_detection_summary(
                current_observations,
                self.config.review,
                behavior_classes=self.behavior.class_summaries(),
                behavior_candidates=behavior_candidates,
            )
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
            reviewed, review_summary = self.review.prepare(
                current_observations,
                summaries,
                detection_summary,
            )
            review_summary.uncertain_policy = self.config.fusion.uncertain_policy
            review_summary.review_failure_policy = self.config.fusion.review_failure_policy
            provider = self.review.provider
            selection_started = time.perf_counter()
            review_candidates = []
            try:
                review_candidates = select_review_candidates(
                    reviewed,
                    behavior_candidates,
                    self.config.review.candidate_selection,
                    self.config.review.low_confidence_threshold,
                )
            except Exception as exc:
                errors.append(
                    RuntimeErrorInfo(
                        stage="candidate_selection",
                        code="candidate_selection_failure",
                        message=str(exc) or exc.__class__.__name__,
                    )
                )
                postprocessing_failed = True
            candidate_selection_duration = (
                time.perf_counter() - selection_started
            ) * 1000.0

            crop_started = time.perf_counter()
            important_crops = []
            try:
                important_crops = generate_important_crops(
                    image,
                    review_candidates,
                    self.config.review.important_crops,
                )
            except Exception as exc:
                errors.append(
                    RuntimeErrorInfo(
                        stage="crop_generation",
                        code="crop_generation_failure",
                        message=str(exc) or exc.__class__.__name__,
                    )
                )
                postprocessing_failed = True
            crop_generation_duration = (
                time.perf_counter() - crop_started
            ) * 1000.0

            if provider is not None and self.config.review.multi_image.enabled:
                multi_started = time.perf_counter()
                try:
                    result = provider.review_multi_image(
                        MultiImageReviewRequest(
                            image=image,
                            summary=detection_summary,
                            candidates=tuple(review_candidates),
                            crops=tuple(important_crops),
                        )
                    )
                    result = _map_finding_geometries(
                        result,
                        image,
                        require_bbox=self.config.review.require_finding_bbox,
                    )
                    reviewed, review_summary = self.review.apply_result(
                        reviewed,
                        review_summary,
                        result,
                    )
                except Exception as exc:
                    message = str(exc) or exc.__class__.__name__
                    errors.append(
                        RuntimeErrorInfo(
                            stage="multi_image_review",
                            code="multi_image_review_failure",
                            message=message,
                        )
                    )
                    if "review_failure" not in review_summary.reasons:
                        review_summary.reasons.append("review_failure")
                    review_summary.attempted = True
                    review_summary.status = "failed"
                    review_summary.passes.append(
                        ReviewPassSummary(
                            pass_id="multi_image",
                            attempted=True,
                            status="failed",
                            error=message,
                        )
                    )
                    postprocessing_failed = True
                multi_image_review_duration = (
                    time.perf_counter() - multi_started
                ) * 1000.0
                review_summary.passes[-1].duration_ms = multi_image_review_duration
            else:
                review_summary.passes = [
                    ReviewPassSummary(
                        pass_id="multi_image",
                        enabled=self.config.review.multi_image.enabled,
                        status="not_required",
                    )
                ]
        review_duration = (time.perf_counter() - review_started) * 1000.0

        behavior_observations = []
        try:
            behavior_observations = self.behavior.build_observations(review_summary)
        except Exception as exc:
            errors.append(
                RuntimeErrorInfo(
                    stage="behavior",
                    code="behavior_result_failure",
                    message=str(exc) or exc.__class__.__name__,
                )
            )
            postprocessing_failed = True

        fusion_started = time.perf_counter()
        try:
            fused_observations, fusion_summary = fuse_review_results(
                reviewed,
                review_summary,
                behavior_observations=behavior_observations,
                finding_iou_threshold=self.config.fusion.finding_iou_threshold,
            )
        except Exception as exc:
            errors.append(RuntimeErrorInfo(stage="fusion", code="review_fusion_failure", message=str(exc)))
            fused_observations = [
                observation.model_copy(deep=True)
                for observation in [*reviewed, *behavior_observations]
            ]
            fusion_summary = FusionSummary(status="fallback")
            postprocessing_failed = True
        fusion_duration = (time.perf_counter() - fusion_started) * 1000.0

        if successful_modules == 0:
            status = "failed"
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
                detection=detection_duration,
                detection_summary=detection_summary_duration,
                candidate_selection=candidate_selection_duration,
                review=review_duration,
                crop_generation=crop_generation_duration,
                multi_image_review=multi_image_review_duration,
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
