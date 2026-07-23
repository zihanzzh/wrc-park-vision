"""Runtime YAML loading, environment expansion, and startup validation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


ENV_PATTERN = re.compile(r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))")
BEHAVIOR_CLASS_NAMES = (
    "trampling_grass",
    "smoking",
    "blocking_fire_lane",
    "standing_or_lying_on_bench",
)
BEHAVIOR_ACTION_CLASS_NAMES = {
    *BEHAVIOR_CLASS_NAMES,
    "standing_on_bench",
    "lying_on_bench",
}


class ConfigError(ValueError):
    """Raised when runtime configuration cannot be used safely."""


class RuntimeSettings(BaseModel):
    execution: Literal["sequential"] = "sequential"
    output_dir: Path = Path("runtime_outputs")


class OpenVocabularyClassSettings(BaseModel):
    task_group: str = Field(min_length=1)
    class_id: int = Field(ge=0)
    class_name: str = Field(min_length=1)
    prompts: list[str] = Field(min_length=1)
    visual_description: Optional[str] = None
    distinguishing_rules: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_prompts(self) -> "OpenVocabularyClassSettings":
        if not self.task_group.strip() or not self.class_name.strip():
            raise ValueError("open-vocabulary task_group and class_name must not be blank")
        if any(not prompt.strip() for prompt in self.prompts):
            raise ValueError(f"open-vocabulary class {self.class_name} contains a blank prompt")
        if len(self.prompts) != len(set(self.prompts)):
            raise ValueError(f"open-vocabulary class {self.class_name} contains duplicate prompts")
        if self.visual_description is not None and not self.visual_description.strip():
            raise ValueError(f"open-vocabulary class {self.class_name} has a blank visual_description")
        if any(not rule.strip() for rule in self.distinguishing_rules):
            raise ValueError(f"open-vocabulary class {self.class_name} contains a blank distinguishing rule")
        if len(self.distinguishing_rules) != len(set(self.distinguishing_rules)):
            raise ValueError(f"open-vocabulary class {self.class_name} contains duplicate distinguishing rules")
        if (
            self.task_group == "uncivilized_behavior"
            and self.class_name in BEHAVIOR_ACTION_CLASS_NAMES
        ):
            raise ValueError(
                f"behavior action {self.class_name} must be handled by the Behavior Pipeline, "
                "not configured as a YOLO-World object class"
            )
        return self


class ModuleSettings(BaseModel):
    id: str = Field(min_length=1)
    enabled: bool = True
    type: Literal["detection", "behavior"]
    task_group: str = Field(min_length=1)
    backend: Literal["ultralytics", "tensorrt", "yolo_world"]
    model_path: Optional[Path] = None
    model_id: str = Field(min_length=1)
    expected_class_names: Optional[list[str]] = None
    open_vocabulary_classes: Optional[list[OpenVocabularyClassSettings]] = None
    device: str = "auto"
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.7, ge=0.0, le=1.0)
    imgsz: int = Field(default=640, gt=0)

    @model_validator(mode="after")
    def validate_expected_class_names(self) -> "ModuleSettings":
        names = self.expected_class_names
        if names is not None:
            if not names:
                raise ValueError(f"module {self.id} expected_class_names must not be empty")
            if any(not name.strip() for name in names):
                raise ValueError(f"module {self.id} expected_class_names must not contain blank names")
            if len(names) != len(set(names)):
                raise ValueError(f"module {self.id} expected_class_names must not contain duplicates")
        open_classes = self.open_vocabulary_classes
        if self.backend == "yolo_world":
            if names is not None:
                raise ValueError(
                    f"YOLO-World module {self.id} must use open_vocabulary_classes instead of expected_class_names"
                )
            if self.enabled and self.type == "detection" and not open_classes:
                raise ValueError(
                    f"enabled YOLO-World module {self.id} requires non-empty open_vocabulary_classes"
                )
        else:
            if open_classes is not None:
                raise ValueError(
                    f"module {self.id} may only use open_vocabulary_classes with backend yolo_world"
                )
            if self.enabled and self.type == "detection" and not names:
                raise ValueError(f"enabled detection module {self.id} requires non-empty expected_class_names")

        if open_classes:
            pairs = [(item.task_group, item.class_name) for item in open_classes]
            if len(pairs) != len(set(pairs)):
                raise ValueError(f"YOLO-World module {self.id} contains duplicate task_group/class_name pairs")
            prompts = [prompt for item in open_classes for prompt in item.prompts]
            if len(prompts) != len(set(prompts)):
                raise ValueError(f"YOLO-World module {self.id} contains prompts assigned to multiple classes")
            task_groups = {item.task_group for item in open_classes}
            for task_group in task_groups:
                group_items = [item for item in open_classes if item.task_group == task_group]
                class_ids = sorted(item.class_id for item in group_items)
                if class_ids != list(range(len(group_items))):
                    raise ValueError(
                        f"YOLO-World module {self.id} class IDs for {task_group} "
                        f"must be continuous from 0: actual_ids={class_ids}"
                    )
                class_names = [item.class_name for item in group_items]
                if len(class_names) != len(set(class_names)):
                    raise ValueError(
                        f"YOLO-World module {self.id} contains duplicate class names for {task_group}"
                    )
        return self


class ReviewSettings(BaseModel):
    low_confidence_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    cross_model_iou_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    review_cross_task_overlap: bool = True
    review_module_failure: bool = True
    provider: "ReviewProviderSettings" = Field(default_factory=lambda: ReviewProviderSettings())


class ReviewProviderSettings(BaseModel):
    enabled: bool = False
    type: Literal["qwen2_5_vl"] = "qwen2_5_vl"
    endpoint: Optional[str] = None
    model_id: Optional[str] = None
    api_key_env: Optional[str] = None
    timeout_seconds: float = Field(default=10.0, gt=0.0)
    max_tokens: int = Field(default=1200, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def validate_enabled_provider(self) -> "ReviewProviderSettings":
        if self.enabled and (not self.endpoint or not self.model_id):
            raise ValueError("enabled review provider requires endpoint and model_id")
        return self


class BehaviorClassSettings(BaseModel):
    class_id: int = Field(ge=0)
    class_name: Literal[
        "trampling_grass",
        "smoking",
        "blocking_fire_lane",
        "standing_or_lying_on_bench",
    ]
    required_object_classes: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_required_objects(self) -> "BehaviorClassSettings":
        if any(not name.strip() for name in self.required_object_classes):
            raise ValueError(f"behavior class {self.class_name} contains a blank required object")
        if len(self.required_object_classes) != len(set(self.required_object_classes)):
            raise ValueError(f"behavior class {self.class_name} contains duplicate required objects")
        return self


class BehaviorSettings(BaseModel):
    enabled: bool = False
    module_id: str = Field(default="behavior_pipeline", min_length=1)
    task_group: Literal["uncivilized_behavior"] = "uncivilized_behavior"
    object_task_group: str = Field(default="uncivilized_behavior", min_length=1)
    classes: list[BehaviorClassSettings] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_classes(self) -> "BehaviorSettings":
        if not self.enabled:
            return self
        if not self.module_id.strip() or not self.object_task_group.strip():
            raise ValueError("enabled behavior pipeline requires non-blank module and object task groups")
        names = [item.class_name for item in self.classes]
        if names != list(BEHAVIOR_CLASS_NAMES):
            raise ValueError(
                "enabled behavior pipeline requires the four canonical classes in order: "
                + ", ".join(BEHAVIOR_CLASS_NAMES)
            )
        class_ids = [item.class_id for item in self.classes]
        if class_ids != list(range(len(BEHAVIOR_CLASS_NAMES))):
            raise ValueError("behavior class IDs must be continuous from 0 in canonical order")
        return self


class PreviewSettings(BaseModel):
    enabled: bool = True
    task_group_colors: dict[str, str] = Field(
        default_factory=lambda: {
            "prohibited_items": "#dc2626",
            "garbage": "#16a34a",
            "uncivilized_behavior": "#2563eb",
            "unknown": "#d97706",
        }
    )

    @model_validator(mode="after")
    def validate_colors(self) -> "PreviewSettings":
        for task_group, color in self.task_group_colors.items():
            if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                raise ValueError(f"invalid preview color for {task_group}: {color}")
        return self


class RuntimeConfig(BaseModel):
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    modules: list[ModuleSettings]
    behavior: BehaviorSettings = Field(default_factory=BehaviorSettings)
    review: ReviewSettings = Field(default_factory=ReviewSettings)
    preview: PreviewSettings = Field(default_factory=PreviewSettings)

    @model_validator(mode="after")
    def validate_modules(self) -> "RuntimeConfig":
        ids = [module.id for module in self.modules]
        if len(ids) != len(set(ids)):
            raise ValueError("module ids must be unique")
        if not any(module.enabled for module in self.modules):
            raise ValueError("at least one module must be enabled")
        for module in self.modules:
            if module.enabled and module.type == "detection" and module.model_path is None:
                raise ValueError(f"enabled detection module {module.id} requires model_path")
        return self


def _expand_string(value: str, allow_missing: bool = False) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("plain")
        if name not in os.environ:
            if allow_missing:
                return match.group(0)
            raise ConfigError(f"environment variable is not set: {name}")
        return os.environ[name]

    return ENV_PATTERN.sub(replace, value)


def _expand_environment(value: Any, allow_missing: bool = False) -> Any:
    if isinstance(value, str):
        return _expand_string(value, allow_missing=allow_missing)
    if isinstance(value, list):
        return [_expand_environment(item, allow_missing=allow_missing) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item, allow_missing=allow_missing) for key, item in value.items()}
    return value


def _expand_config(raw: dict[str, Any]) -> dict[str, Any]:
    expanded = {
        key: _expand_environment(value)
        for key, value in raw.items()
        if key != "modules"
    }
    modules = raw.get("modules", [])
    if not isinstance(modules, list):
        expanded["modules"] = modules
        return expanded
    expanded["modules"] = [
        _expand_environment(module, allow_missing=isinstance(module, dict) and not module.get("enabled", True))
        for module in modules
    ]
    return expanded


def load_runtime_config(config_path: Path, validate_model_paths: bool = True) -> RuntimeConfig:
    path = config_path.expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"runtime config does not exist: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"runtime config must contain a YAML mapping: {path}")
    expanded = _expand_config(raw)
    try:
        config = RuntimeConfig.model_validate(expanded)
    except ValidationError as exc:
        raise ConfigError(f"invalid runtime config {path}:\n{exc}") from exc

    for module in config.modules:
        if module.model_path is not None and not module.model_path.is_absolute():
            module.model_path = (path.parent / module.model_path).resolve()
        if validate_model_paths and module.enabled and module.model_path is not None:
            if not module.model_path.is_file():
                raise ConfigError(f"model file for module {module.id} does not exist: {module.model_path}")
    return config
