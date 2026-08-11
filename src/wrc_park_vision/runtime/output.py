"""JSON and preview artifact writing."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import PreviewSettings
from .competition import CompetitionResponse, build_competition_response
from .preview import render_preview
from .schemas import PipelineResponse, RuntimeErrorInfo


@dataclass(frozen=True)
class OutputArtifacts:
    directory: Path
    json_path: Optional[Path]
    competition_json_path: Optional[Path]
    preview_path: Optional[Path]
    vlm_raw_response_path: Optional[Path]


def _serialize_json(response: object) -> str:
    if not hasattr(response, "model_dump"):
        raise TypeError("JSON output requires a Pydantic model")
    return (
        json.dumps(
            response.model_dump(mode="json"),  # type: ignore[attr-defined]
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def _write_serialized_json(content: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(output_path)


def write_json(response: object, output_path: Path) -> None:
    _write_serialized_json(_serialize_json(response), output_path)


def write_runtime_outputs(
    response: PipelineResponse,
    output_dir: Path,
    preview_settings: PreviewSettings,
    preview_enabled: bool,
    output_mode: str = "both",
) -> OutputArtifacts:
    if output_mode not in {"both", "internal", "competition"}:
        raise ValueError(f"unsupported output mode: {output_mode}")
    request_dir = output_dir.expanduser().resolve() / response.request_id
    json_path: Optional[Path] = (
        request_dir / "result.json"
        if output_mode in {"both", "internal"}
        else None
    )
    competition_json_path: Optional[Path] = (
        request_dir / "competition_result.json"
        if output_mode in {"both", "competition"}
        else None
    )
    preview_path: Optional[Path] = None
    vlm_raw_response_path: Optional[Path] = None

    if response.review.raw_response_debug is not None:
        vlm_raw_response_path = request_dir / "vlm_raw_response.txt"
        _write_serialized_json(
            response.review.raw_response_debug,
            vlm_raw_response_path,
        )

    competition_response: CompetitionResponse | None = None
    if competition_json_path is not None:
        adapter_started = time.perf_counter()
        competition_response = build_competition_response(response)
        adapter_duration = (time.perf_counter() - adapter_started) * 1000.0
        response.timing_ms.competition_response_adapter = adapter_duration
        response.timing_ms.total += adapter_duration
        competition_response.processing_time_ms = response.timing_ms.total

    serialization_started = time.perf_counter()
    if competition_response is not None:
        _serialize_json(competition_response)
    if json_path is not None:
        _serialize_json(response)
    serialization_duration = (
        time.perf_counter() - serialization_started
    ) * 1000.0
    response.timing_ms.output_serialization = serialization_duration
    response.timing_ms.total += serialization_duration
    if response.timing_ms.deadline_remaining_ms is not None:
        response.timing_ms.deadline_remaining_ms -= (
            (response.timing_ms.competition_response_adapter or 0.0)
            + serialization_duration
        )
        if response.timing_ms.deadline_remaining_ms < 0:
            if not any(
                error.code == "total_deadline_exceeded"
                for error in response.errors
            ):
                response.errors.append(
                    RuntimeErrorInfo(
                        stage="deadline",
                        code="total_deadline_exceeded",
                        message="runtime deadline exceeded while building SDK output",
                    )
                )
            if response.status == "success":
                response.status = "partial_success"
            response.timing_ms.degraded_reason = "total_deadline_exceeded"
    if competition_response is not None:
        competition_response = build_competition_response(response)
        competition_response.processing_time_ms = response.timing_ms.total
        _write_serialized_json(
            _serialize_json(competition_response),
            competition_json_path,  # type: ignore[arg-type]
        )
    if json_path is not None:
        _write_serialized_json(_serialize_json(response), json_path)

    can_render = response.input.width > 0 and response.input.height > 0
    if (
        output_mode != "competition"
        and preview_enabled
        and preview_settings.enabled
        and can_render
    ):
        candidate = request_dir / "preview.jpg"
        preview_started = time.perf_counter()
        try:
            render_preview(Path(response.input.image_path), response, candidate, preview_settings)
            preview_path = candidate
        except Exception as exc:
            response.errors.append(
                RuntimeErrorInfo(
                    stage="output",
                    code="preview_failure",
                    message=str(exc) or exc.__class__.__name__,
                )
            )
        finally:
            preview_duration = (time.perf_counter() - preview_started) * 1000.0
            response.timing_ms.preview = preview_duration
            response.timing_ms.total += preview_duration
            # Persist output-stage timing and any preview failure without losing inference results.
            if json_path is not None:
                write_json(response, json_path)

    return OutputArtifacts(
        directory=request_dir,
        json_path=json_path,
        competition_json_path=competition_json_path,
        preview_path=preview_path,
        vlm_raw_response_path=vlm_raw_response_path,
    )
