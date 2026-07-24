"""JSON and preview artifact writing."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import PreviewSettings
from .preview import render_preview
from .schemas import PipelineResponse, RuntimeErrorInfo


@dataclass(frozen=True)
class OutputArtifacts:
    directory: Path
    json_path: Path
    preview_path: Optional[Path]


def write_json(response: PipelineResponse, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def write_runtime_outputs(
    response: PipelineResponse,
    output_dir: Path,
    preview_settings: PreviewSettings,
    preview_enabled: bool,
) -> OutputArtifacts:
    request_dir = output_dir.expanduser().resolve() / response.request_id
    json_path = request_dir / "result.json"
    preview_path: Optional[Path] = None

    # JSON is durable before preview generation starts.
    write_json(response, json_path)
    can_render = response.input.width > 0 and response.input.height > 0
    if preview_enabled and preview_settings.enabled and can_render:
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
            write_json(response, json_path)

    return OutputArtifacts(directory=request_dir, json_path=json_path, preview_path=preview_path)
