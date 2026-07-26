"""Warm-run Runtime V3 benchmark for NVIDIA Thor.

This tool runs real configured detectors and VLM. Do not use it in unit tests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from wrc_park_vision.runtime.competition import build_competition_response
from wrc_park_vision.runtime.config import load_runtime_config
from wrc_park_vision.runtime.pipeline import RuntimePipeline


def _models_endpoint(chat_endpoint: str) -> str:
    parts = urlsplit(chat_endpoint)
    path = parts.path
    marker = "/v1/chat/completions"
    if path.endswith(marker):
        path = path[: -len(marker)] + "/v1/models"
    else:
        path = "/v1/models"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def check_vlm_readiness(
    endpoint: str,
    model_id: str,
    *,
    api_key: str | None = None,
    timeout_seconds: float = 2.0,
) -> None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(
        _models_endpoint(endpoint),
        headers=headers,
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if response.status >= 400:
                raise RuntimeError(f"VLM readiness returned HTTP {response.status}")
            payload = json.loads(response.read().decode("utf-8"))
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(f"VLM service is not ready: {exc}") from exc
    models = payload.get("data") if isinstance(payload, dict) else None
    available_ids = {
        str(item["id"])
        for item in models or []
        if isinstance(item, dict) and item.get("id") is not None
    }
    if model_id not in available_ids:
        raise RuntimeError(
            f"configured VLM model {model_id!r} is not exposed by /v1/models; "
            f"available={sorted(available_ids)!r}"
        )


def _run_once(pipeline: RuntimePipeline, image_path: Path) -> dict[str, Any]:
    response = pipeline.process(image_path)
    adapter_started = time.perf_counter()
    sdk_response = build_competition_response(response)
    adapter_ms = (time.perf_counter() - adapter_started) * 1000.0
    metrics = response.review.metrics
    image_encode_ms = None
    if metrics is not None:
        image_encode_ms = (
            (metrics.original_image_encode_ms or 0.0)
            + (metrics.crops_encode_ms or 0.0)
        )
    total_ms = response.timing_ms.total + adapter_ms
    return {
        "request_id": response.request_id,
        "status": response.status,
        "review_attempted": response.review.attempted,
        "review_status": response.review.status,
        "vlm_completed": (
            response.review.attempted
            and response.review.status == "completed"
            and metrics is not None
            and metrics.vlm_http_round_trip_ms is not None
        ),
        "degraded": sdk_response.degraded,
        "total_ms": total_ms,
        "detection_ms": response.timing_ms.detection_wall_time,
        "image_encode_ms": image_encode_ms,
        "request_payload_bytes": (
            metrics.request_payload_bytes if metrics else None
        ),
        "image_count": metrics.image_count if metrics else None,
        "prompt_tokens": metrics.prompt_tokens if metrics else None,
        "completion_tokens": metrics.completion_tokens if metrics else None,
        "total_tokens": metrics.total_tokens if metrics else None,
        "vlm_http_round_trip_ms": (
            metrics.vlm_http_round_trip_ms if metrics else None
        ),
        "parse_ms": metrics.response_parse_ms if metrics else None,
        "fusion_ms": response.timing_ms.fusion,
        "competition_adapter_ms": adapter_ms,
        "finish_reason": metrics.finish_reason if metrics else None,
        "prompt_character_count": (
            metrics.prompt_character_count if metrics else None
        ),
        "required_review_count": (
            metrics.required_review_count if metrics else None
        ),
        "crop_count": metrics.crop_count if metrics else None,
        "errors": [error.model_dump(mode="json") for error in response.errors],
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _summarize_numeric_metric(
    runs: list[dict[str, Any]],
    metric: str,
) -> dict[str, float] | None:
    values = [
        float(item[metric])
        for item in runs
        if isinstance(item.get(metric), (int, float))
    ]
    if not values:
        return None
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 0.9),
        "max": max(values),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark warm Runtime V3 requests on NVIDIA Thor."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-readiness-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup_runs < 1:
        raise ValueError("warmup-runs must be at least 1")
    if args.runs < 5:
        raise ValueError("runs must be at least 5")

    config = load_runtime_config(args.config)
    provider = config.review.provider
    if not provider.enabled or provider.endpoint is None:
        raise ValueError("benchmark requires an enabled VLM provider")
    if provider.model_id is None:
        raise ValueError("benchmark requires review.provider.model_id")
    if not args.skip_readiness_check:
        api_key = (
            os.environ.get(provider.api_key_env)
            if provider.api_key_env
            else None
        )
        check_vlm_readiness(
            provider.endpoint,
            provider.model_id,
            api_key=api_key,
        )

    with RuntimePipeline(config) as pipeline:
        warmups = [
            _run_once(pipeline, args.image)
            for _ in range(args.warmup_runs)
        ]
        runs = [_run_once(pipeline, args.image) for _ in range(args.runs)]
        initialization_ms = pipeline.model_initialization_ms

    all_vlm_completed = all(item["vlm_completed"] for item in runs)
    all_non_degraded = all(not item["degraded"] for item in runs)
    summary = {
        metric: _summarize_numeric_metric(runs, metric)
        for metric in (
            "total_ms",
            "detection_ms",
            "image_encode_ms",
            "request_payload_bytes",
            "image_count",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "vlm_http_round_trip_ms",
            "parse_ms",
            "fusion_ms",
            "competition_adapter_ms",
            "prompt_character_count",
            "required_review_count",
            "crop_count",
        )
    }
    total_summary = summary["total_ms"]
    assert total_summary is not None
    passed = (
        all_vlm_completed
        and all_non_degraded
        and total_summary["median"] < 10_000.0
    )
    report = {
        "performance_passed": passed,
        "criteria": {
            "all_vlm_completed": all_vlm_completed,
            "all_non_degraded": all_non_degraded,
            "warm_median_under_10000_ms": (
                total_summary["median"] < 10_000.0
            ),
        },
        "model_initialization_ms": initialization_ms,
        "warmup_runs": warmups,
        "warm_runs": runs,
        "summary": summary,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
