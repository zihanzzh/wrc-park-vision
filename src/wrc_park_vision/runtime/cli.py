"""Command-line entry point for single-image runtime execution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_runtime_config
from .output import write_runtime_outputs
from .pipeline import RuntimePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the WRC multi-module vision pipeline on one image.")
    parser.add_argument("--config", type=Path, required=True, help="Runtime YAML configuration.")
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument("--no-preview", action="store_true", help="Do not generate preview.jpg.")
    parser.add_argument("--output-dir", type=Path, help="Override runtime.output_dir.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_runtime_config(args.config)
        output_dir = args.output_dir or config.runtime.output_dir
        with RuntimePipeline(config) as pipeline:
            response = pipeline.process(args.image)
        artifacts = write_runtime_outputs(
            response=response,
            output_dir=output_dir,
            preview_settings=config.preview,
            preview_enabled=not args.no_preview,
        )
    except (ConfigError, FileNotFoundError, ValueError, RuntimeError, NotImplementedError) as exc:
        print(f"runtime error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"unexpected runtime error: {exc}", file=sys.stderr)
        return 1

    succeeded = sum(module.status == "success" for module in response.modules)
    failed = sum(module.status == "failure" for module in response.modules)
    print(f"request_id: {response.request_id}")
    print(f"status: {response.status}")
    print(f"modules: {succeeded} succeeded, {failed} failed")
    print(f"observations: {len(response.observations)}")
    print(f"result_json: {artifacts.json_path}")
    print(f"preview: {artifacts.preview_path if artifacts.preview_path else 'not generated'}")
    print(f"total_ms: {response.timing_ms.total:.2f}")

    if response.status == "success":
        return 0
    if response.status == "partial_success":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

