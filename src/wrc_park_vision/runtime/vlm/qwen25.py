"""OpenAI-compatible Qwen2.5-VL Runtime V3 multi-image provider."""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from io import BytesIO
from typing import Any

from PIL import Image

from ..config import ReviewProviderSettings, ReviewSettings
from ..review.multi_image_request import MultiImageReviewRequest
from ..schemas import (
    DetectionSummary,
    ReviewIssue,
    VLMRequestMetrics,
    VLMReviewResult,
    ValidatedImage,
)
from .base import ReviewProvider
from .parser import ReviewResponseError, parse_review_response
from .prompt import build_multi_image_prompt


RAW_RESPONSE_EXCERPT_LIMIT = 512


class VLMResponseTruncated(ReviewResponseError):
    """Raised when the endpoint visibly truncates the JSON response."""


def _raw_response_excerpt(content: str) -> str:
    compact = " ".join(content.split())
    if len(compact) <= RAW_RESPONSE_EXCERPT_LIMIT:
        return compact
    return compact[:RAW_RESPONSE_EXCERPT_LIMIT] + "..."


def _looks_like_truncated_json(content: str) -> bool:
    stripped = content.strip()
    if stripped.startswith("```") and not stripped.endswith("```"):
        return True
    if stripped.startswith("```json"):
        stripped = stripped[7:].lstrip()
    elif stripped.startswith("```"):
        stripped = stripped[3:].lstrip()
    return stripped.startswith("{") and not stripped.rstrip().endswith("}")


class Qwen25VLProvider(ReviewProvider):
    """Send the original image and focused crops in one VLM request."""

    def __init__(
        self,
        settings: ReviewProviderSettings,
        class_catalog: dict[str, list[str]],
        visual_class_guide: dict[str, dict[str, dict[str, object]]] | None = None,
        review_settings: ReviewSettings | None = None,
    ) -> None:
        if not settings.enabled or settings.endpoint is None or settings.model_id is None:
            raise ValueError("Qwen2.5-VL provider requires enabled settings, endpoint, and model_id")
        self.settings = settings
        self.class_catalog = class_catalog
        self.visual_class_guide = visual_class_guide or {}
        self.review_settings = review_settings
        self.provider_name = "qwen2_5_vl"
        self.model_id = settings.model_id

    def _image_data_url(self, image: Any) -> tuple[str, int]:
        buffer = BytesIO()
        source = image.image if isinstance(image, ValidatedImage) else image
        max_side = max(source.size)
        if max_side > self.settings.image_max_side:
            scale = self.settings.image_max_side / max_side
            resized = source.resize(
                (
                    max(1, round(source.width * scale)),
                    max(1, round(source.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
        else:
            resized = source
        resized.save(
            buffer,
            format="JPEG",
            quality=self.settings.jpeg_quality,
            optimize=True,
        )
        jpeg_bytes = buffer.getvalue()
        encoded = base64.b64encode(jpeg_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}", len(jpeg_bytes)

    def _request_body(
        self,
        content: list[dict[str, Any]],
        max_tokens: int,
    ) -> bytes:
        payload = {
            "model": self.settings.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "temperature": self.settings.temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.settings.response_format_json_object:
            payload["response_format"] = {"type": "json_object"}
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def _response_content(self, payload: dict[str, Any]) -> str:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ReviewResponseError("VLM endpoint response is missing choices[0].message.content") from exc
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
            if text_parts:
                return "".join(text_parts)
        raise ReviewResponseError("VLM endpoint returned unsupported message content")

    def _request(
        self,
        *,
        content_parts: list[dict[str, Any]],
        summary: DetectionSummary,
        review_pass: str,
        timeout_seconds: float,
        max_tokens: int,
        valid_crop_ids: set[str] | None = None,
        required_review_observation_ids: tuple[str, ...] | None = None,
        metrics: VLMRequestMetrics | None = None,
        allow_new_findings: bool = True,
    ) -> VLMReviewResult:
        started = time.perf_counter()
        request_metrics = metrics or VLMRequestMetrics()
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key_env:
            api_key = os.environ.get(self.settings.api_key_env)
            if not api_key:
                raise RuntimeError(f"review API key environment variable is not set: {self.settings.api_key_env}")
            headers["Authorization"] = f"Bearer {api_key}"
        serialization_started = time.perf_counter()
        request_body = self._request_body(content_parts, max_tokens)
        request_metrics.request_json_serialize_ms = (
            time.perf_counter() - serialization_started
        ) * 1000.0
        request_metrics.request_payload_bytes = len(request_body)
        request = urllib.request.Request(
            self.settings.endpoint,
            data=request_body,
            headers=headers,
            method="POST",
        )
        http_started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                response_bytes = response.read()
            request_metrics.vlm_http_round_trip_ms = (
                time.perf_counter() - http_started
            ) * 1000.0
            payload = json.loads(response_bytes.decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            request_metrics.vlm_http_round_trip_ms = (
                time.perf_counter() - http_started
            ) * 1000.0
            error = RuntimeError(f"Qwen2.5-VL review request failed: {exc}")
            error.metrics = request_metrics  # type: ignore[attr-defined]
            raise error from exc
        choices = payload.get("choices")
        first_choice = choices[0] if isinstance(choices, list) and choices else {}
        if isinstance(first_choice, dict):
            finish_reason = first_choice.get("finish_reason")
            request_metrics.finish_reason = (
                str(finish_reason) if finish_reason is not None else None
            )
        usage = payload.get("usage")
        if isinstance(usage, dict):
            for source_name, target_name in (
                ("prompt_tokens", "prompt_tokens"),
                ("completion_tokens", "completion_tokens"),
                ("total_tokens", "total_tokens"),
            ):
                value = usage.get(source_name)
                if isinstance(value, int) and value >= 0:
                    setattr(request_metrics, target_name, value)
        content = self._response_content(payload)
        request_metrics.response_content_length = len(content)
        if request_metrics.finish_reason == "length":
            error = VLMResponseTruncated(
                "VLM response was truncated by the token limit; "
                "finish_reason='length'"
            )
            error.metrics = request_metrics  # type: ignore[attr-defined]
            error.raw_response_debug = content  # type: ignore[attr-defined]
            raise error
        parse_started = time.perf_counter()
        try:
            parsed = parse_review_response(
                content,
                summary,
                self.class_catalog,
                review_pass=review_pass,  # type: ignore[arg-type]
                require_finding_bbox=(
                    self.review_settings.require_finding_bbox
                    if self.review_settings is not None
                    else False
                ),
                valid_crop_ids=valid_crop_ids,
                required_review_observation_ids=required_review_observation_ids,
                max_new_findings=(
                    self.review_settings.max_new_findings
                    if self.review_settings is not None
                    else 8
                ),
                new_finding_existing_iou_threshold=(
                    self.review_settings.new_finding_existing_iou_threshold
                    if self.review_settings is not None
                    else 0.80
                ),
                allow_new_findings=allow_new_findings,
            )
        except ReviewResponseError as exc:
            request_metrics.response_parse_ms = (
                time.perf_counter() - parse_started
            ) * 1000.0
            excerpt = _raw_response_excerpt(content)
            finish_reason = request_metrics.finish_reason or "unknown"
            error_type = (
                VLMResponseTruncated
                if _looks_like_truncated_json(content)
                else ReviewResponseError
            )
            error = error_type(
                f"{exc}; finish_reason={finish_reason!r}; "
                f"raw_response_excerpt={excerpt!r}"
            )
            error.metrics = request_metrics  # type: ignore[attr-defined]
            error.raw_response_debug = content  # type: ignore[attr-defined]
            raise error from exc
        request_metrics.response_parse_ms = (
            time.perf_counter() - parse_started
        ) * 1000.0
        if not allow_new_findings and any(
            issue.code
            in {
                "missing_observation_review",
                "missing_candidate_decision",
                "missing_section",
            }
            for issue in parsed.issues
        ):
            error = ReviewResponseError(
                "compact fallback did not cover every required object and "
                "behavior candidate"
            )
            error.metrics = request_metrics  # type: ignore[attr-defined]
            error.raw_response_debug = content  # type: ignore[attr-defined]
            raise error
        return VLMReviewResult(
            provider="qwen2_5_vl",
            model_id=self.settings.model_id,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            review_pass=review_pass,  # type: ignore[arg-type]
            decisions=parsed.decisions,
            findings=parsed.findings,
            behaviors=parsed.behaviors,
            issues=parsed.issues,
            metrics=request_metrics,
        )

    def review(self, image: ValidatedImage, summary: DetectionSummary) -> VLMReviewResult:
        request = MultiImageReviewRequest(image=image, summary=summary)
        return self._review_multi_image(
            request,
            required_review_observation_ids=tuple(
                item.observation_id for item in summary.detections
            ),
        )

    def review_multi_image(
        self,
        request: MultiImageReviewRequest,
    ) -> VLMReviewResult:
        return self._review_multi_image(
            request,
            required_review_observation_ids=request.required_review_observation_ids,
        )

    def _review_multi_image(
        self,
        request: MultiImageReviewRequest,
        *,
        required_review_observation_ids: tuple[str, ...],
    ) -> VLMReviewResult:
        pass_settings = (
            self.review_settings.multi_image
            if self.review_settings is not None
            else None
        )
        configured_pass_timeout = (
            pass_settings.timeout_seconds
            if pass_settings is not None and pass_settings.timeout_seconds is not None
            else self.settings.timeout_seconds
        )
        timeout_candidates = [
            configured_pass_timeout,
            self.settings.timeout_seconds,
        ]
        if request.timeout_seconds is not None:
            timeout_candidates.append(request.timeout_seconds)
        timeout_seconds = min(timeout_candidates)
        max_tokens = (
            pass_settings.max_tokens
            if pass_settings is not None and pass_settings.max_tokens is not None
            else self.settings.max_tokens
        )
        metrics = VLMRequestMetrics(
            image_count=1 + len(request.crops),
            required_review_count=len(required_review_observation_ids),
            crop_count=len(request.crops),
        )
        prompt_started = time.perf_counter()
        prompt = build_multi_image_prompt(
            request,
            self.class_catalog,
            visual_class_guide=self.visual_class_guide,
            required_review_observation_ids=required_review_observation_ids,
            max_new_findings=(
                self.review_settings.max_new_findings
                if self.review_settings is not None
                else 8
            ),
        )
        metrics.prompt_build_ms = (time.perf_counter() - prompt_started) * 1000.0
        metrics.prompt_character_count = len(prompt)

        original_encode_started = time.perf_counter()
        original_data_url, original_bytes = self._image_data_url(request.image)
        metrics.original_image_encode_ms = (
            time.perf_counter() - original_encode_started
        ) * 1000.0
        metrics.encoded_image_bytes_total = original_bytes
        content_parts: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": prompt,
            },
            {"type": "text", "text": "image_id=original_image"},
            {
                "type": "image_url",
                "image_url": {"url": original_data_url},
            },
        ]
        crop_encode_started = time.perf_counter()
        for crop in request.crops:
            crop_data_url, crop_bytes = self._image_data_url(crop.image)
            metrics.encoded_image_bytes_total += crop_bytes
            content_parts.extend(
                [
                    {"type": "text", "text": f"crop_id={crop.crop_id}"},
                    {"type": "image_url", "image_url": {"url": crop_data_url}},
                ]
            )
        metrics.crops_encode_ms = (
            time.perf_counter() - crop_encode_started
        ) * 1000.0
        review_started = time.perf_counter()
        try:
            return self._request(
                content_parts=content_parts,
                summary=request.summary,
                review_pass="multi_image",
                timeout_seconds=timeout_seconds,
                max_tokens=max_tokens,
                valid_crop_ids={crop.crop_id for crop in request.crops},
                required_review_observation_ids=required_review_observation_ids,
                metrics=metrics,
                allow_new_findings=True,
            )
        except VLMResponseTruncated as primary_error:
            fallback_settings = (
                self.review_settings.truncation_fallback
                if self.review_settings is not None
                else None
            )
            if fallback_settings is not None and not fallback_settings.enabled:
                raise
            fallback_max_tokens = (
                fallback_settings.max_tokens
                if fallback_settings is not None
                else 1800
            )
            primary_metrics = getattr(primary_error, "metrics", metrics)
            if not isinstance(primary_metrics, VLMRequestMetrics):
                primary_metrics = metrics
            primary_raw = getattr(primary_error, "raw_response_debug", "")

            remaining_timeout = timeout_seconds - (
                time.perf_counter() - review_started
            )
            if remaining_timeout <= 0:
                primary_error.metrics = primary_metrics  # type: ignore[attr-defined]
                raise primary_error

            primary_metrics.request_count = 2
            primary_metrics.fallback_attempted = True
            primary_metrics.fallback_reason = "response_truncated"
            primary_metrics.fallback_max_tokens = fallback_max_tokens

            fallback_prompt_started = time.perf_counter()
            fallback_prompt = build_multi_image_prompt(
                request,
                self.class_catalog,
                visual_class_guide=self.visual_class_guide,
                required_review_observation_ids=required_review_observation_ids,
                include_new_findings=False,
                max_new_findings=0,
            )
            primary_metrics.fallback_prompt_character_count = len(fallback_prompt)
            fallback_content_parts = [
                {"type": "text", "text": fallback_prompt},
                *content_parts[1:],
            ]
            fallback_metrics = VLMRequestMetrics(
                prompt_build_ms=(time.perf_counter() - fallback_prompt_started)
                * 1000.0,
                prompt_character_count=len(fallback_prompt),
                image_count=1 + len(request.crops),
                required_review_count=len(required_review_observation_ids),
                crop_count=len(request.crops),
                encoded_image_bytes_total=metrics.encoded_image_bytes_total,
            )
            try:
                result = self._request(
                    content_parts=fallback_content_parts,
                    summary=request.summary,
                    review_pass="multi_image",
                    timeout_seconds=remaining_timeout,
                    max_tokens=fallback_max_tokens,
                    valid_crop_ids={crop.crop_id for crop in request.crops},
                    required_review_observation_ids=required_review_observation_ids,
                    metrics=fallback_metrics,
                    allow_new_findings=False,
                )
            except Exception as fallback_error:
                observed_fallback_metrics = getattr(
                    fallback_error,
                    "metrics",
                    fallback_metrics,
                )
                if isinstance(observed_fallback_metrics, VLMRequestMetrics):
                    self._copy_fallback_metrics(
                        primary_metrics,
                        observed_fallback_metrics,
                    )
                fallback_raw = getattr(
                    fallback_error,
                    "raw_response_debug",
                    "",
                )
                combined_raw = self._format_raw_debug(primary_raw, fallback_raw)
                fallback_error.metrics = primary_metrics  # type: ignore[attr-defined]
                fallback_error.raw_response_debug = combined_raw  # type: ignore[attr-defined]
                raise fallback_error

            self._copy_fallback_metrics(primary_metrics, fallback_metrics)
            result.duration_ms = (time.perf_counter() - review_started) * 1000.0
            result.findings = []
            result.issues.extend(
                [
                    ReviewIssue(
                        section="response",
                        code="primary_response_truncated",
                        message=(
                            "primary response was truncated; compact fallback "
                            "completed object and behavior review"
                        ),
                        review_pass="multi_image",
                    ),
                    ReviewIssue(
                        section="new_findings",
                        code="missed_object_scan_skipped",
                        message=(
                            "new_findings were intentionally disabled in compact fallback"
                        ),
                        review_pass="multi_image",
                    ),
                ]
            )
            result.metrics = primary_metrics
            result.raw_response_debug = self._format_raw_debug(primary_raw, "")
            return result

    @staticmethod
    def _copy_fallback_metrics(
        primary: VLMRequestMetrics,
        fallback: VLMRequestMetrics,
    ) -> None:
        primary.fallback_vlm_http_round_trip_ms = fallback.vlm_http_round_trip_ms
        primary.fallback_response_content_length = fallback.response_content_length
        primary.fallback_response_parse_ms = fallback.response_parse_ms
        primary.fallback_finish_reason = fallback.finish_reason
        primary.fallback_prompt_tokens = fallback.prompt_tokens
        primary.fallback_completion_tokens = fallback.completion_tokens
        primary.fallback_total_tokens = fallback.total_tokens

    @staticmethod
    def _format_raw_debug(primary: str, fallback: str) -> str:
        parts = ["=== primary_response ===\n", primary]
        if fallback:
            parts.extend(["\n\n=== compact_fallback_response ===\n", fallback])
        return "".join(parts)
