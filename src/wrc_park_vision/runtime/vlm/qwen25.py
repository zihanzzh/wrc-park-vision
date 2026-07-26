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
    VLMRequestMetrics,
    VLMReviewResult,
    ValidatedImage,
)
from .base import ReviewProvider
from .parser import ReviewResponseError, parse_review_response
from .prompt import build_multi_image_prompt


RAW_RESPONSE_EXCERPT_LIMIT = 512


def _raw_response_excerpt(content: str) -> str:
    compact = " ".join(content.split())
    if len(compact) <= RAW_RESPONSE_EXCERPT_LIMIT:
        return compact
    return compact[:RAW_RESPONSE_EXCERPT_LIMIT] + "..."


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
            )
        except ReviewResponseError as exc:
            request_metrics.response_parse_ms = (
                time.perf_counter() - parse_started
            ) * 1000.0
            excerpt = _raw_response_excerpt(content)
            finish_reason = request_metrics.finish_reason or "unknown"
            error = ReviewResponseError(
                f"{exc}; finish_reason={finish_reason!r}; "
                f"raw_response_excerpt={excerpt!r}"
            )
            error.metrics = request_metrics  # type: ignore[attr-defined]
            raise error from exc
        request_metrics.response_parse_ms = (
            time.perf_counter() - parse_started
        ) * 1000.0
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
        return self._request(
            content_parts=content_parts,
            summary=request.summary,
            review_pass="multi_image",
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            valid_crop_ids={crop.crop_id for crop in request.crops},
            required_review_observation_ids=required_review_observation_ids,
            metrics=metrics,
        )
