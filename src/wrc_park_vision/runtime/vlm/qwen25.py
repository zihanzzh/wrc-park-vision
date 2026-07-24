"""OpenAI-compatible Qwen2.5-VL full-image review provider."""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from io import BytesIO
from typing import Any

from ..config import CropScanSettings, ReviewProviderSettings, ReviewSettings
from ..crops import ImageCrop
from ..schemas import DetectionSummary, ReviewSummary, VLMReviewResult, ValidatedImage
from .base import ReviewProvider
from .parser import ReviewResponseError, parse_review_response
from .prompt import build_crop_scan_prompt, build_full_image_prompt


RAW_RESPONSE_EXCERPT_LIMIT = 512


def _raw_response_excerpt(content: str) -> str:
    compact = " ".join(content.split())
    if len(compact) <= RAW_RESPONSE_EXCERPT_LIMIT:
        return compact
    return compact[:RAW_RESPONSE_EXCERPT_LIMIT] + "..."


class Qwen25VLProvider(ReviewProvider):
    """Call a configured Qwen2.5-VL endpoint with the complete decoded image."""

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

    def _image_data_url(self, image: Any) -> str:
        buffer = BytesIO()
        source = image.image if isinstance(image, ValidatedImage) else image
        source.save(buffer, format="JPEG", quality=92)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

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
        }
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
    ) -> VLMReviewResult:
        started = time.perf_counter()
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key_env:
            api_key = os.environ.get(self.settings.api_key_env)
            if not api_key:
                raise RuntimeError(f"review API key environment variable is not set: {self.settings.api_key_env}")
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            self.settings.endpoint,
            data=self._request_body(content_parts, max_tokens),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Qwen2.5-VL review request failed: {exc}") from exc
        content = self._response_content(payload)
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
            )
        except ReviewResponseError as exc:
            excerpt = _raw_response_excerpt(content)
            raise ReviewResponseError(
                f"{exc}; raw_response_excerpt={excerpt!r}"
            ) from exc
        return VLMReviewResult(
            provider="qwen2_5_vl",
            model_id=self.settings.model_id,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            review_pass=review_pass,  # type: ignore[arg-type]
            decisions=parsed.decisions,
            findings=parsed.findings,
            behaviors=parsed.behaviors,
            issues=parsed.issues,
        )

    def review(self, image: ValidatedImage, summary: DetectionSummary) -> VLMReviewResult:
        pass_settings = self.review_settings.full_image if self.review_settings else None
        timeout_seconds = (
            pass_settings.timeout_seconds
            if pass_settings is not None and pass_settings.timeout_seconds is not None
            else self.settings.timeout_seconds
        )
        max_tokens = (
            pass_settings.max_tokens
            if pass_settings is not None and pass_settings.max_tokens is not None
            else self.settings.max_tokens
        )
        return self._request(
            content_parts=[
                {"type": "image_url", "image_url": {"url": self._image_data_url(image)}},
                {
                    "type": "text",
                    "text": build_full_image_prompt(
                        summary,
                        self.class_catalog,
                        visual_class_guide=self.visual_class_guide,
                    ),
                },
            ],
            summary=summary,
            review_pass="full_image",
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
        )

    def review_crops(
        self,
        crops: list[ImageCrop],
        image: ValidatedImage,
        summary: DetectionSummary,
        full_image_review: ReviewSummary,
    ) -> VLMReviewResult:
        settings = (
            self.review_settings.crop_scan
            if self.review_settings is not None
            else CropScanSettings()
        )
        timeout_seconds = settings.timeout_seconds or self.settings.timeout_seconds
        max_tokens = settings.max_tokens or self.settings.max_tokens
        content_parts: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": build_crop_scan_prompt(
                    summary,
                    self.class_catalog,
                    crops,
                    full_image_review,
                    visual_class_guide=self.visual_class_guide,
                ),
            }
        ]
        for crop in crops:
            content_parts.extend(
                [
                    {"type": "text", "text": f"crop_id={crop.crop_id}"},
                    {"type": "image_url", "image_url": {"url": self._image_data_url(crop.image)}},
                ]
            )
        return self._request(
            content_parts=content_parts,
            summary=summary,
            review_pass="crop_scan",
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            valid_crop_ids={crop.crop_id for crop in crops},
        )
    supports_crop_scan = True
