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

from ..config import ReviewProviderSettings
from ..schemas import DetectionSummary, VLMReviewResult, ValidatedImage
from .base import ReviewProvider
from .parser import ReviewResponseError, parse_review_response
from .prompt import build_review_prompt


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
    ) -> None:
        if not settings.enabled or settings.endpoint is None or settings.model_id is None:
            raise ValueError("Qwen2.5-VL provider requires enabled settings, endpoint, and model_id")
        self.settings = settings
        self.class_catalog = class_catalog
        self.visual_class_guide = visual_class_guide or {}
        self.provider_name = "qwen2_5_vl"
        self.model_id = settings.model_id

    def _image_data_url(self, image: ValidatedImage) -> str:
        buffer = BytesIO()
        image.image.save(buffer, format="JPEG", quality=92)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _request_body(self, image: ValidatedImage, summary: DetectionSummary) -> bytes:
        payload = {
            "model": self.settings.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self._image_data_url(image)}},
                        {
                            "type": "text",
                            "text": build_review_prompt(
                                summary,
                                self.class_catalog,
                                visual_class_guide=self.visual_class_guide,
                            ),
                        },
                    ],
                }
            ],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
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

    def review(self, image: ValidatedImage, summary: DetectionSummary) -> VLMReviewResult:
        started = time.perf_counter()
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key_env:
            api_key = os.environ.get(self.settings.api_key_env)
            if not api_key:
                raise RuntimeError(f"review API key environment variable is not set: {self.settings.api_key_env}")
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            self.settings.endpoint,
            data=self._request_body(image, summary),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Qwen2.5-VL review request failed: {exc}") from exc
        content = self._response_content(payload)
        try:
            decisions, findings, behaviors = parse_review_response(content, summary, self.class_catalog)
        except ReviewResponseError as exc:
            excerpt = _raw_response_excerpt(content)
            raise ReviewResponseError(
                f"{exc}; raw_response_excerpt={excerpt!r}"
            ) from exc
        return VLMReviewResult(
            provider="qwen2_5_vl",
            model_id=self.settings.model_id,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            decisions=decisions,
            findings=findings,
            behaviors=behaviors,
        )
