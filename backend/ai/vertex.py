from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import google.auth
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account

from .gemini import GeminiAIProvider, logger as gemini_logger, MAX_OUTPUT_TOKENS

logger = logging.getLogger(__name__)
VERTEX_REQUEST_COUNTER = 0


class VertexAIProvider(GeminiAIProvider):
    """
    Google Vertex AI backed provider that reuses the Gemini prompt/response logic.

    Required environment variables:
      - GOOGLE_CLOUD_PROJECT
      - GOOGLE_CLOUD_LOCATION (defaults to us-central1)

    Authentication:
      - Prefer `GOOGLE_APPLICATION_CREDENTIALS` pointing to a service-account JSON file.
      - Alternatively provide the JSON via `GOOGLE_APPLICATION_CREDENTIALS_JSON`.
      - Falls back to `google.auth.default()` if running on GCP with ambient credentials.

    Optional environment variables:
      - COGNIVERSE_VERTEX_MODEL (defaults to gemini-1.5-flash-001)
      - COGNIVERSE_VERTEX_MAX_OUTPUT_TOKENS (limits generation length)
    """

    def __init__(
        self,
        *,
        project: Optional[str] = None,
        location: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 45.0,
    ):
        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for Vertex provider.")
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        self.model = model or os.getenv(
            "COGNIVERSE_VERTEX_MODEL", "gemini-1.5-flash-001"
        )
        self.timeout = timeout
        self.api_key = None  # Attribute used by Gemini parent; not required here.

        self._max_tokens = int(
            os.getenv(
                "COGNIVERSE_VERTEX_MAX_OUTPUT_TOKENS",
                os.getenv("COGNIVERSE_GEMINI_MAX_OUTPUT_TOKENS", str(MAX_OUTPUT_TOKENS)),
            )
        )

        credentials = self._load_credentials()
        self.session = AuthorizedSession(credentials)
        self.base_url = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/"
            f"{self.project}/locations/{self.location}/publishers/google/models/"
            f"{self.model}:generateContent"
        )

    # ------------------------------------------------------------------ #
    # Credential management                                              #
    # ------------------------------------------------------------------ #
    def _load_credentials(self):
        scopes = ("https://www.googleapis.com/auth/cloud-platform",)
        json_blob = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if json_blob:
            try:
                info = json.loads(json_blob)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "GOOGLE_APPLICATION_CREDENTIALS_JSON is not valid JSON."
                ) from exc
            credentials = service_account.Credentials.from_service_account_info(
                info, scopes=scopes
            )
            return credentials

        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_path:
            path = Path(credentials_path)
            if not path.exists():
                raise RuntimeError(
                    f"GOOGLE_APPLICATION_CREDENTIALS file not found at {credentials_path!r}"
                )
            credentials = service_account.Credentials.from_service_account_file(
                path, scopes=scopes
            )
            return credentials

        credentials, _ = google.auth.default(scopes=scopes)
        if hasattr(credentials, "with_scopes_if_required"):
            credentials = credentials.with_scopes_if_required(scopes)
        elif hasattr(credentials, "with_scopes"):
            credentials = credentials.with_scopes(scopes)
        return credentials

    # ------------------------------------------------------------------ #
    # Gemini overrides                                                   #
    # ------------------------------------------------------------------ #
    def _invoke(
        self,
        prompt: str,
        *,
        max_output_tokens: int | None = None,
        attempt: int = 1,
    ) -> str:
        global VERTEX_REQUEST_COUNTER
        VERTEX_REQUEST_COUNTER += 1
        logger.info("Vertex requests so far: %s", VERTEX_REQUEST_COUNTER)

        max_tokens = min(max_output_tokens or self._max_tokens, self._max_tokens)
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.75,
                "topP": 0.9,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }

        response = self.session.post(
            self.base_url,
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            logger.error(
                "Vertex API error %s: %s", response.status_code, response.text
            )
            response.raise_for_status()

        data = response.json()
        candidates = data.get("candidates") or []
        for candidate in candidates:
            finish_reason = candidate.get("finishReason")
            content = candidate.get("content") or {}
            parts = content.get("parts") or []
            for part in parts:
                text = part.get("text")
                if isinstance(text, str):
                    return text.strip()
            if (
                finish_reason == "MAX_TOKENS"
                and max_tokens < self._max_tokens
                and attempt < 3
            ):
                logger.warning(
                    "Vertex output hit MAX_TOKENS at %s tokens on attempt %s; retrying with %s tokens.",
                    max_tokens,
                    attempt,
                    min(max_tokens + 256, self._max_tokens),
                )
                return self._invoke(
                    prompt,
                    max_output_tokens=min(max_tokens + 256, self._max_tokens),
                    attempt=attempt + 1,
                )

        raise ValueError(f"Vertex response missing text payload: {data!r}")


# Keep logger name consistent with Gemini provider for shared logging format.
gemini_logger.addHandler(logging.NullHandler())
