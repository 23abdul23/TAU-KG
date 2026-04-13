import os
from dataclasses import dataclass
from typing import Optional

from src.logger_config import get_logger

logger = get_logger(__name__)

try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


DEFAULT_PROVIDER = "openai"
SUPPORTED_PROVIDERS = {"openai", "gemini", "nvidia_nim", "nvidia", "nim"}
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_NIM_MODEL = "moonshotai/kimi-k2-instruct"
DEFAULT_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
GEMINI_FALLBACK_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-latest",
    "gemini-2.0-flash",
]


@dataclass
class LLMResponse:
    text: str


class LLMClient:
    """Unified client that routes generation requests to OpenAI or Gemini based on env config."""

    def __init__(self) -> None:
        provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            logger.warning(
                "Unsupported LLM_PROVIDER=%r. Falling back to %s.",
                provider,
                DEFAULT_PROVIDER,
            )
            provider = DEFAULT_PROVIDER

        if provider in {"nvidia", "nim"}:
            provider = "nvidia_nim"

        self.provider = provider
        self.client = None
        self.available = False
        self.unavailable_reason = ""

        self.default_openai_model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL
        self.default_gemini_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
        self.default_nim_model = os.getenv("NIM_MODEL", DEFAULT_NIM_MODEL).strip() or DEFAULT_NIM_MODEL
        self.nim_base_url = os.getenv("NIM_BASE_URL", DEFAULT_NIM_BASE_URL).strip() or DEFAULT_NIM_BASE_URL

        self._initialize()

    def _initialize(self) -> None:
        if self.provider == "openai":
            if openai is None:
                self.unavailable_reason = "openai package not installed"
                return

            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                self.unavailable_reason = "OPENAI_API_KEY not set"
                return

            self.client = openai.OpenAI(api_key=api_key)
            self.available = True
            return

        if self.provider == "nvidia_nim":
            if openai is None:
                self.unavailable_reason = "openai package not installed"
                return

            api_key = os.getenv("NIM_API_KEY", "").strip()
            if not api_key:
                self.unavailable_reason = "NIM_API_KEY not set"
                return

            self.client = openai.OpenAI(
                base_url=self.nim_base_url,
                api_key=api_key,
            )
            self.available = True
            return

        if genai is None:
            self.unavailable_reason = "google-generativeai package not installed"
            return

        api_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            self.unavailable_reason = "GEMINI_API_KEY (or GOOGLE_API_KEY) not set"
            return

        genai.configure(api_key=api_key)
        self.available = True

    def is_available(self) -> bool:
        return self.available

    def get_provider_label(self) -> str:
        return self.provider.upper()

    def get_default_model(self) -> str:
        if self.provider == "openai":
            return self.default_openai_model
        if self.provider == "nvidia_nim":
            return self.default_nim_model
        return self.default_gemini_model

    def resolve_model(self, preferred_model: Optional[str]) -> str:
        model = (preferred_model or "").strip()
        if not model:
            return self.get_default_model()

        if self.provider == "gemini" and model.lower().startswith("gpt"):
            logger.warning("Configured model %r looks like an OpenAI model. Using Gemini default model.", model)
            return self.default_gemini_model

        if self.provider == "openai" and "gemini" in model.lower():
            logger.warning("Configured model %r looks like a Gemini model. Using OpenAI default model.", model)
            return self.default_openai_model

        if self.provider == "nvidia_nim" and "gemini" in model.lower():
            logger.warning("Configured model %r looks like a Gemini model. Using NIM default model.", model)
            return self.default_nim_model

        if self.provider == "nvidia_nim" and model.lower().startswith("gpt"):
            logger.warning("Configured model %r may not be valid on NIM. Using NIM default model.", model)
            return self.default_nim_model

        return model

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResponse:
        if not self.available:
            raise RuntimeError(self.unavailable_reason or "LLM client not available")

        resolved_model = self.resolve_model(model)

        if self.provider in {"openai", "nvidia_nim"}:
            kwargs = {
                "model": resolved_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            return LLMResponse(text=content.strip())

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if json_mode:
            generation_config["response_mime_type"] = "application/json"

        candidate_models = [resolved_model]
        for fallback in GEMINI_FALLBACK_MODELS:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        response = None
        last_error = None
        for candidate_model in candidate_models:
            try:
                model_client = genai.GenerativeModel(
                    model_name=candidate_model,
                    system_instruction=system_prompt,
                )
                response = model_client.generate_content(
                    user_prompt,
                    generation_config=generation_config,
                )
                if candidate_model != resolved_model:
                    logger.warning(
                        "Gemini model %r unavailable. Falling back to %r.",
                        resolved_model,
                        candidate_model,
                    )
                break
            except Exception as exc:
                last_error = exc
                if "not found" in str(exc).lower() or "404" in str(exc):
                    continue
                raise

        if response is None:
            raise RuntimeError(f"No working Gemini model found. Last error: {last_error}")

        response_text = getattr(response, "text", None)
        if response_text:
            return LLMResponse(text=response_text.strip())

        candidates = getattr(response, "candidates", None) or []
        if candidates:
            parts = getattr(candidates[0].content, "parts", [])
            merged = "".join(getattr(part, "text", "") for part in parts)
            return LLMResponse(text=merged.strip())

        return LLMResponse(text="")
