"""Real LLM generation. OpenAI-compatible providers with automatic fallback.

Primary is Groq (free tier, no card). If a model/provider fails or rate-limits,
we try the secondary model and then any other provider that has a key in .env.
No mock answers are ever returned.
"""
from __future__ import annotations

from openai import OpenAI

from app.config import settings

PROVIDER_CONFIG = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_attr": "groq_api_key",
        "default_model": "openai/gpt-oss-120b",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_attr": "gemini_api_key",
        "default_model": "gemini-2.5-flash",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_attr": "openrouter_api_key",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
    },
}

SYSTEM_MESSAGE = (
    "You answer questions about a student's AI course notes. "
    "Use only the supplied context. Cite the source file for facts you use. "
    "If the context does not contain the answer, say 'I do not know'. "
    "Be concise and do not add outside knowledge."
)


class GenerationError(RuntimeError):
    pass


class Generator:
    def __init__(self, cfg=settings):
        self.cfg = cfg
        self._clients: dict[str, OpenAI] = {}

    def _candidates(self) -> list[tuple[str, dict, str, str]]:
        primary = self.cfg.llm_provider
        result: list[tuple[str, dict, str, str]] = []

        def add(provider: str, model: str):
            conf = PROVIDER_CONFIG.get(provider)
            if not conf:
                return
            key = getattr(self.cfg, conf["key_attr"], "")
            if key and (provider, model) not in [(p, m) for p, _, _, m in result]:
                result.append((provider, conf, key, model))

        add(primary, self.cfg.llm_model)
        add(primary, self.cfg.llm_fallback_model)
        for provider, conf in PROVIDER_CONFIG.items():
            if provider != primary:
                add(provider, conf["default_model"])
        return result

    def _client(self, provider: str, conf: dict, key: str) -> OpenAI:
        if provider not in self._clients:
            self._clients[provider] = OpenAI(
                base_url=conf["base_url"],
                api_key=key,
                timeout=self.cfg.llm_timeout,
                max_retries=1,
            )
        return self._clients[provider]

    def _kwargs(self, provider: str, model: str, prompt: str, stream: bool) -> dict:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.cfg.llm_temperature,
            "stream": stream,
        }
        # gpt-oss on Groq uses max_completion_tokens and spends part of the
        # budget on reasoning tokens, so keep the budget generous.
        if provider == "groq":
            kwargs["max_completion_tokens"] = self.cfg.llm_max_tokens
        else:
            kwargs["max_tokens"] = self.cfg.llm_max_tokens
        return kwargs

    def generate(self, prompt: str) -> tuple[str, str]:
        candidates = self._candidates()
        if not candidates:
            raise GenerationError(
                "No LLM API key configured. Add GROQ_API_KEY (or another key) to backend/.env"
            )
        errors = []
        for provider, conf, key, model in candidates:
            try:
                client = self._client(provider, conf, key)
                response = client.chat.completions.create(
                    **self._kwargs(provider, model, prompt, stream=False)
                )
                text = (response.choices[0].message.content or "").strip()
                if text:
                    return text, f"{provider}:{model}"
                errors.append(f"{provider}:{model} returned an empty answer")
            except Exception as exc:  # rate limit, timeout, bad model id ...
                errors.append(f"{provider}:{model}: {type(exc).__name__}: {str(exc)[:160]}")
        raise GenerationError("All LLM providers failed -> " + " | ".join(errors))

    def stream(self, prompt: str):
        """Yields ('start', label) then ('token', text). Falls back before first token."""
        candidates = self._candidates()
        if not candidates:
            raise GenerationError(
                "No LLM API key configured. Add GROQ_API_KEY (or another key) to backend/.env"
            )
        errors = []
        for provider, conf, key, model in candidates:
            label = f"{provider}:{model}"
            started = False
            try:
                client = self._client(provider, conf, key)
                chunks = client.chat.completions.create(
                    **self._kwargs(provider, model, prompt, stream=True)
                )
                for chunk in chunks:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        if not started:
                            yield "start", label
                            started = True
                        yield "token", delta
                if started:
                    return
                errors.append(f"{label} returned no text")
            except Exception as exc:
                if started:
                    raise
                errors.append(f"{label}: {type(exc).__name__}: {str(exc)[:160]}")
        raise GenerationError("All LLM providers failed -> " + " | ".join(errors))
