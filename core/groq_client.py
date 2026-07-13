"""
ARAG_PHARMA — Groq LLM Client
"""
import asyncio
import json
import re
from typing import Optional
from openai import AsyncOpenAI, RateLimitError, APIError
from loguru import logger
from config.settings import settings


def _make_groq_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.GROQ_API_KEY or "placeholder",
        base_url=settings.GROQ_BASE_URL,
        timeout=settings.GROQ_REQUEST_TIMEOUT,
        max_retries=0,
    )


class GroqClient:
    FALLBACK_CHAIN = [
        "llama-3.3-70b-versatile",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "llama-3.1-8b-instant",
    ]

    def __init__(self):
        self._client = _make_groq_client()
        self._total_tokens = 0
        self._total_requests = 0

    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        expect_json: bool = False,
    ) -> str:
        """
        FIX: Rewrote retry/fallback logic.
        Old code: `attempt = 1` inside loop caused infinite retry on rate limit.
        New code: Build a flat list of (model, attempt) pairs. Always terminates.
        """
        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to config/.env "
                "(free key at https://console.groq.com)"
            )

        start_model = model or settings.LLM_MODEL
        msgs = self._inject_json_instruction(messages) if expect_json else messages

        # Phase 1: retry start_model with backoff
        # Phase 2: try each fallback model once
        models_to_try: list[tuple[str, int]] = [
            (start_model, attempt)
            for attempt in range(1, settings.GROQ_MAX_RETRIES + 1)
        ] + [
            (fb, 1)
            for fb in self.FALLBACK_CHAIN
            if fb != start_model
        ]

        last_error: Optional[Exception] = None

        for current_model, attempt in models_to_try:
            try:
                resp = await self._client.chat.completions.create(
                    model=current_model,
                    messages=msgs,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content or ""
                if resp.usage:
                    self._total_tokens += resp.usage.total_tokens
                self._total_requests += 1
                if current_model != start_model:
                    logger.info(f"Used fallback model: {current_model}")
                return content.strip()

            except RateLimitError as e:
                last_error = e
                wait = settings.GROQ_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"Rate limit on {current_model} attempt {attempt}. Waiting {wait:.1f}s..."
                )
                await asyncio.sleep(wait)

            except APIError as e:
                last_error = e
                logger.error(f"Groq API error on {current_model}: {e}")
                await asyncio.sleep(settings.GROQ_RETRY_DELAY)
                break  # Skip remaining attempts for this model on hard API error

            except Exception as e:
                last_error = e
                logger.error(f"Groq unexpected error on {current_model}: {e}")
                break

        raise RuntimeError(f"Groq request failed after all retries: {last_error}")

    async def chat_json(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 800,
        fallback: dict = None,
    ) -> dict:
        try:
            raw = await self.chat(
                messages=messages, model=model,
                temperature=temperature, max_tokens=max_tokens,
                expect_json=True,
            )
            return self._parse_json(raw)
        except Exception as e:
            logger.warning(f"Groq JSON parse/call failed: {e}")
            return fallback or {}

    def _parse_json(self, text: str) -> dict:
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return {"items": json.loads(match.group())}
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not extract JSON from: {text[:200]}")

    def _inject_json_instruction(self, messages: list[dict]) -> list[dict]:
        json_instruction = (
            "IMPORTANT: You MUST respond with ONLY a valid JSON object. "
            "No markdown, no code fences, no explanation. "
            "Start your response directly with { and end with }."
        )
        msgs = list(messages)
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = {
                "role": "system",
                "content": msgs[0]["content"] + "\n\n" + json_instruction,
            }
        else:
            msgs.insert(0, {"role": "system", "content": json_instruction})
        return msgs

    @property
    def usage_stats(self) -> dict:
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
        }


_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            _embedding_model = SentenceTransformer(
                settings.EMBEDDING_MODEL, device=settings.EMBEDDING_DEVICE,
            )
            logger.info("Embedding model loaded successfully")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
            _embedding_model = None
    return _embedding_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    if model is None:
        return [[0.0] * 384 for _ in texts]
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


_groq_client: Optional[GroqClient] = None

def get_groq_client() -> GroqClient:
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqClient()
    return _groq_client
