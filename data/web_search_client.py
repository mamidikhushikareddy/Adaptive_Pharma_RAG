"""
ARAG_PHARMA — Web/Internet Search Client
"""
import httpx
import json
import re
from datetime import datetime, timezone
from loguru import logger
from config.settings import settings


class WebSearchClient:
    """
    Internet search client with multiple provider support:
    1. Serper (Google) — if SERPER_API_KEY is set (2500 free/month)
    2. DuckDuckGo HTML — free, no key, more results than Instant Answer API
    3. DuckDuckGo Instant Answer API — fallback
    All results tagged source="WebSearch" with provider metadata.
    """

    DUCKDUCKGO_URL = "https://api.duckduckgo.com/"
    DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
    SERPER_URL = "https://google.serper.dev/search"
    TIMEOUT = 20.0

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Search the internet. Tries providers in order until one works."""
        if settings.SERPER_API_KEY:
            docs = await self._serper_search(query, max_results)
            if docs:
                return docs

        # Try DuckDuckGo HTML (richer results)
        docs = await self._duckduckgo_html_search(query, max_results)
        if docs:
            return docs

        # Final fallback: DuckDuckGo Instant Answer API
        return await self._duckduckgo_instant_search(query, max_results)

    async def _serper_search(self, query: str, max_results: int) -> list[dict]:
        """High-quality Google search via Serper API."""
        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                resp = await client.post(
                    self.SERPER_URL,
                    headers={
                        "X-API-KEY": settings.SERPER_API_KEY,
                        "Content-Type": "application/json",
                    },
                    json={"q": query, "num": max_results},
                )
                resp.raise_for_status()
                data = resp.json()
                docs = []

                # Answer box / knowledge graph (best quality)
                answer_box = data.get("answerBox", {})
                if answer_box.get("answer") or answer_box.get("snippet"):
                    content = answer_box.get("answer") or answer_box.get("snippet", "")
                    docs.append(self._build_doc(
                        answer_box.get("title", query),
                        content,
                        answer_box.get("link", ""),
                        "Google/Serper (Answer Box)",
                    ))

                for item in data.get("organic", [])[:max_results]:
                    snippet = item.get("snippet", "")
                    if snippet and len(snippet) > 40:
                        docs.append(self._build_doc(
                            item.get("title", ""),
                            snippet,
                            item.get("link", ""),
                            "Google/Serper",
                        ))

                logger.info(f"Serper: {len(docs)} results for '{query[:50]}'")
                return docs[:max_results]
        except Exception as e:
            logger.warning(f"Serper failed: {e}")
            return []

    async def _duckduckgo_html_search(self, query: str, max_results: int) -> list[dict]:
        """
        DuckDuckGo HTML search — richer results than the Instant Answer API.
        Parses the HTML response to extract result snippets.
        """
        try:
            async with httpx.AsyncClient(
                timeout=self.TIMEOUT,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; ARAG_PHARMA/3.0; "
                        "+https://github.com/arag-pharma)"
                    )
                },
                follow_redirects=True,
            ) as client:
                resp = await client.post(
                    self.DUCKDUCKGO_HTML_URL,
                    data={"q": query, "b": "", "kl": "us-en"},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                resp.raise_for_status()
                html = resp.text

                docs = []
                # Extract result snippets using regex (avoids heavy HTML parser dependency)
                # DuckDuckGo HTML results have class="result__snippet"
                snippets = re.findall(
                    r'class="result__snippet"[^>]*>(.*?)</a>',
                    html, re.DOTALL
                )
                titles = re.findall(
                    r'class="result__a"[^>]*>(.*?)</a>',
                    html, re.DOTALL
                )
                urls = re.findall(
                    r'class="result__url"[^>]*>(.*?)</a>',
                    html, re.DOTALL
                )

                for i, (snippet, title) in enumerate(zip(snippets, titles)):
                    # Strip HTML tags
                    clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                    clean_title = re.sub(r'<[^>]+>', '', title).strip()
                    url = urls[i].strip() if i < len(urls) else ""
                    url = re.sub(r'<[^>]+>', '', url).strip()

                    if clean_snippet and len(clean_snippet) > 40:
                        docs.append(self._build_doc(
                            clean_title, clean_snippet, url, "DuckDuckGo"
                        ))
                    if len(docs) >= max_results:
                        break

                logger.info(f"DuckDuckGo HTML: {len(docs)} results for '{query[:50]}'")
                return docs
        except Exception as e:
            logger.warning(f"DuckDuckGo HTML search failed: {e}")
            return []

    async def _duckduckgo_instant_search(self, query: str, max_results: int) -> list[dict]:
        """DuckDuckGo Instant Answer API — simple fallback."""
        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                resp = await client.get(
                    self.DUCKDUCKGO_URL,
                    params={
                        "q": query,
                        "format": "json",
                        "no_redirect": "1",
                        "no_html": "1",
                        "skip_disambig": "1",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                docs = []

                abstract = data.get("AbstractText", "")
                if abstract and len(abstract) > 50:
                    docs.append(self._build_doc(
                        data.get("Heading", query),
                        abstract,
                        data.get("AbstractURL", ""),
                        data.get("AbstractSource", "DuckDuckGo"),
                    ))

                for topic in data.get("RelatedTopics", []):
                    if isinstance(topic, dict):
                        text = topic.get("Text", "")
                        url = topic.get("FirstURL", "")
                        if text and len(text) > 40:
                            docs.append(self._build_doc("Related", text, url, "DuckDuckGo"))
                    if len(docs) >= max_results:
                        break

                logger.info(f"DuckDuckGo Instant: {len(docs)} results for '{query[:50]}'")
                return docs[:max_results]
        except Exception as e:
            logger.error(f"DuckDuckGo Instant failed: {e}")
            return []

    def _build_doc(self, title: str, content: str, url: str, provider: str) -> dict:
        """Build a standardised web result document."""
        return {
            "source": settings.WEB_SOURCE_TAG,
            "source_type": "web_search",
            "provider": provider,
            "content": (
                f"[🌐 Web Search — {provider}]\n"
                f"Title: {title}\n"
                f"Content: {content[:2000]}\n"
                f"URL: {url}\n"
                f"⚠️ Internet source — verify with authoritative databases."
            ),
            "url": url,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {"provider": provider, "is_web_source": True},
        }
