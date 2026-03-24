"""Web search and scraping using Bright Data API.

Supports:
- Multi-engine SERP (Google + Bing) via BrightDataClient
- Web Unlocker page scraping (markdown)
- Platform-specific structured data extraction (X, Reddit, YouTube, etc.)
- Batch scraping for parallel page fetches
- Deep scraping: page markdown + structured platform data in one call

Set BRIGHT_DATA_SERP_ZONE for best results. If not set, falls back to
scraping Google via Web Unlocker and parsing raw HTML.
"""

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

from ..logging_config import get_logger
from .bright_data import (
    BrightDataClient,
    ScrapedPage,
    SerpResult,
    PlatformData,
    detect_platform,
    format_platform_data,
)

logger = get_logger(__name__)

dotenv_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    content: str | None = None
    engine: str = "google"
    platform: str | None = None          # detected platform (e.g. "reddit_posts")
    platform_data: dict | None = None    # structured data from platform scraper


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML tag stripper."""
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
    def handle_data(self, data):
        self._parts.append(data)
    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html(html: str) -> str:
    p = _HTMLTextExtractor()
    p.feed(html)
    return p.get_text()


class WebSearchTool:
    """Web search and scraping via Bright Data.

    Uses BrightDataClient under the hood for:
    - Multi-engine SERP search (Google + Bing by default)
    - Page scraping to markdown
    - Platform-specific structured data (auto-detected from URL)
    - Batch scraping for parallel page fetches

    Zone selection:
        BRIGHT_DATA_SERP_ZONE  — dedicated SERP zone (parsed_light JSON)
        BRIGHT_DATA_ZONE       — Web Unlocker zone for page scraping + SERP fallback
    """

    _API_ENDPOINT = "https://api.brightdata.com/request"

    def __init__(
        self,
        api_token: str | None = None,
        zone: str | None = None,
        serp_zone: str | None = None,
        max_results: int = 10,
        multi_engine: bool = True,
    ):
        self.api_token = api_token or os.environ.get("BRIGHT_DATA_API_TOKEN", "")
        self.zone = zone or os.environ.get("BRIGHT_DATA_ZONE", "mcp_unlocker")
        self.serp_zone = serp_zone or os.environ.get("BRIGHT_DATA_SERP_ZONE", "")
        self.max_results = max_results
        self.multi_engine = multi_engine
        self._search_count = 0

        if not self.api_token:
            raise ValueError(
                "Bright Data API token required. Set BRIGHT_DATA_API_TOKEN env var."
            )

        # Initialize the full Bright Data client
        self.client = BrightDataClient(
            api_token=self.api_token,
            zone=self.zone,
            serp_zone=self.serp_zone,
        )

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(self, query_text: str) -> tuple[list[SearchResult], str]:
        """Search via Bright Data. Uses multi-engine (Google+Bing) by default.

        Falls back to single-engine Google if multi-engine fails or is disabled.
        """
        self._search_count += 1
        logger.info("Web search: query=%s", query_text[:200])

        if self.serp_zone and self.multi_engine:
            return await self._search_multi_engine(query_text)
        if self.serp_zone:
            return await self._search_serp(query_text)
        return await self._search_unlocker_fallback(query_text)

    async def _search_multi_engine(
        self, query_text: str
    ) -> tuple[list[SearchResult], str]:
        """Search Google + Bing in parallel, merge results."""
        try:
            serp_results = await self.client.search_multi_engine(
                query_text, engines=["google", "bing"]
            )
        except Exception as e:
            logger.warning("Multi-engine search failed, falling back to Google: %s", e)
            return await self._search_serp(query_text)

        if not serp_results:
            return await self._search_serp(query_text)

        results = [
            SearchResult(
                title=r.title,
                url=r.url,
                snippet=r.description,
                engine=r.engine,
                platform=detect_platform(r.url),
            )
            for r in serp_results
        ][:self.max_results]

        return results, self._build_summary(query_text, results)

    async def _search_serp(self, query_text: str) -> tuple[list[SearchResult], str]:
        """Search using dedicated SERP API zone (returns parsed JSON)."""
        import httpx

        search_url = f"https://www.google.com/search?q={quote(query_text)}&hl=en&gl=us"

        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        self._API_ENDPOINT,
                        headers=self._headers,
                        json={
                            "url": search_url,
                            "zone": self.serp_zone,
                            "format": "raw",
                            "data_format": "parsed_light",
                        },
                    )
                    response.raise_for_status()
                    if not response.text.strip():
                        raise ValueError("Empty SERP response")
                    data = response.json()

                results = self._parse_google_results(data)
                return results[:self.max_results], self._build_summary(query_text, results)

            except Exception as e:
                if attempt < 2:
                    logger.warning("SERP search attempt %d failed: %s", attempt + 1, e)
                    await asyncio.sleep((2 ** attempt) + random.uniform(0, 0.5))
                else:
                    logger.warning("SERP failed, falling back to unlocker: %s", e)
                    return await self._search_unlocker_fallback(query_text)

        return [], "Search failed"

    async def _search_unlocker_fallback(self, query_text: str) -> tuple[list[SearchResult], str]:
        """Fallback: fetch Google via Web Unlocker and parse raw HTML."""
        import httpx

        search_url = f"https://www.google.com/search?q={quote(query_text)}&hl=en&gl=us&num=10"

        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=45.0) as client:
                    response = await client.post(
                        self._API_ENDPOINT,
                        headers=self._headers,
                        json={
                            "url": search_url,
                            "zone": self.zone,
                            "format": "raw",
                        },
                    )
                    logger.debug(
                        "Unlocker SERP response: status=%d, len=%d",
                        response.status_code, len(response.text),
                    )
                    response.raise_for_status()

                    body = response.text.strip()
                    if not body:
                        raise ValueError("Empty response from Web Unlocker")

                    if body.startswith("{"):
                        try:
                            data = json.loads(body)
                            if "status_code" in data and data.get("status_code") != 200:
                                raise ValueError(f"Bright Data error: {data}")
                            results = self._parse_google_results(data)
                            if results:
                                return results[:self.max_results], self._build_summary(query_text, results)
                        except json.JSONDecodeError:
                            pass

                    results = self._parse_google_html(body)
                    return results[:self.max_results], self._build_summary(query_text, results)

            except Exception as e:
                if attempt < 2:
                    logger.warning("Unlocker search attempt %d failed: %s", attempt + 1, e)
                    await asyncio.sleep((2 ** attempt) + random.uniform(0, 0.5))
                else:
                    logger.error("Web search failed after 3 attempts: %s", e, exc_info=True)
                    return [], f"Search failed: {e}"

        return [], "Search failed: exhausted retries"

    async def search_and_summarize(self, query_text: str) -> str:
        _, summary = await self.search(query_text)
        return summary

    # ------------------------------------------------------------------
    # Page scraping
    # ------------------------------------------------------------------

    async def fetch_page(self, url: str, extract_prompt: str = "") -> str | None:
        """Scrape a page via Bright Data Web Unlocker (returns markdown)."""
        page = await self.client.scrape_page(url, fmt="markdown")
        return page.content if page.ok and page.content else None

    async def fetch_pages_batch(self, urls: list[str]) -> list[ScrapedPage]:
        """Scrape multiple pages in parallel."""
        return await self.client.scrape_batch(urls, fmt="markdown", max_concurrent=5)

    # ------------------------------------------------------------------
    # Platform-specific structured data
    # ------------------------------------------------------------------

    async def fetch_platform_data(self, url: str) -> PlatformData | None:
        """Auto-detect platform from URL and fetch structured data.

        Returns None if URL doesn't match a known platform.
        """
        return await self.client.fetch_platform_data(url)

    async def fetch_platform_batch(self, urls: list[str]) -> list[PlatformData]:
        """Fetch structured platform data for multiple URLs."""
        return await self.client.fetch_platform_batch(urls)

    # ------------------------------------------------------------------
    # Deep scraping (markdown + platform data)
    # ------------------------------------------------------------------

    async def deep_scrape(self, url: str) -> dict:
        """Scrape page as markdown AND extract structured platform data if applicable."""
        return await self.client.deep_scrape(url)

    async def deep_scrape_batch(self, urls: list[str]) -> list[dict]:
        """Deep scrape multiple URLs in parallel."""
        return await self.client.deep_scrape_batch(urls, max_concurrent=5)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_google_results(self, data: dict) -> list[SearchResult]:
        """Parse Bright Data's parsed_light SERP response."""
        results = []
        for entry in data.get("organic", []):
            link = entry.get("link", "").strip()
            title = entry.get("title", "").strip()
            if not link or not title:
                continue
            results.append(SearchResult(
                title=title,
                url=link,
                snippet=entry.get("description", "").strip(),
                platform=detect_platform(link),
            ))
        return results

    def _parse_google_html(self, html: str) -> list[SearchResult]:
        """Parse Google search results from raw HTML (fallback)."""
        results = []

        link_pattern = re.compile(
            r'<a[^>]+href="/url\?q=([^"&]+)[^"]*"[^>]*>(.*?)</a>',
            re.DOTALL,
        )
        for url_match, title_html in link_pattern.findall(html):
            url = url_match.strip()
            title = _strip_html(title_html).strip()
            if not title or not url or url.startswith("/"):
                continue
            if any(skip in url for skip in ["google.com", "youtube.com/redirect", "accounts.google"]):
                continue
            results.append(SearchResult(title=title, url=url, snippet=""))

        if not results:
            h3_pattern = re.compile(
                r'<a[^>]+href="(https?://[^"]+)"[^>]*>.*?<h3[^>]*>(.*?)</h3>',
                re.DOTALL,
            )
            for url, title_html in h3_pattern.findall(html):
                title = _strip_html(title_html).strip()
                if title and url and "google.com" not in url:
                    results.append(SearchResult(title=title, url=url, snippet=""))

        for r in results[:10]:
            idx = html.find(r.url)
            if idx > 0:
                chunk = html[idx:idx + 1000]
                span_match = re.search(
                    r'<span[^>]*class="[^"]*"[^>]*>([\s\S]{20,300}?)</span>',
                    chunk,
                )
                if span_match:
                    r.snippet = _strip_html(span_match.group(1)).strip()[:300]

        seen = set()
        unique = []
        for r in results:
            if r.url not in seen:
                seen.add(r.url)
                unique.append(r)
        return unique

    def _build_summary(self, query: str, results: list[SearchResult]) -> str:
        if not results:
            return f"No results found for: {query}"
        lines = [f"Search results for: {query}\n"]
        for i, r in enumerate(results[:5], 1):
            engine_tag = f" [{r.engine}]" if hasattr(r, "engine") and r.engine != "google" else ""
            platform_tag = f" ({r.platform})" if r.platform else ""
            lines.append(f"{i}. {r.title}{engine_tag}{platform_tag}")
            lines.append(f"   {r.url}")
            if r.snippet:
                lines.append(f"   {r.snippet}")
            lines.append("")
        return "\n".join(lines)

    @property
    def search_count(self) -> int:
        return self._search_count

    def reset_count(self) -> None:
        self._search_count = 0
