"""Bright Data scraping toolkit — multi-engine search, page scraping, and platform-specific extractors.

Wraps every scrapable Bright Data product into a single async client:
  - Multi-engine SERP (Google, Bing, Yandex)
  - Page scraping (markdown / HTML, single + batch)
  - Structured web-data extractors for 25+ platforms

All methods are async and use httpx for HTTP.
"""

import asyncio
import json
import os
import re
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from dotenv import load_dotenv

from ..logging_config import get_logger

logger = get_logger(__name__)

dotenv_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SerpResult:
    """A single SERP result."""
    title: str
    url: str
    description: str
    engine: str = "google"


@dataclass
class ScrapedPage:
    """Result of scraping a single URL."""
    url: str
    content: str  # markdown or html
    format: str = "markdown"  # "markdown" | "html"
    ok: bool = True
    error: str | None = None


@dataclass
class PlatformData:
    """Structured data extracted from a known platform."""
    platform: str  # e.g. "x_posts", "reddit_posts"
    url: str
    data: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str | None = None


# ---------------------------------------------------------------------------
# Platform URL detection
# ---------------------------------------------------------------------------

# Maps (domain_regex, path_regex | None) → Bright Data web_data endpoint suffix
_PLATFORM_MATCHERS: list[tuple[str, str | None, str]] = [
    # Social media
    (r"(twitter|x)\.com", r"/\w+/status/", "x_posts"),
    (r"reddit\.com", r"/r/.+/comments/", "reddit_posts"),
    (r"instagram\.com", r"/p/", "instagram_posts"),
    (r"instagram\.com", r"/reel/", "instagram_reels"),
    (r"instagram\.com", None, "instagram_profiles"),  # fallback profile
    (r"facebook\.com", r"/(posts|permalink|story)", "facebook_posts"),
    (r"facebook\.com", r"/events/", "facebook_events"),
    (r"tiktok\.com", r"/@.+/video/", "tiktok_posts"),
    (r"tiktok\.com", r"/@[^/]+/?$", "tiktok_profiles"),
    # Video
    (r"youtube\.com", r"/watch", "youtube_videos"),
    (r"youtu\.be", None, "youtube_videos"),
    (r"youtube\.com", r"/(c/|channel/|@)", "youtube_profiles"),
    # News
    (r"reuters\.com", None, "reuter_news"),
    # Business / Finance
    (r"linkedin\.com", r"/in/", "linkedin_person_profile"),
    (r"linkedin\.com", r"/company/", "linkedin_company_profile"),
    (r"linkedin\.com", r"/posts/", "linkedin_posts"),
    (r"linkedin\.com", r"/jobs/", "linkedin_job_listings"),
    (r"crunchbase\.com", r"/organization/", "crunchbase_company"),
    (r"zoominfo\.com", r"/c/", "zoominfo_company_profile"),
    (r"finance\.yahoo\.com", None, "yahoo_finance_business"),
    # App stores
    (r"play\.google\.com", r"/store/apps/", "google_play_store"),
    (r"apps\.apple\.com", None, "apple_app_store"),
    # Other
    (r"github\.com", r"/.+/.+/blob/", "github_repository_file"),
]


def detect_platform(url: str) -> str | None:
    """Return the Bright Data web_data endpoint name if URL matches a known platform."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().lstrip("www.")
        path = parsed.path
    except Exception:
        return None

    for domain_re, path_re, endpoint in _PLATFORM_MATCHERS:
        if re.search(domain_re, domain):
            if path_re is None or re.search(path_re, path):
                return endpoint
    return None


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

_BD_API = "https://api.brightdata.com"


class BrightDataClient:
    """Full-featured Bright Data async client.

    Env vars:
        BRIGHT_DATA_API_TOKEN  — required
        BRIGHT_DATA_ZONE       — Web Unlocker zone (default: sdk_unlocker)
        BRIGHT_DATA_SERP_ZONE  — dedicated SERP zone (default: sdk_serp)
    """

    def __init__(
        self,
        api_token: str | None = None,
        zone: str | None = None,
        serp_zone: str | None = None,
    ):
        self.api_token = api_token or os.environ.get("BRIGHT_DATA_API_TOKEN", "")
        self.zone = zone or os.environ.get("BRIGHT_DATA_ZONE", "sdk_unlocker")
        self.serp_zone = serp_zone or os.environ.get("BRIGHT_DATA_SERP_ZONE", "sdk_serp")

        if not self.api_token:
            raise ValueError("BRIGHT_DATA_API_TOKEN required")

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Low-level request helper
    # ------------------------------------------------------------------

    async def _request(
        self,
        url: str,
        payload: dict[str, Any],
        timeout: float = 45.0,
        retries: int = 2,
    ) -> Any:
        """POST to Bright Data with retries."""
        import httpx

        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(url, headers=self._headers, json=payload)
                    resp.raise_for_status()
                    text = resp.text.strip()
                    if not text:
                        raise ValueError("Empty response")
                    # Try JSON first, fall back to raw text
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return text
            except Exception as e:
                last_err = e
                if attempt < retries:
                    await asyncio.sleep((2 ** attempt) + random.uniform(0, 0.5))
                    logger.warning("Bright Data request retry %d: %s", attempt + 1, e)
        raise last_err  # type: ignore[misc]

    # ------------------------------------------------------------------
    # 1) Multi-engine SERP search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        engine: str = "google",
        country: str = "us",
    ) -> list[SerpResult]:
        """Search a single engine and return parsed results."""
        search_urls = {
            "google": f"https://www.google.com/search?q={quote(query)}&hl=en&gl={country}",
            "bing": f"https://www.bing.com/search?q={quote(query)}",
            "yandex": f"https://yandex.com/search/?text={quote(query)}",
        }
        search_url = search_urls.get(engine, search_urls["google"])

        try:
            data = await self._request(
                f"{_BD_API}/request",
                {
                    "url": search_url,
                    "zone": self.serp_zone,
                    "format": "raw",
                    "data_format": "parsed_light",
                },
            )
        except Exception as e:
            logger.warning("%s search failed: %s", engine, e)
            return []

        if isinstance(data, str):
            # Got raw HTML instead of JSON — skip
            return []

        results: list[SerpResult] = []
        for entry in data.get("organic", []):
            link = entry.get("link", "").strip()
            title = entry.get("title", "").strip()
            if link and title:
                results.append(SerpResult(
                    title=title,
                    url=link,
                    description=entry.get("description", "").strip(),
                    engine=engine,
                ))
        return results

    async def search_multi_engine(
        self,
        query: str,
        engines: list[str] | None = None,
        country: str = "us",
    ) -> list[SerpResult]:
        """Search multiple engines in parallel and merge results (deduplicated by URL)."""
        engines = engines or ["google", "bing"]
        tasks = [self.search(query, engine=e, country=country) for e in engines]
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        seen_urls: set[str] = set()
        merged: list[SerpResult] = []
        # Google first, then others
        for result_list in results_lists:
            if isinstance(result_list, Exception):
                continue
            for r in result_list:
                normalized = r.url.rstrip("/").lower()
                if normalized not in seen_urls:
                    seen_urls.add(normalized)
                    merged.append(r)
        return merged

    async def search_batch(
        self,
        queries: list[str],
        engine: str = "google",
        country: str = "us",
    ) -> dict[str, list[SerpResult]]:
        """Run multiple queries in parallel on the same engine."""
        tasks = {q: self.search(q, engine=engine, country=country) for q in queries}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        out: dict[str, list[SerpResult]] = {}
        for query, result in zip(tasks.keys(), results):
            out[query] = result if isinstance(result, list) else []
        return out

    # ------------------------------------------------------------------
    # 2) Page scraping (markdown / HTML)
    # ------------------------------------------------------------------

    async def scrape_page(self, url: str, fmt: str = "markdown") -> ScrapedPage:
        """Scrape a single URL and return content in markdown or HTML."""
        try:
            data = await self._request(
                f"{_BD_API}/request",
                {
                    "url": url,
                    "zone": self.zone,
                    "format": "raw",
                    "data_format": fmt,
                },
                timeout=60.0,
            )
            content = data if isinstance(data, str) else json.dumps(data)
            return ScrapedPage(url=url, content=content, format=fmt)
        except Exception as e:
            logger.warning("Scrape failed for %s: %s", url[:80], e)
            return ScrapedPage(url=url, content="", format=fmt, ok=False, error=str(e))

    async def scrape_batch(
        self,
        urls: list[str],
        fmt: str = "markdown",
        max_concurrent: int = 5,
    ) -> list[ScrapedPage]:
        """Scrape multiple URLs in parallel (respecting concurrency limit)."""
        sem = asyncio.Semaphore(max_concurrent)

        async def _scrape(u: str) -> ScrapedPage:
            async with sem:
                return await self.scrape_page(u, fmt=fmt)

        return await asyncio.gather(*[_scrape(u) for u in urls])

    # ------------------------------------------------------------------
    # 3) Platform-specific structured data extraction
    # ------------------------------------------------------------------

    async def fetch_platform_data(self, url: str) -> PlatformData | None:
        """Auto-detect platform from URL and fetch structured data.

        Returns None if the URL doesn't match any known platform.
        """
        platform = detect_platform(url)
        if not platform:
            return None

        logger.info("Platform scrape: %s -> %s", url[:80], platform)
        try:
            data = await self._request(
                f"{_BD_API}/request",
                {
                    "url": url,
                    "zone": self.zone,
                    "format": "raw",
                    "data_format": "json",
                },
                timeout=45.0,
            )
            payload = data if isinstance(data, dict) else {"raw": data}
            return PlatformData(platform=platform, url=url, data=payload)
        except Exception as e:
            logger.warning("Platform scrape failed (%s): %s", platform, e)
            return PlatformData(
                platform=platform, url=url, ok=False, error=str(e),
            )

    async def fetch_platform_batch(
        self,
        urls: list[str],
        max_concurrent: int = 5,
    ) -> list[PlatformData]:
        """Fetch structured data for multiple platform URLs in parallel."""
        sem = asyncio.Semaphore(max_concurrent)

        async def _fetch(u: str) -> PlatformData | None:
            async with sem:
                return await self.fetch_platform_data(u)

        results = await asyncio.gather(*[_fetch(u) for u in urls])
        return [r for r in results if r is not None]

    # ------------------------------------------------------------------
    # 4) Convenience: deep scrape (scrape page + platform data if applicable)
    # ------------------------------------------------------------------

    async def deep_scrape(self, url: str) -> dict[str, Any]:
        """Scrape a URL as markdown AND extract structured platform data if applicable.

        Returns a dict with keys: url, markdown, platform_data (or None).
        """
        tasks: list[Any] = [self.scrape_page(url, fmt="markdown")]
        has_platform = detect_platform(url) is not None
        if has_platform:
            tasks.append(self.fetch_platform_data(url))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        page = results[0] if not isinstance(results[0], Exception) else ScrapedPage(
            url=url, content="", ok=False, error=str(results[0]),
        )
        platform = None
        if has_platform and len(results) > 1 and not isinstance(results[1], Exception):
            platform = results[1]

        return {
            "url": url,
            "markdown": page.content if page.ok else "",
            "platform": platform.platform if platform else None,
            "platform_data": platform.data if platform and platform.ok else None,
        }

    async def deep_scrape_batch(
        self,
        urls: list[str],
        max_concurrent: int = 5,
    ) -> list[dict[str, Any]]:
        """Deep scrape multiple URLs in parallel."""
        sem = asyncio.Semaphore(max_concurrent)

        async def _deep(u: str) -> dict[str, Any]:
            async with sem:
                return await self.deep_scrape(u)

        return await asyncio.gather(*[_deep(u) for u in urls])


# ---------------------------------------------------------------------------
# Helpers for formatting platform data as text (for LLM consumption)
# ---------------------------------------------------------------------------

def format_platform_data(pd: PlatformData) -> str:
    """Format structured platform data into a readable text summary for the LLM."""
    if not pd.ok or not pd.data:
        return ""

    d = pd.data
    lines: list[str] = [f"[{pd.platform.upper()} — {pd.url}]"]

    # Generic extraction of common fields
    for key in ("title", "name", "headline", "full_name", "display_name"):
        if key in d:
            lines.append(f"Title: {d[key]}")
            break

    for key in ("text", "content", "body", "description", "summary", "selftext"):
        if key in d and d[key]:
            text = str(d[key])[:2000]
            lines.append(f"Content: {text}")
            break

    # Author / creator
    for key in ("author", "user", "creator", "username", "screen_name"):
        if key in d:
            val = d[key]
            if isinstance(val, dict):
                val = val.get("name") or val.get("username") or str(val)
            lines.append(f"Author: {val}")
            break

    # Date
    for key in ("date", "created_at", "published_at", "timestamp", "upload_date"):
        if key in d:
            lines.append(f"Date: {d[key]}")
            break

    # Engagement metrics
    metrics_parts = []
    for key in ("likes", "like_count", "favorite_count", "upvotes", "score"):
        if key in d:
            metrics_parts.append(f"Likes: {d[key]}")
            break
    for key in ("retweets", "retweet_count", "shares", "share_count"):
        if key in d:
            metrics_parts.append(f"Shares: {d[key]}")
            break
    for key in ("comments", "comment_count", "reply_count", "num_comments"):
        if key in d:
            metrics_parts.append(f"Comments: {d[key]}")
            break
    for key in ("views", "view_count", "play_count"):
        if key in d:
            metrics_parts.append(f"Views: {d[key]}")
            break
    if metrics_parts:
        lines.append("Engagement: " + " | ".join(metrics_parts))

    # Verification / credibility signals
    for key in ("verified", "is_verified"):
        if key in d:
            lines.append(f"Verified: {d[key]}")
            break

    # Subreddit (Reddit)
    if "subreddit" in d:
        lines.append(f"Subreddit: r/{d['subreddit']}")

    # Comments / replies (if array, summarize first few)
    for key in ("comments", "replies"):
        if key in d and isinstance(d[key], list) and d[key]:
            lines.append(f"Top {key} ({len(d[key])}):")
            for c in d[key][:3]:
                if isinstance(c, dict):
                    text = c.get("text") or c.get("content") or c.get("body") or str(c)
                    lines.append(f"  - {str(text)[:200]}")
            break

    return "\n".join(lines)
