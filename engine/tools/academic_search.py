"""Academic search integration for scholarly paper discovery.

Provides unified access to academic search APIs:
- Semantic Scholar (200M+ papers, citation graphs, TLDRs)
- arXiv (2.4M+ preprints, full-text access)

All APIs are free and require no API keys for basic usage.
"""

import asyncio
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from urllib.parse import quote

from ..logging_config import get_logger
from ..tools.web_search import SearchResult

logger = get_logger(__name__)


# Maximum retries and backoff defaults
_MAX_RETRIES = 5
_BASE_BACKOFF = 1.0  # seconds
_MAX_BACKOFF = 60.0  # seconds
_RATE_LIMIT_BACKOFF = 10.0  # base seconds for 429 responses


def _is_retryable_status(status_code: int) -> bool:
    """Check if an HTTP status code warrants a retry."""
    # 429 = rate limited, 5xx = server error — both retryable
    return status_code == 429 or status_code >= 500


def _backoff_delay(
    attempt: int, base: float = _BASE_BACKOFF, max_delay: float = _MAX_BACKOFF
) -> float:
    """Calculate exponential backoff with jitter.

    Uses full-jitter strategy: delay = random(0, min(max_delay, base * 2^attempt))
    """
    exp_delay = min(max_delay, base * (2**attempt))
    return random.uniform(0, exp_delay)


async def _request_with_retry(
    method: str,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    max_retries: int = _MAX_RETRIES,
    timeout: float = 30.0,
    base_backoff: float = _BASE_BACKOFF,
    api_name: str = "API",
) -> object:
    """Make an HTTP request with exponential backoff and retry logic.

    Handles:
    - 429 rate limits (respects Retry-After header, longer base backoff)
    - 5xx server errors (standard exponential backoff)
    - Connection/timeout errors (standard exponential backoff)
    - Non-retryable 4xx errors (raises immediately)

    Args:
        method: HTTP method ("GET" or "POST")
        url: Request URL
        params: Query parameters
        headers: Request headers
        max_retries: Total number of attempts (including the first request)
        timeout: Request timeout in seconds
        base_backoff: Base backoff delay in seconds
        api_name: Name for logging (e.g., "Semantic Scholar", "arXiv")

    Returns:
        httpx.Response on success

    Raises:
        httpx.HTTPStatusError: On non-retryable HTTP errors (4xx except 429)
        httpx.TimeoutException: If all retries exhausted due to timeouts
        Exception: If all retries exhausted
    """
    import httpx

    last_exception: Exception | None = None

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(method, url, params=params, headers=headers)

            # Success
            if response.status_code < 400:
                return response

            # Rate limited — respect Retry-After header
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = _RATE_LIMIT_BACKOFF
                else:
                    delay = _backoff_delay(attempt, base=_RATE_LIMIT_BACKOFF)
                logger.warning(
                    "%s rate limited (429), retrying in %.1fs (attempt %d/%d)",
                    api_name,
                    delay,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(delay)
                last_exception = httpx.HTTPStatusError(
                    f"Rate limited (429)", request=response.request, response=response,
                )
                continue

            # Server error — retryable
            if response.status_code >= 500:
                delay = _backoff_delay(attempt, base=base_backoff)
                logger.warning(
                    "%s server error (%d), retrying in %.1fs (attempt %d/%d)",
                    api_name,
                    response.status_code,
                    delay,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(delay)
                last_exception = httpx.HTTPStatusError(
                    f"Server error ({response.status_code})",
                    request=response.request,
                    response=response,
                )
                continue

            # Non-retryable client error (400, 403, 404, etc.)
            logger.error(
                "%s request failed with status %d (not retryable): %s",
                api_name,
                response.status_code,
                url,
            )
            response.raise_for_status()

        except httpx.TimeoutException as e:
            delay = _backoff_delay(attempt, base=base_backoff)
            logger.warning(
                "%s request timed out, retrying in %.1fs (attempt %d/%d)",
                api_name,
                delay,
                attempt + 1,
                max_retries,
            )
            await asyncio.sleep(delay)
            last_exception = e

        except httpx.HTTPStatusError:
            # Re-raise non-retryable HTTP errors immediately
            raise

        except (httpx.ConnectError, httpx.ReadError, OSError) as e:
            delay = _backoff_delay(attempt, base=base_backoff)
            logger.warning(
                "%s connection error: %s, retrying in %.1fs (attempt %d/%d)",
                api_name,
                type(e).__name__,
                delay,
                attempt + 1,
                max_retries,
            )
            await asyncio.sleep(delay)
            last_exception = e

        except Exception as e:
            # Unexpected error — log and retry with backoff
            delay = _backoff_delay(attempt, base=base_backoff)
            logger.warning(
                "%s unexpected error: %s, retrying in %.1fs (attempt %d/%d)",
                api_name,
                e,
                delay,
                attempt + 1,
                max_retries,
            )
            await asyncio.sleep(delay)
            last_exception = e

    # All retries exhausted
    logger.error(
        "%s request failed after %d attempts: %s",
        api_name,
        max_retries,
        url,
    )
    raise last_exception  # type: ignore[misc]


@dataclass
class AcademicPaper:
    """Structured academic paper metadata."""

    title: str
    url: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    year: int | None = None
    citation_count: int = 0
    venue: str = ""
    tldr: str = ""
    doi: str = ""
    arxiv_id: str = ""
    source_api: str = ""  # "semantic_scholar" | "arxiv"
    open_access_url: str = ""
    influential_citation_count: int = 0

    def to_search_result(self) -> SearchResult:
        """Convert to SearchResult for compatibility with existing pipeline."""
        snippet_parts = []
        if self.tldr:
            snippet_parts.append(self.tldr)
        elif self.abstract:
            snippet_parts.append(self.abstract[:300])
        if self.authors:
            authors_str = ", ".join(self.authors[:3])
            if len(self.authors) > 3:
                authors_str += " et al."
            snippet_parts.append(f"Authors: {authors_str}")
        if self.year:
            snippet_parts.append(f"Year: {self.year}")
        if self.citation_count:
            snippet_parts.append(f"Citations: {self.citation_count}")
        if self.venue:
            snippet_parts.append(f"Venue: {self.venue}")

        return SearchResult(
            title=self.title,
            url=self.url or self.open_access_url,
            snippet=" | ".join(snippet_parts),
        )


class SemanticScholarSearch:
    """Search Semantic Scholar's API for academic papers.

    Free tier: 100 requests per 5 minutes (no API key needed).
    With API key: 1 request per second sustained.

    API docs: https://api.semanticscholar.org/
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    FIELDS = (
        "title,abstract,year,citationCount,influentialCitationCount,"
        "referenceCount,tldr,authors,venue,openAccessPdf,externalIds,url"
    )

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self._request_count = 0

    @property
    def _headers(self) -> dict:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def search(
        self,
        query: str,
        limit: int = 20,
        year_range: tuple[int, int] | None = None,
        min_citations: int = 0,
        fields_of_study: list[str] | None = None,
    ) -> list[AcademicPaper]:
        """Search for papers by query.

        Args:
            query: Search query string
            limit: Maximum results (max 100)
            year_range: Optional (start_year, end_year) filter
            min_citations: Minimum citation count filter
            fields_of_study: Filter by field (e.g., ["Computer Science", "Medicine"])

        Returns:
            List of AcademicPaper objects
        """
        logger.info("Semantic Scholar search: query=%s", query[:200])
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": self.FIELDS,
        }

        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"

        if min_citations:
            params["minCitationCount"] = str(min_citations)

        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        try:
            response = await _request_with_retry(
                "GET",
                f"{self.BASE_URL}/paper/search",
                params=params,
                headers=self._headers,
                api_name="Semantic Scholar",
            )
            data = response.json()
            self._request_count += 1
            return self._parse_papers(data.get("data", []))
        except Exception as e:
            logger.error("Semantic Scholar search failed: %s", e)
            return []

    async def get_paper(self, paper_id: str) -> AcademicPaper | None:
        """Get detailed paper information by Semantic Scholar ID, DOI, or arXiv ID.

        Args:
            paper_id: Can be S2 ID, DOI (prefix with "DOI:"), arXiv ID (prefix with "ARXIV:"),
                      or URL
        """
        try:
            response = await _request_with_retry(
                "GET",
                f"{self.BASE_URL}/paper/{quote(paper_id, safe=':')}",
                params={"fields": self.FIELDS},
                headers=self._headers,
                api_name="Semantic Scholar",
            )
            data = response.json()
            self._request_count += 1
            papers = self._parse_papers([data])
            return papers[0] if papers else None
        except Exception as e:
            logger.error("Semantic Scholar get_paper failed for %s: %s", paper_id, e)
            return None

    async def get_citations(self, paper_id: str, limit: int = 50) -> list[AcademicPaper]:
        """Get papers that cite the given paper."""
        try:
            response = await _request_with_retry(
                "GET",
                f"{self.BASE_URL}/paper/{quote(paper_id, safe=':')}/citations",
                params={
                    "fields": "title,abstract,year,citationCount,authors,venue,url",
                    "limit": min(limit, 100),
                },
                headers=self._headers,
                api_name="Semantic Scholar",
            )
            data = response.json()
            self._request_count += 1
            citing_papers = []
            for item in data.get("data", []):
                citing = item.get("citingPaper", {})
                if citing.get("title"):
                    citing_papers.append(self._parse_single_paper(citing))
            return citing_papers
        except Exception as e:
            logger.error(
                "Semantic Scholar get_citations failed for %s: %s",
                paper_id,
                e,
            )
            return []

    async def get_recommendations(self, paper_id: str, limit: int = 20) -> list[AcademicPaper]:
        """Get AI-powered paper recommendations based on a seed paper."""
        try:
            rec_url = (
                "https://api.semanticscholar.org"
                f"/recommendations/v1/papers/forpaper/{quote(paper_id, safe=':')}"
            )
            response = await _request_with_retry(
                "GET",
                rec_url,
                params={
                    "fields": self.FIELDS,
                    "limit": min(limit, 100),
                },
                headers=self._headers,
                api_name="Semantic Scholar",
            )
            data = response.json()
            self._request_count += 1
            return self._parse_papers(data.get("recommendedPapers", []))
        except Exception as e:
            logger.error(
                "Semantic Scholar get_recommendations failed for %s: %s",
                paper_id,
                e,
            )
            return []

    def _parse_papers(self, papers_data: list[dict]) -> list[AcademicPaper]:
        """Parse API response into AcademicPaper objects."""
        return [self._parse_single_paper(p) for p in papers_data if p.get("title")]

    def _parse_single_paper(self, p: dict) -> AcademicPaper:
        """Parse a single paper dict into AcademicPaper."""
        authors = [a.get("name", "") for a in (p.get("authors") or [])]

        external_ids = p.get("externalIds") or {}
        doi = external_ids.get("DOI", "")
        arxiv_id = external_ids.get("ArXiv", "")

        tldr = ""
        tldr_data = p.get("tldr")
        if tldr_data and isinstance(tldr_data, dict):
            tldr = tldr_data.get("text", "")

        open_access_url = ""
        oap = p.get("openAccessPdf")
        if oap and isinstance(oap, dict):
            open_access_url = oap.get("url", "")

        url = p.get("url", "")
        if not url and doi:
            url = f"https://doi.org/{doi}"
        elif not url and arxiv_id:
            url = f"https://arxiv.org/abs/{arxiv_id}"

        return AcademicPaper(
            title=p.get("title", ""),
            url=url,
            authors=authors,
            abstract=p.get("abstract", "") or "",
            year=p.get("year"),
            citation_count=p.get("citationCount", 0) or 0,
            influential_citation_count=p.get("influentialCitationCount", 0) or 0,
            venue=p.get("venue", "") or "",
            tldr=tldr,
            doi=doi,
            arxiv_id=arxiv_id,
            source_api="semantic_scholar",
            open_access_url=open_access_url,
        )


class ArxivSearch:
    """Search arXiv's API for preprints.

    Rate limit: 1 request per 3 seconds (be polite).
    API docs: https://info.arxiv.org/help/api/
    """

    BASE_URL = "https://export.arxiv.org/api/query"
    NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    def __init__(self):
        self._request_count = 0

    async def search(
        self,
        query: str,
        max_results: int = 20,
        sort_by: str = "relevance",
        categories: list[str] | None = None,
    ) -> list[AcademicPaper]:
        """Search arXiv for papers.

        Args:
            query: Search query
            max_results: Maximum results (max 100 recommended)
            sort_by: "relevance", "lastUpdatedDate", or "submittedDate"
            categories: Filter by arXiv categories (e.g., ["cs.AI", "cs.CL"])

        Returns:
            List of AcademicPaper objects
        """
        logger.info("arXiv search: query=%s", query[:200])
        # Build search query
        search_query = f"all:{quote(query)}"
        if categories:
            cat_filter = "+OR+".join([f"cat:{cat}" for cat in categories])
            search_query = f"({search_query})+AND+({cat_filter})"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": min(max_results, 100),
            "sortBy": sort_by,
            "sortOrder": "descending",
        }

        try:
            # arXiv asks for 3s between requests, so use larger base backoff
            response = await _request_with_retry(
                "GET",
                self.BASE_URL,
                params=params,
                base_backoff=3.0,
                timeout=30.0,
                api_name="arXiv",
            )
            self._request_count += 1
            return self._parse_atom_feed(response.text)
        except Exception as e:
            logger.error("arXiv search failed: %s", e)
            return []

    def _parse_atom_feed(self, xml_text: str) -> list[AcademicPaper]:
        """Parse arXiv's Atom XML feed into AcademicPaper objects."""
        papers = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

        for entry in root.findall("atom:entry", self.NS):
            title_el = entry.find("atom:title", self.NS)
            title = ""
            if title_el is not None and title_el.text:
                title = title_el.text.strip().replace("\n", " ")

            if not title:
                continue

            summary_el = entry.find("atom:summary", self.NS)
            abstract = ""
            if summary_el is not None and summary_el.text:
                abstract = summary_el.text.strip().replace("\n", " ")

            authors = []
            for author_el in entry.findall("atom:author", self.NS):
                name_el = author_el.find("atom:name", self.NS)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())

            # Get arXiv ID from the entry id URL
            id_el = entry.find("atom:id", self.NS)
            arxiv_url = id_el.text.strip() if id_el is not None and id_el.text else ""
            arxiv_id = ""
            if arxiv_url:
                match = re.search(r"abs/(.+)$", arxiv_url)
                if match:
                    arxiv_id = match.group(1)

            # Get publication date
            published_el = entry.find("atom:published", self.NS)
            year = None
            if published_el is not None and published_el.text:
                try:
                    year = int(published_el.text[:4])
                except (ValueError, IndexError):
                    pass

            # Get PDF link
            pdf_url = ""
            for link_el in entry.findall("atom:link", self.NS):
                if link_el.get("title") == "pdf":
                    pdf_url = link_el.get("href", "")
                    break

            # Get categories
            categories = []
            for cat_el in entry.findall("atom:category", self.NS):
                term = cat_el.get("term", "")
                if term:
                    categories.append(term)

            venue = ", ".join(categories[:3]) if categories else "arXiv preprint"

            # Get DOI if available
            doi_el = entry.find("arxiv:doi", self.NS)
            doi = doi_el.text.strip() if doi_el is not None and doi_el.text else ""

            papers.append(
                AcademicPaper(
                    title=title,
                    url=arxiv_url,
                    authors=authors,
                    abstract=abstract,
                    year=year,
                    citation_count=0,  # arXiv doesn't provide citation counts
                    venue=venue,
                    tldr="",
                    doi=doi,
                    arxiv_id=arxiv_id,
                    source_api="arxiv",
                    open_access_url=pdf_url or arxiv_url,
                )
            )

        return papers


class AcademicSearchTool:
    """Unified academic search across Semantic Scholar and arXiv.

    Combines results from multiple academic APIs, deduplicates, and
    returns papers ranked by relevance and citation count.

    Usage:
        tool = AcademicSearchTool()
        results, summary = await tool.search("transformer attention mechanism")
        papers = await tool.search_papers("CRISPR gene editing", min_citations=50)
    """

    def __init__(
        self,
        semantic_scholar_key: str | None = None,
        max_results: int = 20,
    ):
        self.semantic_scholar = SemanticScholarSearch(api_key=semantic_scholar_key)
        self.arxiv = ArxivSearch()
        self.max_results = max_results
        self._search_count = 0

    async def search(self, query: str) -> tuple[list[SearchResult], str]:
        """Search academic databases and return results compatible with WebSearchTool.

        This method mirrors the WebSearchTool.search() interface so it can be used
        as a drop-in complement in the search pipeline.

        Returns:
            Tuple of (list of SearchResult, summary string)
        """
        logger.info("Academic search: query=%s", query[:200])
        papers = await self.search_papers(query)
        results = [p.to_search_result() for p in papers]
        summary = self._build_summary(query, papers)
        self._search_count += 1
        return results[: self.max_results], summary

    async def search_papers(
        self,
        query: str,
        sources: list[str] | None = None,
        min_citations: int = 0,
        year_range: tuple[int, int] | None = None,
        max_results: int | None = None,
    ) -> list[AcademicPaper]:
        """Search across academic APIs and return unified results.

        Args:
            query: Search query
            sources: Which APIs to search ("semantic_scholar", "arxiv"). Default: both.
            min_citations: Minimum citation count filter (Semantic Scholar only)
            year_range: (start_year, end_year) filter
            max_results: Override default max results

        Returns:
            List of deduplicated AcademicPaper objects, sorted by citation count
        """
        limit = max_results or self.max_results
        sources = sources or ["semantic_scholar", "arxiv"]

        tasks = []
        if "semantic_scholar" in sources:
            tasks.append(
                self.semantic_scholar.search(
                    query,
                    limit=limit,
                    year_range=year_range,
                    min_citations=min_citations,
                )
            )
        if "arxiv" in sources:
            tasks.append(self.arxiv.search(query, max_results=limit))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect all papers
        all_papers = []
        for result in results:
            if isinstance(result, list):
                all_papers.extend(result)

        # Deduplicate by title similarity
        deduped = self._deduplicate_papers(all_papers)

        # Sort by citation count (descending), then by year (descending)
        deduped.sort(
            key=lambda p: (p.citation_count, p.year or 0),
            reverse=True,
        )

        return deduped[:limit]

    async def get_citation_graph(self, paper_id: str, depth: int = 1) -> dict:
        """Build a citation graph from a seed paper.

        Args:
            paper_id: Semantic Scholar paper ID, DOI, or arXiv ID
            depth: How many levels of citations to follow (1 or 2)

        Returns:
            Dict with 'nodes' (papers) and 'edges' (citation links)
        """
        nodes = {}
        edges = []

        seed = await self.semantic_scholar.get_paper(paper_id)
        if not seed:
            return {"nodes": {}, "edges": []}

        nodes[paper_id] = seed

        citations = await self.semantic_scholar.get_citations(paper_id, limit=20)
        for citing_paper in citations:
            cid = citing_paper.doi or citing_paper.arxiv_id or citing_paper.title[:50]
            nodes[cid] = citing_paper
            edges.append({"source": cid, "target": paper_id, "type": "cites"})

        if depth >= 2:
            # Get second-level citations for top-cited papers
            top_citations = sorted(citations, key=lambda p: p.citation_count, reverse=True)[:5]
            for cp in top_citations:
                cp_id = cp.doi or cp.arxiv_id or cp.title[:50]
                try:
                    l2_citations = await self.semantic_scholar.get_citations(
                        f"DOI:{cp.doi}" if cp.doi else cp_id, limit=10
                    )
                    for l2 in l2_citations:
                        l2_id = l2.doi or l2.arxiv_id or l2.title[:50]
                        if l2_id not in nodes:
                            nodes[l2_id] = l2
                        edges.append({"source": l2_id, "target": cp_id, "type": "cites"})
                except Exception:
                    logger.debug("L2 citation fetch failed for %s", cp_id, exc_info=True)
                    continue

        return {"nodes": nodes, "edges": edges}

    def _deduplicate_papers(self, papers: list[AcademicPaper]) -> list[AcademicPaper]:
        """Deduplicate papers by DOI, arXiv ID, or normalized title."""
        seen_ids: set[str] = set()
        seen_titles: set[str] = set()
        unique = []

        for paper in papers:
            # Check DOI
            if paper.doi:
                if paper.doi in seen_ids:
                    continue
                seen_ids.add(paper.doi)

            # Check arXiv ID
            if paper.arxiv_id:
                if paper.arxiv_id in seen_ids:
                    continue
                seen_ids.add(paper.arxiv_id)

            # Check normalized title
            norm_title = self._normalize_title(paper.title)
            if norm_title in seen_titles:
                continue
            seen_titles.add(norm_title)

            unique.append(paper)

        return unique

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Normalize title for deduplication."""
        return re.sub(r"[^a-z0-9]", "", title.lower())

    def _build_summary(self, query: str, papers: list[AcademicPaper]) -> str:
        """Build a text summary from academic search results."""
        if not papers:
            return f"No academic papers found for: {query}"

        lines = [f"Academic search results for: {query}\n"]
        for i, p in enumerate(papers[:5], 1):
            lines.append(f"{i}. {p.title}")
            if p.authors:
                authors_str = ", ".join(p.authors[:3])
                if len(p.authors) > 3:
                    authors_str += " et al."
                lines.append(f"   Authors: {authors_str}")
            lines.append(f"   URL: {p.url}")
            if p.year:
                lines.append(f"   Year: {p.year} | Citations: {p.citation_count}")
            if p.tldr:
                lines.append(f"   TLDR: {p.tldr[:200]}")
            elif p.abstract:
                lines.append(f"   Abstract: {p.abstract[:200]}...")
            if p.venue:
                lines.append(f"   Venue: {p.venue}")
            lines.append("")
        return "\n".join(lines)

    @property
    def search_count(self) -> int:
        """Number of searches performed."""
        return self._search_count

    def reset_count(self) -> None:
        """Reset the search counter."""
        self._search_count = 0
