"""Tools for fact-checking agents."""

from .academic_search import AcademicPaper, AcademicSearchTool
from .bright_data import (
    BrightDataClient,
    PlatformData,
    ScrapedPage,
    SerpResult,
    detect_platform,
    format_platform_data,
)
from .web_search import SearchResult, WebSearchTool

__all__ = [
    "WebSearchTool",
    "SearchResult",
    "AcademicSearchTool",
    "AcademicPaper",
    "BrightDataClient",
    "SerpResult",
    "ScrapedPage",
    "PlatformData",
    "detect_platform",
    "format_platform_data",
]
