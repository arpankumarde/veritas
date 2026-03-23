"""Tools for fact-checking agents."""

from .academic_search import AcademicPaper, AcademicSearchTool
from .web_search import SearchResult, WebSearchTool

__all__ = ["WebSearchTool", "SearchResult", "AcademicSearchTool", "AcademicPaper"]
