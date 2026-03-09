"""Web search tool using Tavily."""
from langchain_community.tools.tavily_search import TavilySearchResults


def get_search_tool(max_results: int = 5) -> TavilySearchResults:
    """Return a configured Tavily search tool."""
    return TavilySearchResults(
        max_results=max_results,
        description=(
            "Search the web for current information. "
            "Input should be a search query string. "
            "Returns a list of relevant web results with titles, URLs, and snippets."
        ),
    )
