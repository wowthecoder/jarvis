"""Tool for fetching and extracting text content from web pages."""
from langchain_core.tools import tool


@tool
def fetch_page(url: str) -> str:
    """Fetch a web page and return its main text content.

    Use this to retrieve the full content of a specific URL after finding it
    via web search. Input must be a valid http/https URL.
    """
    import requests
    from bs4 import BeautifulSoup

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Collapse excessive blank lines
        lines = [line for line in text.splitlines() if line.strip()]
        result = "\n".join(lines)

        # Truncate to stay within context limits
        max_chars = 20_000
        if len(result) > max_chars:
            result = result[:max_chars] + "\n\n[... truncated ...]"

        return result or "No text content found on this page."
    except Exception as e:
        return f"Error fetching page: {e}"


WEB_FETCH_TOOLS = [fetch_page]
