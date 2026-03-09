"""Web browsing agent using Gemini-2.0-Flash with Tavily search."""
import os
import sys
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

sys.path.insert(0, str(Path(__file__).parents[4] / "config"))
from settings import settings

from ..tools.search import get_search_tool
from ..tools.web_fetch import fetch_page
from ..tools.code_exec import get_python_repl

WEB_AGENT_PROMPT = """You are a web research specialist. Your job is to find accurate information online to answer questions.

Strategy:
1. Start with a targeted web search using the search tool
2. Fetch specific pages for detailed information when needed
3. Cross-reference multiple sources for accuracy
4. Use the Python REPL for any calculations on retrieved data

Important instructions:
- Verify information from at least 2 sources when possible
- GAIA expects short, exact answers — not summaries
- End your response with "FINAL ANSWER: <your answer>" on its own line
- For numbers, use digits. For lists, use comma separation.
"""


def create_web_agent():
    """Create the Gemini web browsing ReAct agent."""
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key

    llm = ChatGoogleGenerativeAI(
        model=settings.web_model,
        temperature=0,
        google_api_key=settings.google_api_key,
    )
    tools = [get_search_tool(), fetch_page, get_python_repl()]
    return create_react_agent(
        llm,
        tools,
        prompt=WEB_AGENT_PROMPT,
        max_iterations=settings.max_agent_iterations,
    )
