"""Manager routing logic using DeepSeek-R1 8B with structured JSON output."""
import re
import sys
from pathlib import Path
from typing import Literal

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parents[4] / "config"))
from settings import settings


ROUTING_SYSTEM_PROMPT = """You are a task router for a multi-agent AI system.
Given a question and optional file metadata, decide which specialist agent should handle it.

Available agents:
- "multimodal": Use when the question involves an image, audio file, or video file that must be visually or aurally analyzed.
- "web": Use when the question requires finding current or real-time information online, browsing websites, or when the answer cannot be determined from a document or reasoning alone.
- "text": Use for all other tasks — document analysis (PDF, Excel, CSV), math, coding, logical reasoning, and general knowledge questions.

Rules:
1. If the attached file is an image, audio, or video → always choose "multimodal"
2. If the question contains words like "website", "URL", "current", "latest", "today", "search online", "find on the web" → prefer "web"
3. If the attached file is a PDF, spreadsheet, or CSV → choose "text"
4. Default to "text" for pure reasoning or math questions

Respond ONLY with a JSON object in this exact format:
{"reasoning": "brief explanation", "agent": "text|multimodal|web"}"""


class RoutingDecision(BaseModel):
    reasoning: str = Field(description="Brief explanation for the routing choice")
    agent: Literal["text", "multimodal", "web"] = Field(
        description="The agent to route to"
    )


# Rule-based pre-filters (bypass LLM for clear-cut cases)
MULTIMODAL_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".mp3", ".wav", ".ogg", ".mp4", ".mov"}
WEB_KEYWORDS = {"website", "url", "http", "current", "latest", "today", "search online", "find online", "news"}


def rule_based_route(question: str, file_name: str | None, file_type: str | None) -> str | None:
    """Fast rule-based routing for unambiguous cases. Returns agent name or None."""
    if file_name:
        ext = Path(file_name).suffix.lower()
        if ext in MULTIMODAL_EXTENSIONS or (file_type and (file_type.startswith("image/") or file_type.startswith("audio/") or file_type.startswith("video/"))):
            return "multimodal"

    q_lower = question.lower()
    if any(kw in q_lower for kw in WEB_KEYWORDS):
        return "web"

    return None


def parse_routing_fallback(raw_text: str) -> str:
    """Extract agent name from raw LLM text when JSON parsing fails."""
    text = raw_text.lower()
    # Look for explicit agent names in the output
    for agent in ("multimodal", "web", "text"):
        if agent in text:
            return agent
    return "text"  # Default


def create_router_chain():
    """Create the DeepSeek-R1 routing chain with structured JSON output."""
    llm = ChatOllama(
        model=settings.manager_model,
        base_url=settings.ollama_base_url,
        format="json",
        temperature=0,
    )
    return llm.with_structured_output(RoutingDecision)


_router_chain = None


def get_router_chain():
    global _router_chain
    if _router_chain is None:
        _router_chain = create_router_chain()
    return _router_chain


def decide_route(question: str, file_name: str | None, file_type: str | None) -> str:
    """Determine which agent should handle a GAIA question.

    Tries rule-based routing first, then falls back to the LLM.
    Returns "text", "multimodal", or "web".
    """
    # Fast path: rule-based pre-filter
    rule_result = rule_based_route(question, file_name, file_type)
    if rule_result:
        return rule_result

    # LLM-based routing via DeepSeek-R1
    prompt = f"Question: {question}\nFile name: {file_name or 'none'}\nFile type: {file_type or 'none'}"
    try:
        chain = get_router_chain()
        decision: RoutingDecision = chain.invoke([
            HumanMessage(content=ROUTING_SYSTEM_PROMPT + "\n\n" + prompt)
        ])
        return decision.agent
    except Exception as e:
        # If structured output parsing fails, try a raw call with text parsing
        try:
            raw_llm = ChatOllama(
                model=settings.manager_model,
                base_url=settings.ollama_base_url,
                format="json",
                temperature=0,
            )
            response = raw_llm.invoke([HumanMessage(content=ROUTING_SYSTEM_PROMPT + "\n\n" + prompt)])
            return parse_routing_fallback(response.content)
        except Exception:
            return "text"  # Ultimate fallback
