"""Extract a concise final answer from verbose agent output using Gemini-Flash."""
import os
import sys
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).parents[4] / "config"))
from settings import settings

EXTRACTION_PROMPT = """You are an answer extractor for the GAIA benchmark.

The GAIA benchmark expects answers in one of these formats:
- A short string (a word, a name, a phrase)
- A number (integer or decimal)
- A comma-separated list of values

Given the original question and the agent's response, extract ONLY the final answer.

Rules:
1. If the agent's response contains "FINAL ANSWER:", extract exactly what follows that marker.
2. Otherwise, identify the answer from the agent's reasoning and conclusion.
3. Do NOT include explanations, units (unless the question asks for them), or extra words.
4. For numbers: use digits, not words (e.g., "42" not "forty-two").
5. For lists: separate items with commas and a space.

Question: {question}

Agent Response:
{agent_output}

Extracted answer (just the answer, nothing else):"""


_extractor_llm = None


def get_extractor():
    global _extractor_llm
    if _extractor_llm is None:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
        _extractor_llm = ChatGoogleGenerativeAI(
            model=settings.extractor_model,
            temperature=0,
            google_api_key=settings.google_api_key,
        )
    return _extractor_llm


def extract_answer(question: str, agent_output: str) -> str:
    """Use Gemini-Flash to distill agent output into a short exact answer."""
    # First try to parse "FINAL ANSWER:" marker directly (fast path)
    if "FINAL ANSWER:" in agent_output:
        parts = agent_output.split("FINAL ANSWER:")
        candidate = parts[-1].strip().split("\n")[0].strip()
        if candidate:
            return candidate

    # Fall back to LLM extraction
    try:
        llm = get_extractor()
        prompt = EXTRACTION_PROMPT.format(question=question, agent_output=agent_output)
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception:
        # Last resort: return the last non-empty line of the agent output
        lines = [line.strip() for line in agent_output.splitlines() if line.strip()]
        return lines[-1] if lines else ""
