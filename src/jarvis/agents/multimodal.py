"""Multimodal analysis agent using Gemini-2.0-Flash via Google AI API."""
import os
import sys
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

sys.path.insert(0, str(Path(__file__).parents[4] / "config"))
from settings import settings

from ..tools.file_loaders import load_pdf, load_excel, load_csv
from ..tools.code_exec import get_python_repl

MULTIMODAL_AGENT_PROMPT = """You are a multimodal analysis specialist with the ability to understand images, audio, and documents.

When analyzing content:
- Images embedded in the message can be directly examined — describe what you see in detail
- Audio files can be transcribed and analyzed
- Use file reading tools for PDFs, spreadsheets, or other documents
- Apply reasoning to answer the question based on all available information

Important instructions:
- GAIA expects short, exact answers
- End your response with "FINAL ANSWER: <your answer>" on its own line
- For numbers, use digits. For lists, use comma separation.
"""


def create_multimodal_agent():
    """Create the Gemini multimodal ReAct agent."""
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key

    llm = ChatGoogleGenerativeAI(
        model=settings.multimodal_model,
        temperature=0,
        google_api_key=settings.google_api_key,
    )
    tools = [load_pdf, load_excel, load_csv, get_python_repl()]
    return create_react_agent(
        llm,
        tools,
        prompt=MULTIMODAL_AGENT_PROMPT,
        max_iterations=settings.max_agent_iterations,
    )
