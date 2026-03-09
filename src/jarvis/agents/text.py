"""General-purpose text reasoning agent using Llama3.1 8B via Ollama."""
import sys
from pathlib import Path

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

sys.path.insert(0, str(Path(__file__).parents[4] / "config"))
from settings import settings

from ..tools.file_loaders import FILE_TOOLS
from ..tools.code_exec import get_python_repl

TEXT_AGENT_PROMPT = """You are a general-purpose reasoning and analysis agent.

Your capabilities:
- Read and analyze documents (PDF, Excel, CSV, text files) using the provided file tools
- Perform math calculations and data processing using the Python REPL
- Answer questions through logical reasoning, step-by-step analysis, and coding

Important instructions:
- When you have the answer, state it clearly and concisely
- GAIA expects short, exact answers — do not pad your final answer with explanations
- End your response with "FINAL ANSWER: <your answer>" on its own line
- For numbers, use digits not words. For lists, use comma separation.
"""


def create_text_agent():
    """Create the Llama3.1 text reasoning ReAct agent."""
    llm = ChatOllama(
        model=settings.text_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )
    tools = FILE_TOOLS + [get_python_repl()]
    return create_react_agent(
        llm,
        tools,
        prompt=TEXT_AGENT_PROMPT,
        max_iterations=settings.max_agent_iterations,
    )
