"""State definition for the GAIA solver LangGraph."""
import operator
from typing import Annotated, Optional
from langgraph.graph import MessagesState


class GaiaState(MessagesState):
    """Extended state tracking GAIA task metadata alongside the message history."""

    task_id: str = ""
    question: str = ""
    file_path: Optional[str] = None           # Absolute local path to attached file
    file_name: Optional[str] = None           # Original file name (for display/logging)
    file_type: Optional[str] = None           # Detected MIME type
    routed_to: Optional[str] = None           # "text" | "multimodal" | "web"
    agent_output: Optional[str] = None        # Raw output from the chosen worker
    final_answer: Optional[str] = None        # Concise extracted answer for scoring
    iteration_count: Annotated[int, operator.add] = 0  # Guard against infinite loops
