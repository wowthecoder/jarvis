"""Main LangGraph StateGraph orchestrating the multi-agent GAIA solver."""
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from .state import GaiaState
from .router import decide_route
from ..tools.multimodal import detect_file_type, build_multimodal_content, is_multimodal_file
from ..utils.answer_extract import extract_answer

# Lazy-initialized agent graphs (avoid loading all models on import)
_text_agent = None
_multimodal_agent = None
_web_agent = None


def _get_text_agent():
    global _text_agent
    if _text_agent is None:
        from ..agents.text import create_text_agent
        _text_agent = create_text_agent()
    return _text_agent


def _get_multimodal_agent():
    global _multimodal_agent
    if _multimodal_agent is None:
        from ..agents.multimodal import create_multimodal_agent
        _multimodal_agent = create_multimodal_agent()
    return _multimodal_agent


def _get_web_agent():
    global _web_agent
    if _web_agent is None:
        from ..agents.web import create_web_agent
        _web_agent = create_web_agent()
    return _web_agent


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def prepare_input_node(state: GaiaState) -> dict:
    """Detect file type and build the initial HumanMessage.

    For multimodal files (image/audio/video), embeds the file content directly
    into the message using base64 encoding for Gemini's native multimodal input.
    For other files, the message just carries the question text; agents use
    file tools to load the content themselves.
    """
    file_path = state.get("file_path")
    file_name = state.get("file_name")
    question = state["question"]

    file_type = None
    if file_path:
        file_type = detect_file_type(file_path)

    # Build the initial message
    if file_path and is_multimodal_file(file_path) and file_type:
        # Embed media content directly in the message
        content = build_multimodal_content(question, file_path, file_type)
        message = HumanMessage(content=content)
    elif file_path and file_name:
        # Text file — inform agent about the attachment
        message = HumanMessage(
            content=f"{question}\n\n[An attached file is available at: {file_path}]"
        )
    else:
        message = HumanMessage(content=question)

    return {
        "messages": [message],
        "file_type": file_type,
        "iteration_count": 0,
    }


def manager_node(state: GaiaState) -> dict:
    """Route the question to the appropriate specialist agent."""
    agent = decide_route(
        question=state["question"],
        file_name=state.get("file_name"),
        file_type=state.get("file_type"),
    )
    return {"routed_to": agent, "iteration_count": 1}


def _run_agent(agent_graph, state: GaiaState) -> dict:
    """Invoke a ReAct agent graph and capture its output."""
    result = agent_graph.invoke({"messages": state["messages"]})
    messages = result.get("messages", [])

    # Extract the last AI message as the agent's output
    agent_output = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            agent_output = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    return {
        "messages": messages,
        "agent_output": agent_output,
        "iteration_count": 1,
    }


def text_agent_node(state: GaiaState) -> dict:
    return _run_agent(_get_text_agent(), state)


def multimodal_agent_node(state: GaiaState) -> dict:
    return _run_agent(_get_multimodal_agent(), state)


def web_agent_node(state: GaiaState) -> dict:
    return _run_agent(_get_web_agent(), state)


def extract_answer_node(state: GaiaState) -> dict:
    """Distill the agent's verbose output into a short exact answer."""
    agent_output = state.get("agent_output", "")
    question = state["question"]
    final_answer = extract_answer(question, agent_output)
    return {"final_answer": final_answer}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def _route_to_agent(state: GaiaState) -> str:
    """Conditional edge function: maps routed_to value to node name."""
    return state.get("routed_to", "text")


def build_graph():
    """Build and compile the GAIA solver StateGraph."""
    graph = StateGraph(GaiaState)

    # Register nodes
    graph.add_node("prepare_input", prepare_input_node)
    graph.add_node("manager", manager_node)
    graph.add_node("text_agent", text_agent_node)
    graph.add_node("multimodal_agent", multimodal_agent_node)
    graph.add_node("web_agent", web_agent_node)
    graph.add_node("extract_answer", extract_answer_node)

    # Linear edges
    graph.add_edge(START, "prepare_input")
    graph.add_edge("prepare_input", "manager")

    # Conditional routing from manager to one of the three agents
    graph.add_conditional_edges(
        "manager",
        _route_to_agent,
        {
            "text": "text_agent",
            "multimodal": "multimodal_agent",
            "web": "web_agent",
        },
    )

    # All agents feed into the answer extractor
    graph.add_edge("text_agent", "extract_answer")
    graph.add_edge("multimodal_agent", "extract_answer")
    graph.add_edge("web_agent", "extract_answer")
    graph.add_edge("extract_answer", END)

    return graph.compile()


# Module-level singleton — build once on first import of this function
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
