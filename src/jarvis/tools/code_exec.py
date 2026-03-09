"""Python code execution tool for math, data processing, and general computation."""
from langchain_experimental.tools import PythonREPLTool


def get_python_repl() -> PythonREPLTool:
    """Return a Python REPL tool for executing code."""
    return PythonREPLTool(
        description=(
            "Execute Python code for calculations, data analysis, or logic. "
            "Input should be valid Python code as a string. "
            "The result of the last expression or print() output is returned. "
            "Use this for math, string manipulation, list operations, etc."
        )
    )
