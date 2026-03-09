"""Jarvis CLI — Multi-agent GAIA benchmark solver."""
import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="jarvis",
    help="Multi-agent GAIA benchmark solver using LangGraph.",
    add_completion=False,
)
console = Console()

# Add config to path
sys.path.insert(0, str(Path(__file__).parents[3] / "config"))


@app.command()
def evaluate(
    level: Optional[int] = typer.Option(None, "--level", "-l", help="Filter by GAIA difficulty level (1, 2, or 3)."),
    max_tasks: Optional[int] = typer.Option(None, "--max-tasks", "-n", help="Maximum number of tasks to evaluate."),
    output: str = typer.Option("outputs/eval_results.json", "--output", "-o", help="Path to save evaluation results JSON."),
):
    """Run evaluation on the GAIA validation set (includes ground truth answers)."""
    from .evaluation.runner import EvaluationRunner

    runner = EvaluationRunner(level=level, max_tasks=max_tasks)
    metrics = runner.run()

    # Save results
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        # Don't dump full question text to keep the file readable
        save_data = {k: v for k, v in metrics.items() if k != "results"}
        save_data["results"] = metrics.get("results", [])
        json.dump(save_data, f, indent=2)

    console.print(f"[dim]Full results saved to: {output}[/dim]")


@app.command()
def submit(
    output: str = typer.Option("outputs/submission.jsonl", "--output", "-o", help="Path for the output JSONL submission file."),
    level: Optional[int] = typer.Option(None, "--level", "-l", help="Filter by level (for partial submissions)."),
):
    """Generate a JSONL submission file for the GAIA test set leaderboard.

    Upload the output file to: https://huggingface.co/spaces/gaia-benchmark/leaderboard
    """
    from .evaluation.submission import SubmissionGenerator

    generator = SubmissionGenerator(output_path=output, level=level)
    out_path = generator.generate()

    console.print(Panel(
        f"[bold]Submission file:[/bold] {out_path}\n\n"
        "Upload this file to the GAIA leaderboard:\n"
        "[link]https://huggingface.co/spaces/gaia-benchmark/leaderboard[/link]",
        title="Submission Ready",
        border_style="green",
    ))


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer."),
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Path to an attached file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full agent output, not just the final answer."),
):
    """Ask a single question (useful for testing and debugging the system).

    Example:
        jarvis ask "What is the capital of France?"
        jarvis ask "How many rows are in this CSV?" --file data/myfile.csv
    """
    from .graph.orchestrator import get_graph

    graph = get_graph()

    console.print(Panel(f"[bold]Question:[/bold] {question}", border_style="blue"))
    if file:
        console.print(f"  [dim]Attached file: {file}[/dim]")

    try:
        state = graph.invoke(
            {
                "task_id": "interactive",
                "question": question,
                "file_path": file,
                "file_name": os.path.basename(file) if file else None,
                "messages": [],
            },
            config={"recursion_limit": 60},
        )

        routed_to = state.get("routed_to", "unknown")
        final_answer = state.get("final_answer", "")
        agent_output = state.get("agent_output", "")

        console.print(f"\n[dim]Routed to:[/dim] [cyan]{routed_to}[/cyan]")

        if verbose and agent_output:
            console.print(Panel(agent_output, title="Agent Output", border_style="yellow"))

        console.print(Panel(
            f"[bold green]{final_answer}[/bold green]",
            title="Final Answer",
            border_style="green",
        ))

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
