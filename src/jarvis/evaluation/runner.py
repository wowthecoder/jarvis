"""Validation set evaluation runner."""
import json
import sys
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parents[4] / "config"))
from settings import settings

from ..data.loader import GaiaDataLoader
from ..graph.orchestrator import get_graph
from .scorer import score_answer, compute_metrics

console = Console()


class EvaluationRunner:
    """Runs GAIA benchmark evaluation on the validation split."""

    def __init__(
        self,
        level: Optional[int] = None,
        max_tasks: Optional[int] = None,
        hf_token: Optional[str] = None,
    ):
        self.level = level
        self.max_tasks = max_tasks
        self.loader = GaiaDataLoader(
            cache_dir=settings.data_cache_dir,
            hf_token=hf_token or settings.huggingface_token or None,
        )
        self.graph = get_graph()

    def run(self) -> dict:
        """Run evaluation. Returns metrics dict with all per-task results."""
        self.loader.download()
        tasks = self.loader.load_validation(level=self.level)

        if self.max_tasks is not None:
            tasks = tasks[: self.max_tasks]

        console.print(f"[bold]Running evaluation on {len(tasks)} validation tasks...[/bold]")
        if self.level:
            console.print(f"  Level filter: {self.level}")

        results = []
        with Progress(console=console) as progress:
            task_bar = progress.add_task("Evaluating...", total=len(tasks))

            for gaia_task in tasks:
                file_path = self.loader.get_file_path(gaia_task)

                try:
                    state = self.graph.invoke(
                        {
                            "task_id": gaia_task.task_id,
                            "question": gaia_task.question,
                            "file_path": file_path,
                            "file_name": gaia_task.file_name,
                            "messages": [],
                        },
                        config={"recursion_limit": settings.max_agent_iterations * 3},
                    )
                    predicted = state.get("final_answer", "")
                    routed_to = state.get("routed_to", "unknown")
                except Exception as e:
                    console.print(f"[red]Error on task {gaia_task.task_id}: {e}[/red]")
                    predicted = ""
                    routed_to = "error"

                is_correct = score_answer(predicted, gaia_task.final_answer or "")
                result = {
                    "task_id": gaia_task.task_id,
                    "question": gaia_task.question[:120],  # Truncate for display
                    "predicted": predicted,
                    "ground_truth": gaia_task.final_answer,
                    "correct": is_correct,
                    "level": gaia_task.level,
                    "routed_to": routed_to,
                }
                results.append(result)
                progress.advance(task_bar)

                # Small pause to avoid rate limiting
                time.sleep(0.5)

        metrics = compute_metrics(results)
        metrics["results"] = results

        self._print_summary(metrics)
        return metrics

    def _print_summary(self, metrics: dict):
        """Print a formatted summary table to console."""
        console.print("\n[bold green]== Evaluation Complete ==[/bold green]")

        table = Table(title="Results by Level")
        table.add_column("Level", style="cyan")
        table.add_column("Correct", style="green")
        table.add_column("Total")
        table.add_column("Accuracy", style="bold")

        for lvl, lvl_data in metrics["by_level"].items():
            table.add_row(
                f"Level {lvl}",
                str(lvl_data["correct"]),
                str(lvl_data["total"]),
                f"{lvl_data['accuracy']:.1%}",
            )

        table.add_row(
            "[bold]Overall[/bold]",
            f"[bold]{metrics['correct']}[/bold]",
            f"[bold]{metrics['total']}[/bold]",
            f"[bold]{metrics['accuracy']:.1%}[/bold]",
        )
        console.print(table)
