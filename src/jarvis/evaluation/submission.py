"""Generate JSONL submission files for the GAIA test set leaderboard."""
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress

sys.path.insert(0, str(Path(__file__).parents[4] / "config"))
from settings import settings

from ..data.loader import GaiaDataLoader
from ..graph.orchestrator import get_graph

console = Console()


class SubmissionGenerator:
    """Generates a JSONL submission file for the GAIA test set."""

    def __init__(
        self,
        output_path: str = "outputs/submission.jsonl",
        level: Optional[int] = None,
        hf_token: Optional[str] = None,
    ):
        self.output_path = output_path
        self.level = level
        self.loader = GaiaDataLoader(
            cache_dir=settings.data_cache_dir,
            hf_token=hf_token or settings.huggingface_token or None,
        )
        self.graph = get_graph()

    def generate(self) -> str:
        """Run the graph on all test tasks and write a JSONL submission file.

        Returns the path to the written file.
        """
        self.loader.download()
        tasks = self.loader.load_test(level=self.level)
        console.print(f"[bold]Generating submission for {len(tasks)} test tasks...[/bold]")

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        written = 0
        with open(self.output_path, "w") as out_file, Progress(console=console) as progress:
            task_bar = progress.add_task("Processing...", total=len(tasks))

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
                    model_answer = state.get("final_answer", "")
                except Exception as e:
                    console.print(f"[red]Error on {gaia_task.task_id}: {e}[/red]")
                    model_answer = ""

                line = json.dumps({
                    "task_id": gaia_task.task_id,
                    "model_answer": model_answer,
                })
                out_file.write(line + "\n")
                written += 1
                progress.advance(task_bar)
                time.sleep(0.5)  # Rate limit buffer

        console.print(f"[bold green]Submission written to: {self.output_path}[/bold green]")
        console.print(f"  Tasks answered: {written}")
        return self.output_path
