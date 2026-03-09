import os
import sys
from typing import Optional
from pathlib import Path

# Add config to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "config"))

from .schemas import GaiaTask


class GaiaDataLoader:
    """Loads and manages the GAIA benchmark dataset from HuggingFace."""

    def __init__(self, cache_dir: str = "data/gaia", hf_token: Optional[str] = None):
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.data_dir: Optional[str] = None

    def download(self) -> str:
        """Download the GAIA dataset snapshot. Returns the local data directory path."""
        from huggingface_hub import snapshot_download

        print("Downloading GAIA dataset from HuggingFace...")
        self.data_dir = snapshot_download(
            repo_id="gaia-benchmark/GAIA",
            repo_type="dataset",
            local_dir=self.cache_dir,
            token=self.hf_token,
        )
        print(f"Dataset downloaded to: {self.data_dir}")
        return self.data_dir

    def _ensure_downloaded(self):
        if self.data_dir is None:
            if os.path.exists(self.cache_dir):
                self.data_dir = self.cache_dir
            else:
                self.download()

    def _load_split(self, split: str, level: Optional[int] = None) -> list[GaiaTask]:
        from datasets import load_dataset

        self._ensure_downloaded()
        ds = load_dataset(
            self.data_dir,
            "2023_all",
            split=split,
            trust_remote_code=True,
        )

        tasks = []
        for row in ds:
            # Map dataset column names to our schema
            task = GaiaTask(
                task_id=row.get("task_id", ""),
                question=row.get("Question", ""),
                level=row.get("Level", 1),
                final_answer=row.get("Final answer") or None,
                file_name=row.get("file_name") or None,
                file_path=row.get("file_path") or None,
            )
            if level is None or task.level == level:
                tasks.append(task)

        return tasks

    def load_validation(self, level: Optional[int] = None) -> list[GaiaTask]:
        """Load the validation split (165 questions with ground truth answers)."""
        return self._load_split("validation", level=level)

    def load_test(self, level: Optional[int] = None) -> list[GaiaTask]:
        """Load the test split (answers hidden, for submission generation)."""
        return self._load_split("test", level=level)

    def get_file_path(self, task: GaiaTask) -> Optional[str]:
        """Return the absolute local path to the task's attached file, or None."""
        if not task.file_path:
            return None
        self._ensure_downloaded()
        abs_path = os.path.join(self.data_dir, task.file_path)
        return abs_path if os.path.exists(abs_path) else None
