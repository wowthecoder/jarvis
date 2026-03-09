from typing import Optional
from pydantic import BaseModel


class GaiaTask(BaseModel):
    task_id: str
    question: str
    level: int
    final_answer: Optional[str] = None  # None for test split
    file_name: Optional[str] = None
    file_path: Optional[str] = None     # Relative path within dataset repo
