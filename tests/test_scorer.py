"""Tests for the answer scorer and normalization."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jarvis.evaluation.scorer import score_answer, compute_metrics
from jarvis.utils.normalize import normalize_answer


class TestNormalize:
    def test_strips_whitespace(self):
        assert normalize_answer("  hello  ") == "hello"

    def test_lowercases(self):
        assert normalize_answer("Paris") == "paris"

    def test_removes_articles(self):
        assert normalize_answer("the Eiffel Tower") == "eiffel tower"
        assert normalize_answer("a cat") == "cat"
        assert normalize_answer("an apple") == "apple"

    def test_normalizes_integer(self):
        assert normalize_answer("42") == "42"
        assert normalize_answer("42.0") == "42"

    def test_normalizes_float(self):
        assert normalize_answer("3.14") == "3.14"
        assert normalize_answer("3.10") == "3.1"

    def test_removes_comma_from_number(self):
        assert normalize_answer("1,000") == "1000"


class TestScoreAnswer:
    def test_exact_match(self):
        assert score_answer("Paris", "Paris") is True

    def test_case_insensitive(self):
        assert score_answer("paris", "Paris") is True

    def test_article_stripped(self):
        assert score_answer("The Eiffel Tower", "Eiffel Tower") is True

    def test_number_match(self):
        assert score_answer("42.0", "42") is True

    def test_wrong_answer(self):
        assert score_answer("London", "Paris") is False

    def test_empty_prediction(self):
        assert score_answer("", "Paris") is False

    def test_none_prediction(self):
        assert score_answer(None, "Paris") is False


class TestComputeMetrics:
    def test_all_correct(self):
        results = [
            {"task_id": "1", "correct": True, "level": 1},
            {"task_id": "2", "correct": True, "level": 2},
        ]
        m = compute_metrics(results)
        assert m["accuracy"] == 1.0
        assert m["correct"] == 2
        assert m["total"] == 2

    def test_half_correct(self):
        results = [
            {"task_id": "1", "correct": True, "level": 1},
            {"task_id": "2", "correct": False, "level": 1},
        ]
        m = compute_metrics(results)
        assert m["accuracy"] == 0.5

    def test_by_level(self):
        results = [
            {"task_id": "1", "correct": True, "level": 1},
            {"task_id": "2", "correct": False, "level": 2},
        ]
        m = compute_metrics(results)
        assert m["by_level"][1]["accuracy"] == 1.0
        assert m["by_level"][2]["accuracy"] == 0.0

    def test_empty(self):
        m = compute_metrics([])
        assert m["accuracy"] == 0.0
