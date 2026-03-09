"""Tests for file loading tools (no LLM required)."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestLoadCsv:
    def test_reads_valid_csv(self):
        from jarvis.tools.file_loaders import load_csv

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,age\nAlice,30\nBob,25\n")
            tmp_path = f.name

        try:
            result = load_csv.invoke({"file_path": tmp_path})
            assert "Alice" in result
            assert "Bob" in result
        finally:
            os.unlink(tmp_path)

    def test_missing_file_returns_error(self):
        from jarvis.tools.file_loaders import load_csv

        result = load_csv.invoke({"file_path": "/nonexistent/path/file.csv"})
        assert "Error" in result or "not found" in result


class TestReadTextFile:
    def test_reads_text(self):
        from jarvis.tools.file_loaders import read_text_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, world!\nThis is a test.")
            tmp_path = f.name

        try:
            result = read_text_file.invoke({"file_path": tmp_path})
            assert "Hello, world!" in result
        finally:
            os.unlink(tmp_path)


class TestMultimodalHelpers:
    def test_detect_image_type(self):
        from jarvis.tools.multimodal import detect_file_type, is_multimodal_file

        assert detect_file_type("photo.jpg").startswith("image/")
        assert detect_file_type("audio.mp3").startswith("audio/")
        assert is_multimodal_file("photo.png") is True
        assert is_multimodal_file("report.pdf") is False
        assert is_multimodal_file(None) is False
