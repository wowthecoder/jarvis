"""Tests for the rule-based routing pre-filters (no LLM required)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "config"))

from jarvis.graph.router import rule_based_route


class TestRuleBasedRoute:
    def test_image_file_routes_to_multimodal(self):
        assert rule_based_route("Describe this image", "photo.jpg", "image/jpeg") == "multimodal"
        assert rule_based_route("What do you hear?", "clip.mp3", "audio/mpeg") == "multimodal"
        assert rule_based_route("What's in the video?", "video.mp4", "video/mp4") == "multimodal"

    def test_png_file_routes_to_multimodal(self):
        assert rule_based_route("What color is this?", "chart.png", None) == "multimodal"

    def test_web_keywords_route_to_web(self):
        assert rule_based_route("What is the current price of gold?", None, None) == "web"
        assert rule_based_route("Search online for the latest news", None, None) == "web"
        assert rule_based_route("Visit this website and tell me", None, None) == "web"

    def test_url_keyword_routes_to_web(self):
        assert rule_based_route("What does this URL contain?", None, None) == "web"

    def test_pdf_file_returns_none(self):
        # PDFs don't trigger a rule — LLM decides (usually text)
        result = rule_based_route("Summarize this document", "report.pdf", "application/pdf")
        assert result is None

    def test_generic_question_returns_none(self):
        result = rule_based_route("What is 42 times 17?", None, None)
        assert result is None
