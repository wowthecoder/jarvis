"""Helpers for encoding files as base64 for multimodal LLM input."""
import base64
import mimetypes
import os
from typing import Optional


# File extensions to MIME type mappings
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def detect_file_type(file_path: str) -> Optional[str]:
    """Detect MIME type from file extension. Returns None if unknown."""
    if not file_path:
        return None
    ext = os.path.splitext(file_path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return mimetypes.types_map.get(ext, "image/jpeg")
    if ext in AUDIO_EXTENSIONS:
        return mimetypes.types_map.get(ext, "audio/mpeg")
    if ext in VIDEO_EXTENSIONS:
        return mimetypes.types_map.get(ext, "video/mp4")
    # Try system mime detection as fallback
    mime, _ = mimetypes.guess_type(file_path)
    return mime


def encode_file_base64(file_path: str) -> str:
    """Read a file and return its base64-encoded content."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_multimodal_content(question: str, file_path: str, mime_type: str) -> list:
    """Build a LangChain HumanMessage content list for multimodal input.

    Returns a list suitable for use as HumanMessage(content=...).
    Gemini accepts: text, image_url (base64 data URI), and media (audio/video).
    """
    b64_data = encode_file_base64(file_path)
    content: list = [{"type": "text", "text": question}]

    if mime_type.startswith("image/"):
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
        })
    elif mime_type.startswith("audio/") or mime_type.startswith("video/"):
        content.append({
            "type": "media",
            "data": b64_data,
            "mime_type": mime_type,
        })

    return content


def is_multimodal_file(file_path: Optional[str]) -> bool:
    """Return True if the file requires multimodal handling (image/audio/video)."""
    if not file_path:
        return False
    ext = os.path.splitext(file_path)[1].lower()
    return ext in IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
