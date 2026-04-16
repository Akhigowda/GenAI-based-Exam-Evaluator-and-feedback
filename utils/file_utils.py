"""
Module: utils/file_utils.py
Responsibility: Handle temporary file storage for Streamlit uploads.
                Save uploaded bytes to disk, clean up temp files.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional


SUPPORTED_DOC_TYPES = [".pdf", ".pptx", ".txt", ".md"]
SUPPORTED_IMAGE_TYPES = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]


def save_uploaded_file(uploaded_file, dest_dir: Optional[str] = None) -> str:
    """
    Save a Streamlit UploadedFile object to disk.

    Args:
        uploaded_file: Streamlit's UploadedFile object.
        dest_dir: Directory to save to. Uses system temp dir if None.

    Returns:
        Absolute path to the saved file.
    """
    if dest_dir is None:
        dest_dir = tempfile.gettempdir()

    dest_path = Path(dest_dir) / uploaded_file.name
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(str(dest_path), "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(dest_path)


def get_file_extension(filename: str) -> str:
    """Return lowercase file extension with dot. E.g. '.pdf'"""
    return Path(filename).suffix.lower()


def is_document_file(filename: str) -> bool:
    """True if the file is a supported document type."""
    return get_file_extension(filename) in SUPPORTED_DOC_TYPES


def is_image_file(filename: str) -> bool:
    """True if the file is a supported image type."""
    return get_file_extension(filename) in SUPPORTED_IMAGE_TYPES


def cleanup_file(file_path: str) -> None:
    """Delete a temporary file if it exists."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass  # Non-critical cleanup
