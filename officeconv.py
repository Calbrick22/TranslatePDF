"""
Office document -> PDF conversion.
===================================

Word (.docx/.doc) and PowerPoint (.pptx/.ppt) files are converted to PDF
with LibreOffice in headless mode, then fed through the existing PDF
pipeline. Doing it this way means layout, images, tables and slide
backgrounds are rendered by a real office engine rather than something
hand-rolled, so fidelity is as good as opening the file and printing it.

LibreOffice is an external program, not a Python package - see README for
install instructions per platform.
"""

import os
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple

# Extensions we can convert, mapped to a friendly label
CONVERTIBLE = {
    ".docx": "Word document",
    ".doc": "Word document",
    ".pptx": "PowerPoint presentation",
    ".ppt": "PowerPoint presentation",
    ".odt": "OpenDocument text",
    ".odp": "OpenDocument presentation",
    ".rtf": "Rich text document",
}

PRESENTATION_EXTS = {".pptx", ".ppt", ".odp"}

CONVERSION_TIMEOUT = 180  # seconds


def find_soffice() -> Optional[str]:
    """Locate the LibreOffice binary across platforms."""
    for name in ("soffice", "libreoffice"):
        path = shutil.which(name)
        if path:
            return path

    # Common install locations that aren't always on PATH
    candidates = [
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",       # macOS
        r"C:\Program Files\LibreOffice\program\soffice.exe",          # Windows
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        "/usr/lib/libreoffice/program/soffice",                       # Linux
        "/snap/bin/libreoffice",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def office_available() -> bool:
    return find_soffice() is not None


def is_convertible(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in CONVERTIBLE


def is_presentation(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in PRESENTATION_EXTS


def describe(filename: str) -> str:
    return CONVERTIBLE.get(os.path.splitext(filename.lower())[1], "document")


def convert_to_pdf(file_bytes: bytes, filename: str) -> bytes:
    """
    Convert an office file to PDF bytes.

    Each conversion runs in its own temporary directory. That isolation
    matters: LibreOffice names output purely after the input's stem, so
    converting two files with the same stem into a shared folder silently
    overwrites one with the other.

    Raises RuntimeError with a readable message on any failure.
    """
    soffice = find_soffice()
    if not soffice:
        raise RuntimeError(
            "LibreOffice is required to open Word and PowerPoint files, but it "
            "wasn't found on this machine. See the README for the one-time "
            "install step, or convert the file to PDF yourself first."
        )

    ext = os.path.splitext(filename.lower())[1]
    if ext not in CONVERTIBLE:
        raise RuntimeError(f"Unsupported file type: {ext}")

    with tempfile.TemporaryDirectory() as tmpdir:
        in_dir = os.path.join(tmpdir, "in")
        out_dir = os.path.join(tmpdir, "out")
        profile_dir = os.path.join(tmpdir, "profile")
        os.makedirs(in_dir)
        os.makedirs(out_dir)

        # Sanitise the name; LibreOffice can choke on odd characters, and the
        # stem determines the output filename.
        safe_stem = "document"
        in_path = os.path.join(in_dir, safe_stem + ext)
        with open(in_path, "wb") as f:
            f.write(file_bytes)

        cmd = [
            soffice,
            "--headless",
            "--norestore",
            "--nolockcheck",
            "--nodefault",
            "--nofirststartwizard",
            # A private profile avoids clashing with a desktop LibreOffice
            # that may already be running, which otherwise makes the
            # headless call exit immediately without converting.
            f"-env:UserInstallation=file://{profile_dir}",
            "--convert-to", "pdf",
            "--outdir", out_dir,
            in_path,
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=CONVERSION_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "Converting this file took too long and was stopped. It may be "
                "very large or unusually complex. Try saving it as a PDF "
                "yourself and uploading that instead."
            )

        produced = os.path.join(out_dir, safe_stem + ".pdf")
        if not os.path.exists(produced):
            stderr = (proc.stderr or b"").decode("utf-8", "replace").strip()
            detail = f"\n\nLibreOffice said: {stderr[:400]}" if stderr else ""
            raise RuntimeError(
                "The file could not be converted to PDF. It may be corrupt, "
                "password-protected, or in an unexpected format." + detail
            )

        with open(produced, "rb") as f:
            data = f.read()

    if not data:
        raise RuntimeError("Conversion produced an empty PDF.")
    return data


def prepare_pdf_bytes(file_bytes: bytes, filename: str) -> Tuple[bytes, bool]:
    """
    Return (pdf_bytes, was_converted).

    PDFs pass through untouched; office formats are converted.
    """
    if filename.lower().endswith(".pdf"):
        return file_bytes, False
    return convert_to_pdf(file_bytes, filename), True
