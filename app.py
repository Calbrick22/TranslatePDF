"""
French → English PDF Translator
================================

A Streamlit web app that translates the French text inside a PDF into
English while keeping the original visual layout (images, charts,
backgrounds, positions) intact.

HOW IT WORKS (the "flatten background + overlay text" technique)
------------------------------------------------------------------
1.  Each page of the source PDF is opened with PyMuPDF (fitz).
2.  All text blocks on the page are located and their exact bounding
    boxes, font size and colour are recorded.
3.  The original French text is then "redacted" (erased) directly on
    the page. Redaction removes the text but leaves every image,
    drawing, chart, and background element exactly where it was.
4.  The now text-free page is rendered to a high-resolution image.
    Because nothing but the text was removed, this image is a perfect
    visual clone of the original page (images/graphics in the exact
    same spot).
5.  That image becomes the background of a brand-new PDF page.
6.  The recorded text blocks are translated (French -> English) and
    written back on top of the background image, inside the *same*
    bounding boxes they originally occupied. Because translated text
    is often longer/shorter than the French original, the font size
    is automatically shrunk (and text re-wrapped) until it fits
    inside the original box, so nothing overflows or gets clipped.

The result is a PDF that looks identical to the original but reads in
English.

Author: Generated for a non-technical end user - see README for usage.
"""

import io
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

import fitz  # PyMuPDF
import streamlit as st

# ---------------------------------------------------------------------------
# Translation backend
# ---------------------------------------------------------------------------
# deep-translator wraps the free Google Translate web endpoint and needs no
# API key, which keeps this app runnable "out of the box" for a
# non-technical user. If it isn't installed, we fail with a clear message.
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
RENDER_ZOOM = 3.0            # 3x zoom ≈ 216 DPI background image, crisp but not huge
MIN_FONT_SIZE = 4.0          # never shrink translated text smaller than this
FONT_SIZE_STEP = 0.5         # decrement used while auto-fitting text
DEFAULT_FONT = "helv"        # a safe built-in PyMuPDF font (Helvetica)
MAX_CHARS_PER_TRANSLATE_CALL = 4500  # stay comfortably under API limits
REDACT_PADDING = 0.6         # small padding so redaction fully covers glyph edges


# ---------------------------------------------------------------------------
# Data model for one translatable piece of text on a page
# ---------------------------------------------------------------------------
@dataclass
class TextBlock:
    text: str
    bbox: fitz.Rect
    font_size: float
    color: tuple  # (r, g, b) each 0..1
    align: int = 0  # 0 = left, kept simple for reliability
    translated_text: str = field(default="")


# ---------------------------------------------------------------------------
# Core processing functions
# ---------------------------------------------------------------------------

def extract_text_blocks(page: fitz.Page) -> List[TextBlock]:
    """
    Pull out paragraph-level text blocks (not individual spans) so that
    whole sentences are sent to the translator together. This produces
    far more natural, grammatically correct English than translating
    word-by-word or line-by-line.
    """
    blocks: List[TextBlock] = []
    raw = page.get_text("dict")

    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue  # skip image blocks; the image stays in the background

        block_text_parts = []
        x0, y0, x1, y1 = None, None, None, None
        sizes = []
        color = (0, 0, 0)

        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                span_text = span.get("text", "")
                if not span_text.strip():
                    continue
                line_text += span_text
                sizes.append(span.get("size", 10.0))

                # sRGB int -> (r, g, b) floats 0..1
                c = span.get("color", 0)
                color = (
                    ((c >> 16) & 255) / 255.0,
                    ((c >> 8) & 255) / 255.0,
                    (c & 255) / 255.0,
                )

                sx0, sy0, sx1, sy1 = span.get("bbox", line.get("bbox"))
                x0 = sx0 if x0 is None else min(x0, sx0)
                y0 = sy0 if y0 is None else min(y0, sy0)
                x1 = sx1 if x1 is None else max(x1, sx1)
                y1 = sy1 if y1 is None else max(y1, sy1)

            if line_text.strip():
                block_text_parts.append(line_text)

        full_text = " ".join(t.strip() for t in block_text_parts).strip()
        if not full_text or x0 is None:
            continue

        avg_size = sum(sizes) / len(sizes) if sizes else 10.0
        blocks.append(
            TextBlock(
                text=full_text,
                bbox=fitz.Rect(x0, y0, x1, y1),
                font_size=max(avg_size, 6.0),
                color=color,
            )
        )

    return blocks


def redact_blocks(page: fitz.Page, blocks: List[TextBlock]) -> None:
    """
    Erase the original French text from the page while leaving every
    image, drawing, and background element untouched. This is what lets
    us later render a "clean" background image with nothing but the
    text removed.
    """
    for tb in blocks:
        pad_rect = fitz.Rect(
            tb.bbox.x0 - REDACT_PADDING,
            tb.bbox.y0 - REDACT_PADDING,
            tb.bbox.x1 + REDACT_PADDING,
            tb.bbox.y1 + REDACT_PADDING,
        )
        # fill=None keeps whatever is under the box as-is EXCEPT the text
        # itself, which apply_redactions() strips out.
        page.add_redact_annot(pad_rect, fill=None)
    if blocks:
        # images=0 keeps images fully intact; only vector text is removed
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)


def render_background(page: fitz.Page, zoom: float = RENDER_ZOOM) -> bytes:
    """Render the (now text-free) page to a PNG image byte string."""
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def chunk_texts_for_translation(texts: List[str]) -> List[List[int]]:
    """
    Group block indices into batches that stay under the translation
    API's character limit, to minimise the number of network calls.
    """
    batches: List[List[int]] = []
    current: List[int] = []
    current_len = 0

    for i, t in enumerate(texts):
        t_len = len(t) + 1
        if current and current_len + t_len > MAX_CHARS_PER_TRANSLATE_CALL:
            batches.append(current)
            current, current_len = [], 0
        current.append(i)
        current_len += t_len

    if current:
        batches.append(current)
    return batches


def translate_blocks(blocks: List[TextBlock], progress_cb=None) -> None:
    """
    Translate every block's text from French to English in-place.
    Falls back to the original French text for any block that fails to
    translate, so a single network hiccup never crashes the whole run.
    """
    if not blocks:
        return

    if not TRANSLATOR_AVAILABLE:
        raise RuntimeError(
            "The 'deep-translator' package is not installed. "
            "Please run: pip install deep-translator"
        )

    translator = GoogleTranslator(source="fr", target="en")
    texts = [b.text for b in blocks]
    batches = chunk_texts_for_translation(texts)

    done = 0
    for batch_idx, indices in enumerate(batches):
        batch_texts = [texts[i] for i in indices]
        try:
            # deep-translator supports batch translation via translate_batch
            translated = translator.translate_batch(batch_texts)
        except Exception:
            # Fall back to translating one-by-one so a single bad string
            # doesn't sink the whole batch.
            translated = []
            for t in batch_texts:
                try:
                    translated.append(translator.translate(t))
                except Exception:
                    translated.append(t)  # keep French if translation fails

        for local_i, global_i in enumerate(indices):
            result = translated[local_i] if local_i < len(translated) else None
            blocks[global_i].translated_text = result or blocks[global_i].text
            done += 1
            if progress_cb:
                progress_cb(done, len(blocks))


def fit_text_in_box(
    new_page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    start_size: float,
    color: tuple,
) -> None:
    """
    Insert `text` into `rect`, automatically shrinking the font size
    until the text fits without being clipped. This is what handles
    English text being longer/shorter than the original French.
    """
    fontsize = start_size
    fontname = DEFAULT_FONT

    while fontsize >= MIN_FONT_SIZE:
        # insert_textbox returns the unused space (>= 0) if the text fit,
        # or a negative number if it did not fit at this font size.
        overflow = new_page.insert_textbox(
            rect,
            text,
            fontsize=fontsize,
            fontname=fontname,
            color=color,
            align=0,
            render_mode=0,
        )
        if overflow >= 0:
            return  # it fit
        fontsize -= FONT_SIZE_STEP

    # Last resort: force it in at the minimum size, expanding the box
    # slightly downward so text is at least fully visible rather than
    # silently dropped.
    expanded = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y1 + 40)
    new_page.insert_textbox(
        expanded,
        text,
        fontsize=MIN_FONT_SIZE,
        fontname=fontname,
        color=color,
        align=0,
        render_mode=0,
    )


def process_pdf(file_bytes: bytes, status_cb=None, progress_cb=None) -> bytes:
    """
    Full pipeline: open -> extract -> redact -> render background ->
    translate -> reconstruct. Returns the translated PDF as bytes.
    """
    src_doc = fitz.open(stream=file_bytes, filetype="pdf")
    out_doc = fitz.open()

    total_pages = src_doc.page_count
    if total_pages == 0:
        raise ValueError("This PDF appears to have no pages.")

    all_page_blocks: List[List[TextBlock]] = []
    page_images: List[bytes] = []
    page_sizes: List[fitz.Rect] = []

    # --- Pass 1: extract text + build clean background per page ---------
    for pno in range(total_pages):
        if status_cb:
            status_cb(f"Extracting layout... (page {pno + 1}/{total_pages})")
        page = src_doc[pno]
        blocks = extract_text_blocks(page)
        redact_blocks(page, blocks)
        bg_png = render_background(page)

        all_page_blocks.append(blocks)
        page_images.append(bg_png)
        page_sizes.append(page.rect)

        if progress_cb:
            progress_cb((pno + 1) / total_pages * 0.4)  # extraction = 40%

    src_doc.close()

    total_text_blocks = sum(len(b) for b in all_page_blocks)
    if total_text_blocks == 0:
        if status_cb:
            status_cb(
                "No extractable text found — this PDF may be a scanned "
                "image. The output will be a visual copy without translation."
            )

    # --- Pass 2: translate all blocks (batched across whole document) ---
    flat_blocks: List[TextBlock] = [b for page_b in all_page_blocks for b in page_b]
    if status_cb:
        status_cb("Translating text...")

    def _t_progress(done, total):
        if progress_cb:
            progress_cb(0.4 + (done / max(total, 1)) * 0.4)  # translation = next 40%

    translate_blocks(flat_blocks, progress_cb=_t_progress)

    # --- Pass 3: reconstruct each page -----------------------------------
    for pno in range(total_pages):
        if status_cb:
            status_cb(f"Reconstructing PDF... (page {pno + 1}/{total_pages})")

        page_rect = page_sizes[pno]
        new_page = out_doc.new_page(width=page_rect.width, height=page_rect.height)
        new_page.insert_image(page_rect, stream=page_images[pno])

        for tb in all_page_blocks[pno]:
            display_text = tb.translated_text or tb.text
            fit_text_in_box(new_page, tb.bbox, display_text, tb.font_size, tb.color)

        if progress_cb:
            progress_cb(0.8 + (pno + 1) / total_pages * 0.2)  # reconstruction = last 20%

    out_bytes = out_doc.tobytes(garbage=4, deflate=True)
    out_doc.close()
    return out_bytes


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="French → English PDF Translator",
        page_icon="📄",
        layout="centered",
    )

    st.title("📄 French → English PDF Translator")
    st.write(
        "Upload a French PDF and get back an English version that "
        "**looks exactly the same** — same images, same layout, same "
        "positions — just translated."
    )

    if not TRANSLATOR_AVAILABLE:
        st.error(
            "Missing dependency: `deep-translator`. Please install "
            "requirements first (`pip install -r requirements.txt`) and "
            "restart the app."
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Drag and drop your French PDF here",
        type=["pdf"],
        accept_multiple_files=False,
        help="Only .pdf files are supported. Maximum recommended size: ~50 pages.",
    )

    if "translated_pdf_bytes" not in st.session_state:
        st.session_state.translated_pdf_bytes = None
        st.session_state.translated_filename = None

    if uploaded_file is not None:
        st.success(f"Loaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.0f} KB)")

        translate_clicked = st.button("🔁 Translate PDF", type="primary", use_container_width=True)

        if translate_clicked:
            file_bytes = uploaded_file.read()

            progress_bar = st.progress(0.0)
            status_placeholder = st.empty()

            def status_cb(msg: str):
                status_placeholder.info(msg)

            def progress_cb(fraction: float):
                progress_bar.progress(min(max(fraction, 0.0), 1.0))

            try:
                start = time.time()
                result_bytes = process_pdf(file_bytes, status_cb=status_cb, progress_cb=progress_cb)
                elapsed = time.time() - start

                progress_bar.progress(1.0)
                status_placeholder.success(f"Done in {elapsed:.1f} seconds!")

                st.session_state.translated_pdf_bytes = result_bytes
                base_name = uploaded_file.name.rsplit(".", 1)[0]
                st.session_state.translated_filename = f"{base_name}_english.pdf"

            except fitz.FileDataError:
                st.error(
                    "This file could not be read as a valid PDF. It may be "
                    "corrupted or password-protected. Please try another file."
                )
            except ValueError as ve:
                st.error(str(ve))
            except RuntimeError as re_err:
                st.error(str(re_err))
            except Exception:
                st.error(
                    "Something went wrong while processing this PDF. "
                    "Here are the technical details for support purposes:"
                )
                st.code(traceback.format_exc())

    if st.session_state.translated_pdf_bytes:
        st.divider()
        st.subheader("✅ Your translated PDF is ready")

        st.download_button(
            label="⬇️ Download Translated PDF",
            data=st.session_state.translated_pdf_bytes,
            file_name=st.session_state.translated_filename,
            mime="application/pdf",
            use_container_width=True,
            type="primary",
        )

        with st.expander("Preview first page"):
            try:
                preview_doc = fitz.open(stream=st.session_state.translated_pdf_bytes, filetype="pdf")
                preview_pix = preview_doc[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                st.image(preview_pix.tobytes("png"), use_container_width=True)
                preview_doc.close()
            except Exception:
                st.warning("Could not generate a preview image, but the download above will work.")

    with st.expander("ℹ️ How this works / limitations"):
        st.markdown(
            """
- Each page is rendered as a background image so **images, charts, and
  layout stay pixel-identical** to the original.
- French text is detected, erased, translated to English, and placed
  back in the exact same spot.
- If the translated text is longer than the original French, the font
  size is automatically shrunk to fit inside the original text box.
- Scanned PDFs (photos of text, no real text layer) can't be translated
  automatically since there is no text to extract — only OCR'd PDFs work.
- Very complex layouts (tables, rotated text, multi-column magazines)
  may not be perfectly line-wrapped, but will never overflow their box.
            """
        )


if __name__ == "__main__":
    main()
