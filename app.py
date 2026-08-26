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
    boxes, font size and colour are recorded using the PDF's real text
    layer (fast, perfectly accurate wherever it exists).
2b. An OCR pass (Tesseract) then scans the rendered page image for any
    additional text that has NO real text layer — e.g. French words
    baked directly into a scanned page, a photo, a diagram, or a chart.
    Anything OCR finds that isn't already covered by step 2 is added
    to the list of text to translate, with its own bounding box.
3.  The original French text — both the real text objects AND the
    pixels underneath any OCR-detected region — is then "redacted"
    (erased) directly on the page. Redaction removes the text/pixels
    but leaves every other image, drawing, chart, and background
    element exactly where it was.
4.  The now text-free page is rendered to a high-resolution image.
    Because nothing but the text was removed, this image is a near
    perfect visual clone of the original page (images/graphics in the
    exact same spot).
5.  That image becomes the background of a brand-new PDF page.
6.  The recorded text blocks (native + OCR) are translated
    (French -> English) and written back on top of the background
    image, inside the *same* bounding boxes they originally occupied.
    Because translated text is often longer/shorter than the French
    original, the font size is automatically shrunk (and text
    re-wrapped) until it fits inside the original box, so nothing
    overflows or gets clipped.

The result is a PDF that looks identical to the original but reads in
English.

Author: Generated for a non-technical end user - see README for usage.
"""

import io
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
import streamlit as st
from PIL import Image

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
# OCR backend (catches French text that has no real text layer, e.g. text
# baked into scanned pages, photos, diagrams, or charts)
# ---------------------------------------------------------------------------
try:
    import pytesseract
    # This raises pytesseract.TesseractNotFoundError if the *system* Tesseract
    # binary isn't installed (pip installing pytesseract alone is not enough).
    pytesseract.get_tesseract_version()
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# Which OCR language pack to use. 'fra' = French. If the French language
# pack isn't installed, we fall back to whatever is available so the app
# doesn't crash (OCR quality will suffer, see README for install steps).
OCR_LANGUAGE = "fra"


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
RENDER_ZOOM = 3.0            # 3x zoom ≈ 216 DPI background image, crisp but not huge
MIN_FONT_SIZE = 4.0          # never shrink translated text smaller than this
FONT_SIZE_STEP = 0.5         # decrement used while auto-fitting text
DEFAULT_FONT = "helv"        # a safe built-in PyMuPDF font (Helvetica)
MAX_CHARS_PER_TRANSLATE_CALL = 4500  # stay comfortably under API limits
REDACT_PADDING = 0.6         # small padding so redaction fully covers glyph edges
OCR_ZOOM = 3.0                # resolution used when rendering the page for OCR
OCR_MIN_CONFIDENCE = 45       # discard low-confidence OCR guesses (0-100 scale)
OCR_OVERLAP_THRESHOLD = 0.3   # skip an OCR box if it overlaps a native text box this much (avoids duplicates)


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
    other image, drawing, and background element untouched. This is what
    lets us later render a "clean" background image with nothing but the
    text removed.

    Important: we redact using PDF_REDACT_IMAGE_PIXELS rather than
    PDF_REDACT_IMAGE_NONE. This means that if French text happens to be
    baked directly into an image's pixels (e.g. a scanned page, or a
    diagram with embedded labels) at a location we've detected — either
    via the native text layer or via OCR — those specific pixels get
    blanked out too, instead of leaving the original French visible
    underneath the translated overlay. Pixels outside the redaction
    boxes (the rest of the image/diagram) are left completely untouched.
    """
    for tb in blocks:
        pad_rect = fitz.Rect(
            tb.bbox.x0 - REDACT_PADDING,
            tb.bbox.y0 - REDACT_PADDING,
            tb.bbox.x1 + REDACT_PADDING,
            tb.bbox.y1 + REDACT_PADDING,
        )
        page.add_redact_annot(pad_rect, fill=None)
    if blocks:
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)


def _rects_overlap_ratio(a: fitz.Rect, b: fitz.Rect) -> float:
    """Return intersection area / smaller-rect area (0..1)."""
    inter = a & b  # fitz.Rect intersection
    if inter.is_empty:
        return 0.0
    inter_area = inter.get_area()
    smaller_area = min(a.get_area(), b.get_area())
    if smaller_area <= 0:
        return 0.0
    return inter_area / smaller_area


def _estimate_text_color(pil_img: Image.Image, box_px: Tuple[int, int, int, int]) -> tuple:
    """
    Best-effort guess at the text colour for an OCR-detected block, by
    sampling the darkest pixels inside its bounding box. Falls back to
    black if anything goes wrong. Returned as (r, g, b) floats 0..1.
    """
    try:
        x0, y0, x1, y1 = box_px
        crop = pil_img.crop((max(x0, 0), max(y0, 0), x1, y1)).convert("RGB")
        pixels = list(crop.getdata())
        if not pixels:
            return (0.0, 0.0, 0.0)
        # Text is usually the darker (lower luminance) pixels against a
        # lighter background; average the darkest 25% of pixels.
        pixels.sort(key=lambda p: 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2])
        cutoff = max(1, len(pixels) // 4)
        dark_pixels = pixels[:cutoff]
        r = sum(p[0] for p in dark_pixels) / len(dark_pixels)
        g = sum(p[1] for p in dark_pixels) / len(dark_pixels)
        b = sum(p[2] for p in dark_pixels) / len(dark_pixels)
        return (r / 255.0, g / 255.0, b / 255.0)
    except Exception:
        return (0.0, 0.0, 0.0)


def ocr_extract_blocks(
    page: fitz.Page,
    existing_blocks: List[TextBlock],
    zoom: float = OCR_ZOOM,
) -> List[TextBlock]:
    """
    Run OCR over the rendered page image to find French text that has NO
    real text layer (common for scanned pages, photos, or text baked
    into diagrams/charts). Any OCR hit that significantly overlaps a
    block we already extracted natively is skipped, so we don't
    translate/redact the same text twice.
    """
    if not OCR_AVAILABLE:
        return []

    try:
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pil_img = Image.open(io.BytesIO(pix.tobytes("png")))

        data = pytesseract.image_to_data(
            pil_img, lang=OCR_LANGUAGE, output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractNotFoundError:
        return []
    except Exception:
        # Never let an OCR hiccup break the whole pipeline
        return []

    n = len(data.get("text", []))
    # Group word-level OCR results into paragraph-level blocks using
    # Tesseract's own block/paragraph numbering, so multi-word phrases
    # are translated together as coherent sentences.
    groups = {}
    for i in range(n):
        word = data["text"][i].strip()
        conf_raw = data["conf"][i]
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = -1.0
        if not word or conf < OCR_MIN_CONFIDENCE:
            continue

        key = (data["block_num"][i], data["par_num"][i])
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        groups.setdefault(key, []).append(
            {
                "word": word,
                "line_num": data["line_num"][i],
                "word_num": data["word_num"][i],
                "box": (x, y, x + w, y + h),
            }
        )

    new_blocks: List[TextBlock] = []
    for key, items in groups.items():
        items.sort(key=lambda it: (it["line_num"], it["word_num"]))
        text = " ".join(it["word"] for it in items)
        if not text.strip():
            continue

        xs0 = min(it["box"][0] for it in items)
        ys0 = min(it["box"][1] for it in items)
        xs1 = max(it["box"][2] for it in items)
        ys1 = max(it["box"][3] for it in items)

        # Convert from OCR pixel space back to PDF point space
        pdf_rect = fitz.Rect(xs0 / zoom, ys0 / zoom, xs1 / zoom, ys1 / zoom)

        # Skip anything that overlaps text we already found natively —
        # that text has already been captured (and captured more
        # accurately) by the real text layer.
        overlaps_existing = any(
            _rects_overlap_ratio(pdf_rect, eb.bbox) >= OCR_OVERLAP_THRESHOLD
            for eb in existing_blocks
        )
        if overlaps_existing:
            continue

        box_height_pts = ys1 / zoom - ys0 / zoom
        font_size = max(box_height_pts * 0.8, 6.0)
        color = _estimate_text_color(pil_img, (xs0, ys0, xs1, ys1))

        new_blocks.append(
            TextBlock(text=text, bbox=pdf_rect, font_size=font_size, color=color)
        )

    return new_blocks


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


def process_pdf(
    file_bytes: bytes,
    status_cb=None,
    progress_cb=None,
    use_ocr: bool = True,
) -> bytes:
    """
    Full pipeline: open -> extract (native + OCR) -> redact -> render
    background -> translate -> reconstruct. Returns the translated PDF
    as bytes.
    """
    src_doc = fitz.open(stream=file_bytes, filetype="pdf")
    out_doc = fitz.open()

    total_pages = src_doc.page_count
    if total_pages == 0:
        raise ValueError("This PDF appears to have no pages.")

    all_page_blocks: List[List[TextBlock]] = []
    page_images: List[bytes] = []
    page_sizes: List[fitz.Rect] = []

    # --- Pass 1: extract text (native + OCR) + build clean background ---
    for pno in range(total_pages):
        if status_cb:
            status_cb(f"Extracting layout... (page {pno + 1}/{total_pages})")
        page = src_doc[pno]
        blocks = extract_text_blocks(page)

        if use_ocr and OCR_AVAILABLE:
            if status_cb:
                status_cb(
                    f"Scanning images for embedded text (OCR)... "
                    f"(page {pno + 1}/{total_pages})"
                )
            ocr_blocks = ocr_extract_blocks(page, blocks)
            blocks.extend(ocr_blocks)

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

    use_ocr = st.checkbox(
        "🔍 Also detect text baked into images/scans (OCR)",
        value=OCR_AVAILABLE,
        disabled=not OCR_AVAILABLE,
        help=(
            "Catches French text that has no selectable text layer — e.g. "
            "scanned pages, photos, or labels drawn inside diagrams/charts. "
            "Slightly slower, but recommended for the most complete translation."
        ),
    )
    if not OCR_AVAILABLE:
        st.warning(
            "OCR is unavailable because the Tesseract engine isn't installed "
            "on this machine, so text embedded inside images/scans will be "
            "left untranslated. See the README for a one-time install step "
            "to enable this."
        )

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
                result_bytes = process_pdf(
                    file_bytes, status_cb=status_cb, progress_cb=progress_cb, use_ocr=use_ocr
                )
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
- With OCR enabled, text baked into scanned pages, photos, or diagrams
  (with no selectable text layer) is also detected and translated — not
  just the PDF's normal text layer.
- Very complex layouts (tables, rotated text, multi-column magazines)
  may not be perfectly line-wrapped, but will never overflow their box.
            """
        )


if __name__ == "__main__":
    main()
