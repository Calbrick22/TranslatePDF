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
import re

import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
import requests

from rowfilter import (
    TableRow, scan_document, summarise_countries, build_rows,
    assign_countries, build_page_unit, page_looks_tabular, UNKNOWN_LABEL,
)
from rowbuild import assign_blocks_to_rows, build_filtered_document
import officeconv
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
# CRITICAL: force a network timeout on the translation library.
#
# deep-translator calls requests.get() with NO timeout argument. If the
# connection stalls (common when the free endpoint throttles you), that
# call blocks *forever*, the worker thread never returns, and the whole
# app appears frozen with no progress and no error.
#
# We wrap the requests module used inside deep_translator so every call
# gets a timeout whether the library asked for one or not.
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT_SECONDS = 20

if TRANSLATOR_AVAILABLE:
    try:
        import deep_translator.google as _dt_google

        class _TimeoutRequests:
            """Thin proxy that injects a default timeout into every call."""

            def __init__(self, real_requests):
                self._real = real_requests

            def get(self, *args, **kwargs):
                kwargs.setdefault("timeout", REQUEST_TIMEOUT_SECONDS)
                return self._real.get(*args, **kwargs)

            def post(self, *args, **kwargs):
                kwargs.setdefault("timeout", REQUEST_TIMEOUT_SECONDS)
                return self._real.post(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._real, name)

        _dt_google.requests = _TimeoutRequests(_dt_google.requests)
    except Exception:
        # If the library's internals change, we still work - just without
        # the enforced timeout.
        pass

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
MAX_CHARS_PER_TRANSLATE_CALL = 4000  # character budget for one batched request
MAX_ITEMS_PER_BATCH = 20             # how many text blocks to pack into one request
TRANSLATE_WORKERS = 6                # parallel requests in flight
BATCH_DELIMITER = "@@@"              # marker used to split a batched response
TRANSLATE_DEADLINE_SECONDS = 180     # hard cap on the whole translation stage
REPAIR_BATCH_SIZE = 10               # small batches when retrying failed items
REDACT_PADDING = 0.6         # small padding so redaction fully covers glyph edges
OCR_ZOOM = 3.0                # resolution used when rendering the page for OCR
OCR_MIN_CONFIDENCE = 65       # discard low-confidence OCR guesses (0-100 scale); high threshold to avoid noise
OCR_OVERLAP_THRESHOLD = 0.25  # skip an OCR box if it overlaps a native text box this much (avoids duplicates)


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


# ---------------------------------------------------------------------------
# Country-of-origin row filtering
# ---------------------------------------------------------------------------
# Lets the user keep only table rows whose country-of-origin matches their
# selection. Removal is structural: the whole row band (text AND images) is
# excluded from the rebuilt document, not merely blanked.
#
# Detection strategy: this document has no vector gridlines (borders are part
# of a raster image), so rows are located from the country column instead.
# Each row is anchored by the item-name cell sitting immediately above its
# country cell in the same narrow left-hand column.
# ---------------------------------------------------------------------------

COUNTRY_VOCAB = {
    "Russia / CIS": ["CEI", "URSS", "USSR", "RUSSIE", "RUSSIA"],
    "Czech Republic": ["TCHEQUE", "CZECH", "TCHECOSLOVAQUIE"],
    "United Kingdom": ["GB", "GRANDE BRETAGNE", "UNITED KINGDOM", "ROYAUME UNI", "BRITAIN"],
    "China": ["CHINE", "CHINA"],
    "Italy": ["ITALIE", "ITALY"],
    "Yugoslavia": ["YOUGOSLAVIE", "YUGOSLAVIA"],
    "Germany": ["ALLEMAGNE", "GERMANY", "RDA", "RFA"],
    "USA": ["USA", "ETATS UNIS", "UNITED STATES"],
    "Pakistan": ["PAKISTAN"],
    "Spain": ["ESPAGNE", "SPAIN"],
    "Belgium": ["BELGIQUE", "BELGIUM"],
    "Romania": ["ROUMANIE", "ROMANIA"],
    "South Africa": ["AFRIQUE DU SUD", "SOUTH AFRICA", "AFRIQUE", "AFRICA"],
    "Albania": ["ALBANIE", "ALBANIA"],
    "France": ["FRANCE"],
    "Egypt": ["EGYPTE", "EGYPT"],
    "Israel": ["ISRAEL"],
    "Sweden": ["SUEDE", "SWEDEN"],
    "Portugal": ["PORTUGAL"],
    "Hungary": ["HONGRIE", "HUNGARY"],
    "Bulgaria": ["BULGARIE", "BULGARIA"],
    "Poland": ["POLOGNE", "POLAND"],
    "Austria": ["AUTRICHE", "AUSTRIA"],
    "Switzerland": ["SUISSE", "SWITZERLAND"],
    "Vietnam": ["VIETNAM"],
    "India": ["INDE", "INDIA"],
    "Iran": ["IRAN"],
    "Iraq": ["IRAK", "IRAQ"],
    "Turkey": ["TURQUIE", "TURKEY"],
    "Greece": ["GRECE", "GREECE"],
}

# Fraction of page width searched for the country column, and max cell width.
COUNTRY_COL_MAX_X_FRAC = 0.35
COUNTRY_CELL_MAX_WIDTH = 140
ROW_TOP_PADDING = 6.0


@dataclass
class TableRow:
    """One logical table row on a page."""
    page_index: int
    top: float
    bottom: float
    country: str
    raw_text: str

    @property
    def height(self) -> float:
        return max(self.bottom - self.top, 0.0)


def _normalise_for_country(s: str) -> str:
    s = s.upper()
    for a, b in (("È", "E"), ("É", "E"), ("Ê", "E"), ("Ë", "E"),
                 ("Ç", "C"), ("Ô", "O"), ("Î", "I"), ("À", "A")):
        s = s.replace(a, b)
    return re.sub(r"[^A-Z ]+", " ", s)


def match_country(text: str) -> Optional[str]:
    """Return the canonical country name found in `text`, or None."""
    n = _normalise_for_country(text)
    best, best_len = None, 0
    for canon, aliases in COUNTRY_VOCAB.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", n) and len(alias) > best_len:
                best, best_len = canon, len(alias)
    return best


def _plain_blocks(page: fitz.Page):
    out = []
    for b in page.get_text("dict").get("blocks", []):
        if b.get("type") != 0:
            continue
        t = " ".join(
            "".join(s["text"] for s in l.get("spans", []))
            for l in b.get("lines", [])
        ).strip()
        if t:
            out.append((fitz.Rect(b["bbox"]), t))
    return out


def detect_table_rows(page: fitz.Page, page_index: int) -> List[TableRow]:
    """
    Locate table rows on a page via the country column.

    Returns rows in reading order. `top`/`bottom` describe the full-width
    band belonging to that row, so callers can slice it out wholesale
    (including any images inside it).
    """
    blocks = _plain_blocks(page)
    max_x = page.rect.width * COUNTRY_COL_MAX_X_FRAC

    hits = []
    for r, t in blocks:
        if r.x0 < max_x and r.width < COUNTRY_CELL_MAX_WIDTH:
            c = match_country(t)
            if c:
                hits.append([r, t, c])
    hits.sort(key=lambda h: h[0].y0)

    # Merge a country cell split across consecutive lines
    merged = []
    for h in hits:
        if merged and h[0].y0 - merged[-1][0].y1 < 14 and merged[-1][2] == h[2]:
            merged[-1][0] = merged[-1][0] | h[0]
            merged[-1][1] = merged[-1][1] + " " + h[1]
        else:
            merged.append(list(h))

    anchors = []
    for r, t, c in merged:
        # The item-name cell is the block IMMEDIATELY above the country cell.
        # Take the nearest (largest y1), not the topmost - otherwise we skip
        # past it onto the previous row or the page header.
        above = [
            rr for rr, _ in blocks
            if rr.y1 <= r.y0 + 2 and rr.y0 > r.y0 - 70
            and rr.x0 < max_x and rr.width < COUNTRY_CELL_MAX_WIDTH
        ]
        top = max(above, key=lambda rr: rr.y1).y0 if above else r.y0
        anchors.append((min(top, r.y0), c, t))

    anchors.sort(key=lambda a: a[0])

    rows: List[TableRow] = []
    for i, (top, country, raw) in enumerate(anchors):
        band_top = max(top - ROW_TOP_PADDING, 0.0)
        if i + 1 < len(anchors):
            band_bottom = max(anchors[i + 1][0] - ROW_TOP_PADDING, band_top)
        else:
            band_bottom = page.rect.height
        rows.append(TableRow(page_index, band_top, band_bottom, country, raw))
    return rows


def scan_document_countries(file_bytes: bytes) -> Tuple[dict, List[TableRow]]:
    """
    Scan the PDF and report which countries of origin appear, with row counts.

    Returns (counts_by_country, all_rows).
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    all_rows: List[TableRow] = []
    try:
        for pno in range(doc.page_count):
            all_rows.extend(detect_table_rows(doc[pno], pno))
    finally:
        doc.close()

    counts: dict = {}
    for r in all_rows:
        counts[r.country] = counts.get(r.country, 0) + 1
    return counts, all_rows


def header_band_for_page(page: fitz.Page, rows: List[TableRow]) -> Optional[fitz.Rect]:
    """
    The strip above the first row - i.e. the column header - so filtered
    output keeps its headings instead of showing bare rows.
    """
    page_rows = [r for r in rows if r.page_index == page.number]
    if not page_rows:
        return None
    first_top = min(r.top for r in page_rows)
    if first_top <= 2:
        return None
    return fitz.Rect(0, 0, page.rect.width, first_top)

def extract_text_lines(page: fitz.Page) -> List[TextBlock]:
    """
    Extract text at LINE granularity rather than block granularity.

    PyMuPDF sometimes reports an entire table as a single text block. When
    filtering by row that's fatal: one block spanning many rows gets
    assigned to whichever row contains its midpoint, so every other row
    comes out empty and that one row gets all the text piled into it.

    Working line-by-line gives fine-grained pieces that can be assigned to
    the correct row and then merged back into paragraphs *within* that row.
    """
    lines: List[TextBlock] = []
    raw = page.get_text("dict")

    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            text_parts = []
            sizes = []
            color = (0, 0, 0)
            x0 = y0 = x1 = y1 = None

            for span in line.get("spans", []):
                span_text = span.get("text", "")
                if not span_text.strip():
                    continue
                text_parts.append(span_text)
                sizes.append(span.get("size", 10.0))

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

            text = "".join(text_parts).strip()
            if not text or x0 is None:
                continue

            lines.append(
                TextBlock(
                    text=text,
                    bbox=fitz.Rect(x0, y0, x1, y1),
                    font_size=max(sum(sizes) / len(sizes) if sizes else 10.0, 6.0),
                    color=color,
                )
            )

    return lines


def merge_within_rows(blocks_by_row: dict) -> dict:
    """
    Merge line fragments into paragraphs, but only among lines that share a
    row. This keeps translation context without ever gluing one table row's
    text onto the next one's.
    """
    return {
        key: merge_blocks_into_paragraphs(blist)
        for key, blist in blocks_by_row.items()
    }


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


def merge_nearby_blocks(
    blocks: List[TextBlock],
    y_gap_factor: float = 1.4,
    min_x_overlap_frac: float = 0.5,
) -> List[TextBlock]:
    """
    Some PDFs (often ones exported from tables/spreadsheets, or with
    generous line spacing) store every single LINE of a paragraph as its
    own separate text block, instead of one block per paragraph. If we
    translated each of those tiny fragments independently, we'd lose all
    sentence context AND multiply the number of translation API calls
    (which risks rate-limiting on large documents).

    This groups blocks that are vertically stacked close together and
    horizontally aligned (i.e. clearly the same paragraph/column) into a
    single merged block covering their combined area, in reading order.
    Blocks in different table columns/cells won't merge because their
    x-ranges won't overlap enough.
    """
    if not blocks:
        return blocks

    remaining = sorted(blocks, key=lambda b: (b.bbox.y0, b.bbox.x0))
    used = [False] * len(remaining)
    merged: List[TextBlock] = []

    for i, seed in enumerate(remaining):
        if used[i]:
            continue
        cluster = [seed]
        used[i] = True

        grew = True
        while grew:
            grew = False
            cx0 = min(c.bbox.x0 for c in cluster)
            cx1 = max(c.bbox.x1 for c in cluster)
            cy1 = max(c.bbox.y1 for c in cluster)

            for j, cand in enumerate(remaining):
                if used[j]:
                    continue
                gap = cand.bbox.y0 - cy1
                if gap < -1:  # already vertically overlapping the cluster
                    gap = 0
                if gap > cand.bbox.height * y_gap_factor:
                    continue  # too far below to be the same paragraph

                ox0 = max(cx0, cand.bbox.x0)
                ox1 = min(cx1, cand.bbox.x1)
                overlap = max(0.0, ox1 - ox0)
                min_width = min(cx1 - cx0, cand.bbox.x1 - cand.bbox.x0)
                overlap_frac = (overlap / min_width) if min_width > 0 else 0.0

                if overlap_frac >= min_x_overlap_frac:
                    cluster.append(cand)
                    used[j] = True
                    grew = True

        cluster.sort(key=lambda c: (c.bbox.y0, c.bbox.x0))
        text = " ".join(c.text for c in cluster)
        x0 = min(c.bbox.x0 for c in cluster)
        y0 = min(c.bbox.y0 for c in cluster)
        x1 = max(c.bbox.x1 for c in cluster)
        y1 = max(c.bbox.y1 for c in cluster)
        font_size = max(c.font_size for c in cluster)
        color = cluster[0].color

        merged.append(
            TextBlock(text=text, bbox=fitz.Rect(x0, y0, x1, y1), font_size=font_size, color=color)
        )

    return merged


def redact_blocks(page: fitz.Page, blocks: List[TextBlock], redact_image_pixels: bool = False) -> None:
    """
    Erase the original French text from the page while leaving every
    other image, drawing, and background element untouched. This is what
    lets us later render a "clean" background image with nothing but the
    text removed.

    By default (redact_image_pixels=False), we use PDF_REDACT_IMAGE_NONE,
    which only removes the PDF's text layer, leaving images/diagrams
    completely intact. This preserves visual quality perfectly for PDFs
    with clean native text.

    If redact_image_pixels=True (only when OCR is enabled), we use
    PDF_REDACT_IMAGE_PIXELS instead, which also blanks pixels under each
    text box. This catches French text baked into image pixels (scanned
    pages, diagram labels) but at the cost of potentially destroying
    fine details in diagrams. Only use this if you've explicitly enabled OCR.
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
        image_mode = fitz.PDF_REDACT_IMAGE_PIXELS if redact_image_pixels else fitz.PDF_REDACT_IMAGE_NONE
        page.apply_redactions(images=image_mode)


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


def merge_blocks_into_paragraphs(blocks: List[TextBlock]) -> List[TextBlock]:
    """
    PyMuPDF often reports each visual LINE as its own separate block,
    especially inside table cells. Translating line-by-line is bad for two
    reasons: it produces poor, context-free translations ("Deformation"
    split from "of the shoulders"), and it multiplies the number of
    network requests by ~2-3x, which triggers rate limiting.

    This merges blocks that clearly belong to the same paragraph/cell:
    same horizontal column (bboxes overlap on X) and vertically adjacent
    (gap smaller than roughly one line height). The merged block keeps the
    union bounding box, so it still renders in exactly the right place.
    """
    if not blocks:
        return []

    # Sort into reading order: top-to-bottom, then left-to-right.
    ordered = sorted(blocks, key=lambda b: (round(b.bbox.y0, 1), round(b.bbox.x0, 1)))

    merged: List[TextBlock] = []
    current = ordered[0]

    for nxt in ordered[1:]:
        # Horizontal overlap ratio between the two boxes
        overlap_x = min(current.bbox.x1, nxt.bbox.x1) - max(current.bbox.x0, nxt.bbox.x0)
        min_width = max(min(current.bbox.width, nxt.bbox.width), 1.0)
        same_column = (overlap_x / min_width) > 0.75

        # Left edges should roughly line up — real paragraphs are aligned.
        # This stops a wide header being glued to a narrow cell below it.
        left_aligned = abs(current.bbox.x0 - nxt.bbox.x0) <= max(current.font_size * 1.5, 8.0)

        # Widths should be comparable, so we don't merge a full-width
        # banner row into a narrow table cell.
        w_ratio = min(current.bbox.width, nxt.bbox.width) / max(current.bbox.width, nxt.bbox.width, 1.0)
        similar_width = w_ratio > 0.6

        # Vertical gap between bottom of current and top of next
        vertical_gap = nxt.bbox.y0 - current.bbox.y1
        line_height = max(current.font_size, 6.0)
        adjacent = -line_height * 0.5 <= vertical_gap <= line_height * 1.2

        # Similar font size => same logical paragraph (avoids merging a
        # heading into the body text below it)
        similar_size = abs(current.font_size - nxt.font_size) <= max(current.font_size * 0.25, 0.8)

        if same_column and left_aligned and similar_width and adjacent and similar_size:
            current = TextBlock(
                text=(current.text.rstrip() + " " + nxt.text.lstrip()).strip(),
                bbox=current.bbox | nxt.bbox,  # union of the two rects
                font_size=min(current.font_size, nxt.font_size),
                color=current.color,
            )
        else:
            merged.append(current)
            current = nxt

    merged.append(current)
    return merged


# ---------------------------------------------------------------------------
# Translation providers
# ---------------------------------------------------------------------------
# Two backends are supported:
#
#   DeepL (recommended) - needs a free API key, but is reliable, fast, has
#       NATIVE batch support (up to 50 texts per request, returned aligned
#       by index) and generally better French->English quality.
#
#   Google (no key) - convenient for a quick try, but it is an unofficial
#       scraped endpoint that aggressively rate-limits and stalls. This is
#       what caused the freezing.
#
# Both implement: translate_many(texts) -> List[Optional[str]], returning a
# list the SAME length as the input, with None for anything that failed.
# ---------------------------------------------------------------------------

DEEPL_FREE_URL = "https://api-free.deepl.com/v2/translate"
DEEPL_PRO_URL = "https://api.deepl.com/v2/translate"
DEEPL_MAX_TEXTS_PER_REQUEST = 50


class DeepLProvider:
    """
    Direct DeepL API client (current v2 spec).

    Authentication uses the `Authorization: DeepL-Auth-Key <key>` header.
    DeepL removed the older `auth_key` body/query parameter in November 2025;
    requests using it now fail with 403 regardless of how valid the key is.

    Uses DeepL's native multi-text batching: up to 50 strings per request,
    returned as an ordered list. Because the API preserves order and count,
    there is no delimiter to be mangled and no risk of a translation landing
    in the wrong box.
    """

    name = "DeepL"
    max_batch_items = DEEPL_MAX_TEXTS_PER_REQUEST
    max_batch_chars = 100_000  # DeepL caps the request body at 128 KiB

    def __init__(self, api_key: str, source: str = "FR", target: str = "EN-GB"):
        self.api_key = api_key.strip()
        self.source = source
        self.target = target
        # DeepL free keys conventionally end in ":fx"
        self.url = DEEPL_FREE_URL if self.api_key.endswith(":fx") else DEEPL_PRO_URL

    def _headers(self) -> dict:
        return {
            "Authorization": f"DeepL-Auth-Key {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _extract_api_message(resp) -> str:
        """Pull DeepL's own error text out of the response, if present."""
        try:
            body = resp.json()
            msg = body.get("message") or body.get("detail")
            return f" DeepL said: {msg}" if msg else ""
        except Exception:
            return ""

    def _raise_for_status(self, resp) -> None:
        detail = self._extract_api_message(resp)

        if resp.status_code in (401, 403):
            hint = ""
            if not self.api_key.endswith(":fx") and "free" in self.url:
                hint = ""
            raise RuntimeError(
                "DeepL rejected the API key (HTTP "
                f"{resp.status_code}).{detail}\n\n"
                "Things to check:\n"
                "• The key is copied in full, with no missing characters or "
                "trailing spaces.\n"
                "• Free keys end in ':fx' and use the free endpoint; keys "
                "without that suffix are treated as Pro.\n"
                "• The DeepL account is activated — new accounts sometimes "
                "need email/identity verification before the API works."
                + hint
            )
        if resp.status_code == 456:
            raise RuntimeError(
                "This DeepL key has used up its monthly character quota."
                f"{detail} The free tier allows 500,000 characters per month."
            )
        if resp.status_code == 400:
            raise RuntimeError(
                f"DeepL rejected the request (HTTP 400).{detail}"
            )

    def check_key(self) -> str:
        """
        Verify the key with one tiny translation. Raises RuntimeError with a
        readable explanation if the key is bad, otherwise returns a sample.
        """
        out = self.translate_many(["Bonjour"], _raise_on_http_error=True)
        if not out or not out[0]:
            raise RuntimeError(
                "DeepL accepted the key but returned no translation. "
                "Please try again in a moment."
            )
        return out[0]

    def translate_many(
        self, texts: List[str], _raise_on_http_error: bool = True
    ) -> List[Optional[str]]:
        if not texts:
            return []

        payload = {
            "text": texts,
            "source_lang": self.source,
            "target_lang": self.target,
        }

        try:
            resp = requests.post(
                self.url,
                headers=self._headers(),
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except Exception:
            return [None] * len(texts)

        # Auth/quota problems must reach the user rather than silently
        # producing an all-French document.
        if _raise_on_http_error:
            self._raise_for_status(resp)

        if resp.status_code != 200:
            return [None] * len(texts)

        try:
            payload = resp.json()
            out = [item.get("text") for item in payload.get("translations", [])]
        except Exception:
            return [None] * len(texts)

        # Alignment guarantee: refuse anything that doesn't match exactly.
        if len(out) != len(texts):
            return [None] * len(texts)

        return [o if (o and o.strip()) else None for o in out]


class GoogleProvider:
    """
    Free Google endpoint via deep-translator.

    No native batching, so we emulate it by joining texts with a delimiter
    and splitting the response. If the split doesn't line up exactly, the
    whole batch is discarded and the caller retries those items solo.
    """

    name = "Google (free, no key)"
    max_batch_items = MAX_ITEMS_PER_BATCH
    max_batch_chars = MAX_CHARS_PER_TRANSLATE_CALL

    def __init__(self, source: str = "fr", target: str = "en"):
        if not TRANSLATOR_AVAILABLE:
            raise RuntimeError(
                "The 'deep-translator' package is not installed. "
                "Please run: pip install deep-translator"
            )
        self._translator = GoogleTranslator(source=source, target=target)

    def _translate_single(self, text: str) -> Optional[str]:
        try:
            result = self._translator.translate(text)
            return result if (result and result.strip()) else None
        except Exception:
            return None

    def translate_many(self, texts: List[str]) -> List[Optional[str]]:
        if not texts:
            return []
        if len(texts) == 1:
            return [self._translate_single(texts[0])]

        joined = f"\n{BATCH_DELIMITER}\n".join(texts)
        raw = self._translate_single(joined)
        if not raw:
            return [None] * len(texts)

        parts = re.split(rf"\s*{re.escape(BATCH_DELIMITER)}\s*", raw)
        if len(parts) != len(texts):
            return [None] * len(texts)

        return [p.strip() if p.strip() else None for p in parts]


def _build_batches(texts: List[str], provider) -> List[List[int]]:
    """
    Group indices into batches sized for the provider.

    DeepL accepts up to 50 texts natively per request, so we can pack more
    in. Google's scraped endpoint has a 5000-character input cap and needs
    the delimiter trick, so we stay conservative there.
    """
    max_items = getattr(provider, "max_batch_items", MAX_ITEMS_PER_BATCH)
    max_chars = getattr(provider, "max_batch_chars", MAX_CHARS_PER_TRANSLATE_CALL)

    batches: List[List[int]] = []
    current: List[int] = []
    current_len = 0

    for i, t in enumerate(texts):
        t_len = len(t) + len(BATCH_DELIMITER) + 2
        too_long = current_len + t_len > max_chars
        too_many = len(current) >= max_items

        if current and (too_long or too_many):
            batches.append(current)
            current, current_len = [], 0

        current.append(i)
        current_len += t_len

    if current:
        batches.append(current)
    return batches


def translate_blocks(blocks, provider, progress_cb=None) -> dict:
    """
    Translate every block from French to English in-place.

    The design priority here is *never freezing*. Every stage checks a shared
    deadline, so this always returns within TRANSLATE_DEADLINE_SECONDS no
    matter how badly the network misbehaves.

    Returns {"translated": n, "failed": n, "cached": n, "timed_out": bool}.
    """
    stats = {"translated": 0, "failed": 0, "cached": 0, "timed_out": False}
    if not blocks:
        return stats

    deadline = time.monotonic() + TRANSLATE_DEADLINE_SECONDS

    def out_of_time() -> bool:
        return time.monotonic() >= deadline

    # --- De-duplicate ----------------------------------------------------
    unique_texts = []
    seen = {}
    for b in blocks:
        key = b.text.strip()
        if key not in seen:
            seen[key] = len(unique_texts)
            unique_texts.append(key)

    results = [None] * len(unique_texts)

    # --- Build batches ---------------------------------------------------
    batches = _build_batches(unique_texts, provider)
    total_units = max(len(batches), 1)
    completed = 0

    def run_batch(batch_indices):
        """Worker thread. Never touches Streamlit."""
        if out_of_time():
            return
        batch_texts = [unique_texts[i] for i in batch_indices]
        out = provider.translate_many(batch_texts)
        for local_i, global_i in enumerate(batch_indices):
            results[global_i] = out[local_i] if local_i < len(out) else None

    # --- Pass 1: batched + parallel --------------------------------------
    # Progress is reported from THIS (main) thread. Streamlit calls made
    # from worker threads silently do nothing, which is why the bar
    # previously appeared frozen.
    #
    # We deliberately avoid `with ThreadPoolExecutor(...)`: its __exit__
    # calls shutdown(wait=True), which blocks on hung network threads and
    # would freeze the UI despite the deadline.
    pool = ThreadPoolExecutor(max_workers=TRANSLATE_WORKERS)
    try:
        futures = [pool.submit(run_batch, b) for b in batches]
        remaining = max(deadline - time.monotonic(), 0.1)
        try:
            for fut in as_completed(futures, timeout=remaining):
                try:
                    fut.result()
                except Exception:
                    pass
                completed += 1
                if progress_cb:
                    progress_cb(completed, total_units)
        except FuturesTimeoutError:
            stats["timed_out"] = True
    finally:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            pool.shutdown(wait=False)

    # --- Pass 2: repair leftovers, STRICTLY time-boxed -------------------
    # This is the path that froze the app. Previously every failed item got
    # 4 solo retries at a 20s timeout, so ONE bad batch of 20 items could
    # block for ~27 minutes showing no progress. Now repair works in small
    # batches, checks the deadline before each one, and stops cleanly.
    missing = [i for i, r in enumerate(results) if r is None]
    if missing and not out_of_time():
        repair_batches = [
            missing[i:i + REPAIR_BATCH_SIZE]
            for i in range(0, len(missing), REPAIR_BATCH_SIZE)
        ]
        for rb in repair_batches:
            if out_of_time():
                stats["timed_out"] = True
                break
            try:
                out = provider.translate_many([unique_texts[i] for i in rb])
            except RuntimeError:
                raise  # API key / quota errors must reach the user
            except Exception:
                out = [None] * len(rb)
            for local_i, global_i in enumerate(rb):
                if local_i < len(out) and out[local_i]:
                    results[global_i] = out[local_i]

    # --- Apply results ---------------------------------------------------
    for b in blocks:
        value = results[seen[b.text.strip()]]
        if value:
            b.translated_text = value
            stats["translated"] += 1
        else:
            b.translated_text = b.text  # honest fallback, counted
            stats["failed"] += 1

    stats["cached"] = max(len(blocks) - len(unique_texts), 0)
    return stats


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


def scan_pdf_countries(file_bytes: bytes, zoom: float = 2.0, granularity: str = "auto"):
    """
    Pass 1 for the filtering workflow: detect table rows and their
    countries of origin, without translating anything.

    Returns (rows, summary) where summary is [(country, row_count), ...].
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        rows = scan_document(doc, zoom=zoom, granularity=granularity)
        summary = summarise_countries(rows)
    finally:
        doc.close()
    return rows, summary


def process_pdf_filtered(
    file_bytes: bytes,
    selected_countries: List[str],
    provider,
    status_cb=None,
    progress_cb=None,
    use_ocr: bool = False,
    render_zoom: float = RENDER_ZOOM,
    granularity: str = "auto",
) -> Tuple[bytes, dict]:
    """
    Translate and rebuild the PDF keeping ONLY rows whose country of origin
    is in `selected_countries`.

    Dropped rows are removed whole - their images and diagrams go with
    them, because each row travels as a rendered pixel strip. Kept rows are
    stacked under a repeated header so the output closes up rather than
    leaving gaps.

    Only text from kept rows is sent for translation, so filtering also
    reduces API usage.
    """
    src = fitz.open(stream=file_bytes, filetype="pdf")
    total_pages = src.page_count
    if total_pages == 0:
        raise ValueError("This PDF appears to have no pages.")

    wanted = set(selected_countries)

    all_rows: List[TableRow] = []
    row_blocks: dict = {}
    page_backgrounds: List[Image.Image] = []
    page_rects: List[fitz.Rect] = []
    keep_keys: List[Tuple[int, int]] = []
    country_col = None

    # --- Pass 1: per page, detect rows, keep/drop, build clean background --
    for pno in range(total_pages):
        if status_cb:
            status_cb(f"Analysing table rows... (page {pno + 1}/{total_pages})")

        page = src[pno]

        if granularity == "pages":
            tabular = False
        elif granularity == "rows":
            tabular = True
        else:
            tabular = page_looks_tabular(page, 2.0)

        if tabular:
            rows, v_lines = build_rows(page, pno, zoom=2.0)
            country_col = assign_countries(page, rows, v_lines, country_col) or country_col
        else:
            rows = build_page_unit(page, pno)
            assign_countries(page, rows, [], None)

        # Line-level extraction, so a block spanning several table rows
        # can't dump all its text into one row.
        blocks = extract_text_lines(page)
        if use_ocr and OCR_AVAILABLE:
            blocks.extend(ocr_extract_blocks(page, blocks))

        mapping = assign_blocks_to_rows(blocks, rows)
        # Merge into paragraphs only within each row, never across rows.
        mapping = merge_within_rows(mapping)
        for ridx, blist in mapping.items():
            row_blocks[(pno, ridx)] = blist

        # Redaction needs the full flat list for this page
        blocks = [b for blist in mapping.values() for b in blist]

        for ridx, r in enumerate(rows):
            if not r.is_header and r.country in wanted:
                keep_keys.append((pno, ridx))

        # Erase original text, then render the page as a clean background.
        redact_blocks(page, blocks, redact_image_pixels=False)
        pix = page.get_pixmap(matrix=fitz.Matrix(render_zoom, render_zoom), alpha=False)
        page_backgrounds.append(Image.open(io.BytesIO(pix.tobytes("png"))))
        page_rects.append(fitz.Rect(page.rect))

        all_rows.extend(rows)
        if progress_cb:
            progress_cb((pno + 1) / total_pages * 0.35)

    src.close()

    if not keep_keys:
        raise ValueError(
            "None of the selected countries matched any rows in this document."
        )

    # --- Pass 2: translate ONLY the blocks in kept rows --------------------
    # Headers are kept too so the rebuilt table stays readable.
    header_keys = [
        (r.page_no, _row_index_on_page(r, all_rows))
        for r in all_rows if r.is_header
    ]
    translate_keys = set(keep_keys) | set(header_keys)

    flat_blocks = []
    for key in translate_keys:
        flat_blocks.extend(row_blocks.get(key, []))

    if status_cb:
        status_cb(f"Translating {len(flat_blocks)} text blocks from kept rows...")

    def _tp(done, total):
        if progress_cb:
            progress_cb(0.35 + (done / max(total, 1)) * 0.45)

    tstats = translate_blocks(flat_blocks, provider, progress_cb=_tp)

    # --- Pass 3: rebuild, compacting kept rows -----------------------------
    if status_cb:
        status_cb("Rebuilding filtered PDF...")

    def draw_text(new_page, block, dy):
        shifted = fitz.Rect(
            block.bbox.x0, block.bbox.y0 + dy,
            block.bbox.x1, block.bbox.y1 + dy,
        )
        fit_text_in_box(
            new_page, shifted,
            block.translated_text or block.text,
            block.font_size, block.color,
        )

    out_doc = build_filtered_document(
        page_backgrounds=page_backgrounds,
        page_rects=page_rects,
        rows=all_rows,
        row_blocks=row_blocks,
        keep_row_keys=keep_keys,
        zoom=render_zoom,
        draw_text=draw_text,
    )

    if progress_cb:
        progress_cb(1.0)

    out_bytes = out_doc.tobytes(garbage=4, deflate=True)
    out_doc.close()

    tstats["rows_kept"] = len(keep_keys)
    tstats["rows_total"] = len([r for r in all_rows if not r.is_header])
    return out_bytes, tstats


def _row_index_on_page(row: TableRow, rows: List[TableRow]) -> int:
    i = 0
    for r in rows:
        if r.page_no == row.page_no:
            if r is row:
                return i
            i += 1
    return 0


def process_pdf(
    file_bytes: bytes,
    status_cb=None,
    progress_cb=None,
    use_ocr: bool = True,
    render_zoom: float = RENDER_ZOOM,
    provider=None,
    keep_countries: Optional[set] = None,
) -> Tuple[bytes, dict]:
    """
    Full pipeline: open -> extract (native + OCR) -> merge into paragraphs
    -> redact -> render background -> translate -> reconstruct.

    Returns (pdf_bytes, translation_stats).
    """
    src_doc = fitz.open(stream=file_bytes, filetype="pdf")
    out_doc = fitz.open()

    total_pages = src_doc.page_count
    if total_pages == 0:
        raise ValueError("This PDF appears to have no pages.")

    all_page_blocks: List[List[TextBlock]] = []
    page_images: List[bytes] = []
    page_sizes: List[fitz.Rect] = []
    page_kept_bands: List[Optional[List[fitz.Rect]]] = []
    any_ocr_blocks_found = False  # track if we found any OCR text across all pages

    # --- Pass 1: extract text (native + OCR) + build clean background ---
    for pno in range(total_pages):
        if status_cb:
            status_cb(f"Extracting layout... (page {pno + 1}/{total_pages})")
        page = src_doc[pno]

        # Country filtering: work out which row bands to keep on this page.
        # Detection must happen BEFORE redaction, while text is intact.
        if keep_countries is not None:
            rows_here = detect_table_rows(page, pno)
            kept_bands = [
                fitz.Rect(0, r.top, page.rect.width, r.bottom)
                for r in rows_here if r.country in keep_countries
            ]
            hdr = header_band_for_page(page, rows_here)
            if hdr is not None and kept_bands:
                kept_bands.insert(0, hdr)
            page_kept_bands.append(kept_bands)
        else:
            page_kept_bands.append(None)

        blocks = extract_text_blocks(page)
        ocr_blocks: List[TextBlock] = []

        if use_ocr and OCR_AVAILABLE:
            if status_cb:
                status_cb(
                    f"Scanning images for embedded text (OCR)... "
                    f"(page {pno + 1}/{total_pages})"
                )
            ocr_blocks = ocr_extract_blocks(page, blocks)
            if ocr_blocks:
                any_ocr_blocks_found = True
            blocks.extend(ocr_blocks)

        # Merge line-fragments into coherent paragraphs. This improves
        # translation quality AND cuts the number of network requests,
        # which is the main cause of text being left untranslated.
        blocks = merge_blocks_into_paragraphs(blocks)

        # Discard text belonging to excluded rows: it must not be translated
        # (wasted quota) and must not appear in the output.
        bands = page_kept_bands[pno]
        if bands is not None:
            blocks = [
                b for b in blocks
                if any(_rect_mostly_inside(b.bbox, band) for band in bands)
            ]

        # Only use pixel redaction if we actually found OCR blocks AND OCR is enabled.
        # For clean native-text PDFs, use non-pixel redaction to preserve diagram quality.
        redact_blocks(page, blocks, redact_image_pixels=(any_ocr_blocks_found and use_ocr))
        bg_png = render_background(page, zoom=render_zoom)

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

    translate_stats = translate_blocks(flat_blocks, provider, progress_cb=_t_progress)

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
    return out_bytes, translate_stats


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="French → English PDF Translator",
        page_icon="📄",
        layout="centered",
    )

    st.title("📄 French → English Document Translator")
    st.write(
        "Upload a French **PDF, Word or PowerPoint** file and get back an "
        "English PDF that "
        "**looks exactly the same** — same images, same layout, same "
        "positions — just translated."
    )

    # Note: only the Google engine needs deep-translator. DeepL talks to its
    # API directly, so a missing deep-translator must not block the app.

    use_ocr = st.checkbox(
        "🔍 Also detect text in images/scans (OCR) — only enable if needed",
        value=False,
        disabled=not OCR_AVAILABLE,
        help=(
            "Enable only for PDFs with scanned pages or text baked into diagrams. "
            "OCR can introduce artifacts and is slower. For normal PDFs with "
            "clean selectable text, leave this OFF for best quality."
        ),
    )
    if not OCR_AVAILABLE:
        st.info(
            "ℹ️ OCR is unavailable — the Tesseract engine isn't installed. "
            "The app will still work perfectly for PDFs with native text layers. "
            "See the README to install Tesseract if you need OCR for scanned documents."
        )

    engine = st.radio(
        "Translation engine",
        ["DeepL (recommended — needs a free key)", "Google (no key, less reliable)"],
        index=0,
        help=(
            "DeepL is faster, more accurate, and doesn't stall. Google needs "
            "no setup but uses an unofficial endpoint that frequently "
            "rate-limits and can leave text untranslated."
        ),
    )
    use_deepl = engine.startswith("DeepL")

    deepl_key = ""
    if use_deepl:
        deepl_key = st.text_input(
            "DeepL API key",
            type="password",
            placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:fx",
            help="Your key is used only for this translation and is never stored.",
        )
        if deepl_key:
            if st.button("🔑 Test this key", use_container_width=True):
                try:
                    sample = DeepLProvider(deepl_key).check_key()
                    st.success(f"Key works. \"Bonjour\" → \"{sample}\"")
                except RuntimeError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Could not reach DeepL: {e}")

        with st.expander("How to get a free DeepL key (2 minutes)"):
            st.markdown(
                """
1. Go to **https://www.deepl.com/pro-api** and choose **DeepL API Free**.
2. Sign up. It asks for a card to verify identity but the free tier is
   **not charged** — it covers 500,000 characters per month.
3. Open **Account → API Keys** and copy your key. Free keys end in `:fx`.
4. Paste it into the box above.

500,000 characters is roughly 200–250 pages of a document like yours per month.
                """
            )
        if not deepl_key:
            st.info("Enter your DeepL key above, or switch to Google to try without one.")

    quality = st.select_slider(
        "Output quality",
        options=["Small file (faster)", "Balanced", "High detail (larger)"],
        value="Balanced",
        help=(
            "Controls the resolution of the page background. 'Balanced' is "
            "usually indistinguishable from the original on screen and "
            "produces a much smaller file, which matters on mobile."
        ),
    )
    zoom_map = {
        "Small file (faster)": 1.5,
        "Balanced": 2.0,
        "High detail (larger)": 3.0,
    }
    render_zoom = zoom_map[quality]

    st.caption(
        "📱 On a phone, keep this screen open while it works — switching apps "
        "can disconnect the page and cancel the job. Most documents now finish "
        "in well under a minute."
    )

    uploaded_file = st.file_uploader(
        "Drag and drop your French PDF here",
        type=["pdf", "docx", "doc", "pptx", "ppt", "odt", "odp", "rtf"],
        accept_multiple_files=False,
        help=(
            "PDF, Word (.docx/.doc) and PowerPoint (.pptx/.ppt) are supported. "
            "Office files are converted to PDF automatically. The output is "
            "always a PDF."
        ),
    )

    if "translated_pdf_bytes" not in st.session_state:
        st.session_state.translated_pdf_bytes = None
        st.session_state.translated_filename = None
    if "country_summary" not in st.session_state:
        st.session_state.country_summary = None
        st.session_state.scanned_filename = None

    if uploaded_file is not None:
        needs_conversion = not uploaded_file.name.lower().endswith(".pdf")

        if needs_conversion and not officeconv.office_available():
            st.error(
                "This is a "
                f"{officeconv.describe(uploaded_file.name)}, which needs "
                "LibreOffice installed to convert it to PDF. See the README "
                "for the one-time install step, or save the file as a PDF "
                "yourself and upload that instead."
            )
            st.stop()

        st.success(
            f"Loaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.0f} KB)"
            + (f" — {officeconv.describe(uploaded_file.name)}, will be converted to PDF"
               if needs_conversion else "")
        )

        # ---------------------------------------------------------------
        # Optional filtering by country of origin
        # ---------------------------------------------------------------
        filter_mode = st.checkbox(
            "🌍 Only include certain countries of origin",
            value=False,
            help=(
                "Scans the document's table for a country column, then lets you "
                "keep only the rows you want. Dropped rows are removed entirely, "
                "including their photos and diagrams."
            ),
        )

        selected_countries: List[str] = []

        if filter_mode:
            # Re-scan if a different file was uploaded
            if st.session_state.scanned_filename != uploaded_file.name:
                st.session_state.country_summary = None

            if st.session_state.country_summary is None:
                if st.button("🔎 Scan for countries", use_container_width=True):
                    with st.spinner("Scanning table rows..."):
                        try:
                            pdf_for_scan, _ = officeconv.prepare_pdf_bytes(
                                uploaded_file.getvalue(), uploaded_file.name)
                            _, summary = scan_pdf_countries(pdf_for_scan)
                            st.session_state.country_summary = summary
                            st.session_state.scanned_filename = uploaded_file.name
                        except Exception as e:
                            st.error(f"Could not analyse the table: {e}")

            summary = st.session_state.country_summary
            if summary:
                found = [c for c, _ in summary if c != UNKNOWN_LABEL]
                unknown_n = dict(summary).get(UNKNOWN_LABEL, 0)

                if not found:
                    st.warning(
                        "No countries were recognised in this document. It may not "
                        "have a country column, or the wording may be unfamiliar. "
                        "Uncheck the filter to translate the whole file."
                    )
                else:
                    st.caption(
                        f"Found {len(found)} countries across "
                        f"{sum(n for c, n in summary if c != UNKNOWN_LABEL)} rows."
                    )
                    selected_countries = st.multiselect(
                        "Countries to include",
                        options=[c for c, _ in summary if c != UNKNOWN_LABEL],
                        default=[],
                        format_func=lambda c: f"{c}  ({dict(summary)[c]} rows)",
                    )
                    if unknown_n:
                        if st.checkbox(
                            f"Also include {unknown_n} rows with no identified country"
                        ):
                            selected_countries = selected_countries + [UNKNOWN_LABEL]

                    if selected_countries:
                        kept = sum(dict(summary).get(c, 0) for c in selected_countries)
                        st.info(f"**{kept} rows** will be kept; the rest will be removed.")
                    else:
                        st.warning("Select at least one country to continue.")

        button_label = (
            "🔁 Translate selected rows" if filter_mode else "🔁 Translate PDF"
        )
        translate_clicked = st.button(
            button_label, type="primary", use_container_width=True,
            disabled=(filter_mode and not selected_countries),
        )

        if translate_clicked:
            file_bytes = uploaded_file.getvalue()
            if needs_conversion:
                try:
                    file_bytes, _ = officeconv.prepare_pdf_bytes(
                        file_bytes, uploaded_file.name
                    )
                except RuntimeError as conv_err:
                    st.error(str(conv_err))
                    st.stop()

            progress_bar = st.progress(0.0)
            status_placeholder = st.empty()

            def status_cb(msg: str):
                status_placeholder.info(msg)

            def progress_cb(fraction: float):
                progress_bar.progress(min(max(fraction, 0.0), 1.0))

            try:
                start = time.time()

                # Build the chosen provider
                if use_deepl:
                    if not deepl_key:
                        st.error("Please enter your DeepL API key first.")
                        st.stop()
                    provider = DeepLProvider(deepl_key)
                else:
                    provider = GoogleProvider()

                result_bytes, tstats = (
                    process_pdf_filtered(
                        file_bytes,
                        selected_countries=selected_countries,
                        provider=provider,
                        status_cb=status_cb,
                        progress_cb=progress_cb,
                        use_ocr=use_ocr,
                        render_zoom=render_zoom,
                    )
                    if (filter_mode and selected_countries)
                    else process_pdf(
                        file_bytes,
                        status_cb=status_cb,
                        progress_cb=progress_cb,
                        use_ocr=use_ocr,
                        render_zoom=render_zoom,
                        provider=provider,
                    )
                )
                elapsed = time.time() - start

                progress_bar.progress(1.0)

                failed = tstats.get("failed", 0)
                ok = tstats.get("translated", 0) + tstats.get("cached", 0)
                timed_out = tstats.get("timed_out", False)

                if timed_out:
                    status_placeholder.error(
                        f"⏱️ Translation timed out after "
                        f"{TRANSLATE_DEADLINE_SECONDS // 60} minutes. "
                        f"{ok} blocks were translated; {failed} were left in French. "
                        "The translation service is likely rate-limiting or unreachable "
                        "right now. Your partial PDF is still available below — "
                        "wait a few minutes and try again."
                    )
                elif failed == 0:
                    rows_note = ""
                    if "rows_kept" in tstats:
                        rows_note = (
                            f" Kept {tstats['rows_kept']} of "
                            f"{tstats['rows_total']} rows."
                        )
                    status_placeholder.success(
                        f"Done in {elapsed:.0f} seconds — all {ok} text blocks "
                        f"translated.{rows_note}"
                    )
                else:
                    status_placeholder.warning(
                        f"Done in {elapsed:.0f} seconds — {ok} blocks translated, "
                        f"but **{failed} could not be translated** and were left in French. "
                        "This is usually temporary rate-limiting. "
                        "Wait a minute and run it again."
                    )

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

    with st.expander("ℹ️ How this works / when to use OCR"):
        st.markdown(
            """
**Standard mode (OCR OFF — recommended for most PDFs):**
- Extracts text from the PDF's native text layer (fast, high quality).
- Renders each page as a background image so images, charts, and layout
  stay perfectly intact.
- French text is translated and placed back in the exact same spot.
- Automatically shrinks font size if English is longer than French.

**OCR mode (for scanned documents):**
- Enables Tesseract to also scan rendered page images for text with no
  text layer (scanned pages, handwritten annotations, diagram labels).
- Slower and OCR can introduce errors, especially on complex layouts.
- Only enable if your PDF has significant scanned content or embedded
  image text. For normal digital PDFs, OCR will only add noise.

**Tradeoffs:**
- Complex layouts (tables, multi-column text, rotated elements) may not
  wrap perfectly, but will never overflow their box.
- Text smaller than ~8pt may be difficult for OCR to read accurately.
- If OCR finds false positives (noise artifacts), they will be translated
  and may appear in the output.
            """
        )


if __name__ == "__main__":
    main()
