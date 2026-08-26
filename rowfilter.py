"""
Table row detection + country-of-origin filtering.
===================================================

Lets the user keep only the table rows whose country-of-origin matches a
chosen set, dropping everything else - text AND images - from the output.

How it works
------------
1.  Each page is rendered and scanned for the table's ruling lines by
    looking for long runs of dark pixels. This works whether the borders
    are real vector lines or baked into a scanned image, which matters
    because many real-world PDFs are flattened.
2.  Horizontal lines give row boundaries; vertical lines give columns.
3.  The column most consistently containing a recognised country name is
    picked automatically as the "country" column.
4.  Each row is tagged with its country. Rows are then kept or dropped as
    a whole - the row's pixel strip carries its images and diagrams with
    it, so dropping a row removes its pictures too.
5.  Kept rows are stacked onto fresh pages beneath a repeated header, so
    the output closes up rather than leaving blank gaps.
"""

import io
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import fitz
import numpy as np
from PIL import Image

# Countries seen in this family of documents, with French/English variants
# and common OCR-mangled forms. Each entry maps aliases -> canonical label.
COUNTRY_ALIASES = {
    "CEI / USSR": ["cei", "urss", "ussr", "russie", "russia"],
    "Czech Republic": ["tcheque", "tchèque", "tchecoslovaquie", "czech"],
    "USA": ["usa", "etats-unis", "états-unis", "united states"],
    "UK": ["gb", "grande-bretagne", "royaume-uni", "united kingdom", "uk"],
    "France": ["france", "francais", "français"],
    "Germany": ["allemagne", "germany", "rfa", "rda"],
    "Italy": ["italie", "italy", "italian"],
    "Spain": ["espagne", "spain", "espagnol"],
    "Belgium": ["belgique", "belgium"],
    "Romania": ["roumanie", "romania"],
    "Yugoslavia": ["yougoslavie", "yugoslavia", "yougoslave"],
    "Albania": ["albanie", "albania"],
    "China": ["chine", "china", "chinese"],
    "Pakistan": ["pakistan"],
    "South Africa": ["afrique du sud", "africa sud", "afrique", "south africa", "sud"],
    "Egypt": ["egypte", "égypte", "egypt"],
    "Israel": ["israel", "israël"],
    "Sweden": ["suede", "suède", "sweden"],
    "Austria": ["autriche", "austria"],
    "Switzerland": ["suisse", "switzerland"],
    "Portugal": ["portugal"],
    "Hungary": ["hongrie", "hungary"],
    "Poland": ["pologne", "poland"],
    "Bulgaria": ["bulgarie", "bulgaria"],
    "Vietnam": ["vietnam", "viet nam"],
    "India": ["inde", "india"],
    "Iran": ["iran"],
    "Iraq": ["irak", "iraq"],
}

UNKNOWN_LABEL = "(no country identified)"

# Pixel-scan tuning
LINE_DARKNESS = 128      # below this grey value counts as "ink"
LINE_COVERAGE = 0.5      # fraction of the width/height that must be ink
MIN_ROW_HEIGHT_PTS = 18  # ignore bands thinner than this


@dataclass
class TableRow:
    """One horizontal band of the table on one page."""
    page_no: int
    y0: float
    y1: float
    country: str = UNKNOWN_LABEL
    is_header: bool = False
    block_indices: List[int] = field(default_factory=list)

    @property
    def height(self) -> float:
        return self.y1 - self.y0


def _render_grey(page: fitz.Page, zoom: float) -> np.ndarray:
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
    return np.array(img)


def _find_line_positions(mask_fraction: np.ndarray, zoom: float) -> List[float]:
    """Group consecutive high-coverage indices into single line positions."""
    hits = [i for i, f in enumerate(mask_fraction) if f > LINE_COVERAGE]
    groups: List[List[int]] = []
    for i in hits:
        if groups and i - groups[-1][-1] <= 3:
            groups[-1].append(i)
        else:
            groups.append([i])
    # Use the centre of each band, converted back to PDF points
    return [((g[0] + g[-1]) / 2.0) / zoom for g in groups]


def detect_grid(page: fitz.Page, zoom: float = 2.0) -> Tuple[List[float], List[float]]:
    """
    Return (horizontal_line_ys, vertical_line_xs) in PDF points.

    Detected from pixels rather than vector drawings, so this works on
    flattened/scanned pages as well as native ones.
    """
    try:
        arr = _render_grey(page, zoom)
    except Exception:
        return [], []

    ink = arr < LINE_DARKNESS
    h_lines = _find_line_positions(ink.mean(axis=1), zoom)
    v_lines = _find_line_positions(ink.mean(axis=0), zoom)
    return h_lines, v_lines


def build_rows(page: fitz.Page, page_no: int, zoom: float = 2.0) -> Tuple[List[TableRow], List[float]]:
    """
    Slice the page into rows using detected horizontal lines.

    Returns (rows, vertical_line_xs). If no usable table is found, returns
    a single row covering the whole page so the page passes through intact.
    """
    h_lines, v_lines = detect_grid(page, zoom)
    h_lines = sorted(h_lines)

    if len(h_lines) < 2:
        whole = TableRow(page_no=page_no, y0=page.rect.y0, y1=page.rect.y1)
        return [whole], v_lines

    rows: List[TableRow] = []
    for a, b in zip(h_lines, h_lines[1:]):
        if b - a >= MIN_ROW_HEIGHT_PTS:
            rows.append(TableRow(page_no=page_no, y0=a, y1=b))

    if not rows:
        return [TableRow(page_no=page_no, y0=page.rect.y0, y1=page.rect.y1)], v_lines

    # The first band under the top line is the column-header row.
    rows[0].is_header = True
    return rows, v_lines


def match_country(text: str) -> Optional[str]:
    """Map a chunk of cell text to a canonical country label, if recognised."""
    if not text:
        return None
    low = " " + text.lower().replace("\n", " ") + " "

    best: Optional[str] = None
    best_len = 0
    for canonical, aliases in COUNTRY_ALIASES.items():
        for alias in aliases:
            # Longest alias wins, so "afrique du sud" beats "afrique"
            if alias in low and len(alias) > best_len:
                best, best_len = canonical, len(alias)
    return best


def _column_bounds(v_lines: List[float], page: fitz.Page) -> List[Tuple[float, float]]:
    xs = sorted(set(round(x, 1) for x in v_lines))
    if len(xs) < 2:
        return [(page.rect.x0, page.rect.x1)]
    return list(zip(xs, xs[1:]))


def assign_countries(
    page: fitz.Page,
    rows: List[TableRow],
    v_lines: List[float],
    country_col: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[float, float]]:
    """
    Tag each row with its country.

    If country_col isn't supplied, every column is scored by how many rows
    yield a recognised country, and the best-scoring column is used. That
    column is returned so later pages can reuse it for consistency.
    """
    columns = _column_bounds(v_lines, page)
    words = page.get_text("words")  # (x0, y0, x1, y1, word, ...)

    def cell_text(x0, x1, y0, y1) -> str:
        picked = [
            w[4] for w in words
            if w[0] >= x0 - 2 and w[2] <= x1 + 2 and w[1] >= y0 - 2 and w[3] <= y1 + 2
        ]
        return " ".join(picked)

    data_rows = [r for r in rows if not r.is_header]

    if country_col is None:
        best_col, best_score = None, 0
        for (cx0, cx1) in columns:
            score = 0
            for r in data_rows:
                if match_country(cell_text(cx0, cx1, r.y0, r.y1)):
                    score += 1
            if score > best_score:
                best_col, best_score = (cx0, cx1), score
        country_col = best_col

    if country_col is None:
        return None

    cx0, cx1 = country_col
    for r in data_rows:
        found = match_country(cell_text(cx0, cx1, r.y0, r.y1))
        # Fall back to scanning the whole row if the column came up empty
        if not found:
            found = match_country(cell_text(page.rect.x0, page.rect.x1, r.y0, r.y1))
        r.country = found or UNKNOWN_LABEL

    return country_col


def scan_document(doc: fitz.Document, zoom: float = 2.0) -> List[TableRow]:
    """Detect rows and countries across the whole document."""
    all_rows: List[TableRow] = []
    country_col: Optional[Tuple[float, float]] = None

    for pno in range(doc.page_count):
        page = doc[pno]
        rows, v_lines = build_rows(page, pno, zoom)
        country_col = assign_countries(page, rows, v_lines, country_col) or country_col
        all_rows.extend(rows)

    return all_rows


def summarise_countries(rows: List[TableRow]) -> List[Tuple[str, int]]:
    """Country label -> number of data rows, most common first."""
    counts: dict = {}
    for r in rows:
        if r.is_header:
            continue
        counts[r.country] = counts.get(r.country, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
