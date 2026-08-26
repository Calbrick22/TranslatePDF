"""
Builds the filtered output PDF.
================================

Given the detected rows and the countries the user chose, this assembles a
new document containing ONLY the matching rows.

Two things matter here:

* Dropping a row must remove its pictures too, not just its text. That's
  handled naturally by working with each row's rendered pixel strip: the
  strip carries the row's photos and diagrams with it, so a row that isn't
  copied takes its images with it.

* The result should close up rather than leaving holes where rows were
  removed. Kept strips are stacked onto fresh pages under a repeated
  column header, and each row's translated text is shifted by the same
  offset as its strip so text stays locked to the artwork.
"""

import io
from typing import Callable, Dict, List, Optional, Tuple

import fitz
from PIL import Image

from rowfilter import TableRow

PAGE_MARGIN = 18.0     # top/bottom breathing room on rebuilt pages
ROW_GAP = 0.0          # rows butt together so borders stay continuous


def assign_blocks_to_rows(blocks: List, rows: List[TableRow]) -> Dict[int, List]:
    """
    Map each text block to the row it sits in, by vertical midpoint.

    Returns {row_index: [blocks]}. Blocks that fall outside every row
    (rare - usually page furniture) are attached to the nearest row so
    nothing silently vanishes.
    """
    out: Dict[int, List] = {i: [] for i in range(len(rows))}
    if not rows:
        return out

    for b in blocks:
        mid = (b.bbox.y0 + b.bbox.y1) / 2.0
        placed = False
        for i, r in enumerate(rows):
            if r.y0 <= mid < r.y1:
                out[i].append(b)
                placed = True
                break
        if not placed:
            nearest = min(
                range(len(rows)),
                key=lambda i: min(abs(mid - rows[i].y0), abs(mid - rows[i].y1)),
            )
            out[nearest].append(b)
    return out


def _crop_strip(page_img: Image.Image, y0: float, y1: float, zoom: float) -> Image.Image:
    """Cut the horizontal band [y0, y1] (PDF points) out of a rendered page."""
    top = max(int(round(y0 * zoom)), 0)
    bot = min(int(round(y1 * zoom)), page_img.height)
    if bot <= top:
        bot = min(top + 1, page_img.height)
    return page_img.crop((0, top, page_img.width, bot))


def _img_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def build_filtered_document(
    page_backgrounds: List[Image.Image],
    page_rects: List[fitz.Rect],
    rows: List[TableRow],
    row_blocks: Dict[Tuple[int, int], List],
    keep_row_keys: List[Tuple[int, int]],
    zoom: float,
    draw_text: Callable,
) -> fitz.Document:
    """
    Assemble the output document.

    page_backgrounds : rendered, text-free page images (index = source page)
    page_rects       : source page rectangles
    rows             : every detected row, in document order
    row_blocks       : {(page_no, row_idx): [TextBlock]}
    keep_row_keys    : which (page_no, row_idx) rows to include, in order
    draw_text        : callback(new_page, block, dy) that renders one block
                       shifted vertically by dy
    """
    out = fitz.open()
    if not keep_row_keys:
        return out

    # Header strip per source page, so rebuilt pages keep their column titles
    headers: Dict[int, TableRow] = {}
    for r in rows:
        if r.is_header and r.page_no not in headers:
            headers[r.page_no] = r

    row_lookup: Dict[Tuple[int, int], TableRow] = {}
    idx_by_page: Dict[int, int] = {}
    for r in rows:
        i = idx_by_page.get(r.page_no, 0)
        row_lookup[(r.page_no, i)] = r
        idx_by_page[r.page_no] = i + 1

    template_rect = page_rects[0]
    page_w, page_h = template_rect.width, template_rect.height
    usable_bottom = page_h - PAGE_MARGIN

    current_page: Optional[fitz.Page] = None
    y_cursor = PAGE_MARGIN
    header_src_page: Optional[int] = None

    def start_new_page(src_page_no: int):
        """Open a fresh output page and stamp the column header on it."""
        nonlocal current_page, y_cursor, header_src_page
        current_page = out.new_page(width=page_w, height=page_h)
        y_cursor = PAGE_MARGIN
        header_src_page = src_page_no

        hdr = headers.get(src_page_no) or next(iter(headers.values()), None)
        if hdr is None:
            return

        strip = _crop_strip(page_backgrounds[hdr.page_no], hdr.y0, hdr.y1, zoom)
        h = hdr.height
        rect = fitz.Rect(0, y_cursor, page_w, y_cursor + h)
        current_page.insert_image(rect, stream=_img_to_png_bytes(strip))

        dy = y_cursor - hdr.y0
        for b in row_blocks.get((hdr.page_no, _row_index(hdr, rows)), []):
            draw_text(current_page, b, dy)

        y_cursor += h

    for key in keep_row_keys:
        row = row_lookup.get(key)
        if row is None:
            continue

        h = row.height
        needs_new_page = (
            current_page is None
            or y_cursor + h > usable_bottom
        )
        if needs_new_page:
            start_new_page(row.page_no)

            # A single row taller than a whole page still has to go somewhere;
            # place it and let it use the full page.
            if y_cursor + h > usable_bottom and y_cursor > PAGE_MARGIN:
                pass

        strip = _crop_strip(page_backgrounds[row.page_no], row.y0, row.y1, zoom)
        rect = fitz.Rect(0, y_cursor, page_w, y_cursor + h)
        current_page.insert_image(rect, stream=_img_to_png_bytes(strip))

        dy = y_cursor - row.y0
        for b in row_blocks.get(key, []):
            draw_text(current_page, b, dy)

        y_cursor += h + ROW_GAP

    return out


def _row_index(row: TableRow, rows: List[TableRow]) -> int:
    """Index of a row within its own page."""
    i = 0
    for r in rows:
        if r.page_no == row.page_no:
            if r is row:
                return i
            i += 1
    return 0
