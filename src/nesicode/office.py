"""Nesicode's own Office connector: read and write Word (.docx), Excel (.xlsx) and PowerPoint (.pptx)
in-process (python-docx, openpyxl, python-pptx). No third-party server, nothing to install, and every
path is confined to the conversation's workspace (its cage).

Each tool is `fn(args, ws) -> (text_for_the_model, written_relpath | None)`.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

from .agent.workspace import ToolError, Workspace
from .llm import ToolSpec

MAX_TEXT = 40_000
MAX_FILE = 200 * 1024 * 1024  # documents (and images put in them) larger than this are refused


# ------------------------------------------------------------------ helpers
def _path(ws: Workspace, args: dict, exts: tuple[str, ...], must_exist: bool = False) -> Path:
    rel = str(args.get("path", "")).strip()
    if not rel:
        raise ToolError("`path` is required")
    p = ws.resolve(rel)
    if p.suffix.lower() not in exts:
        raise ToolError(f"`path` must end in {' / '.join(exts)}")
    if must_exist and not p.is_file():
        raise ToolError(f"file not found: {rel}")
    if p.exists() and not p.is_file():  # a folder, a FIFO, a socket… with that name
        raise ToolError(f"{rel} exists and is not a regular file")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# Every file is read and written through the workspace, in memory: never through a link (even one swapped in after
# the checks above) and never a FIFO, which a library's own open() would wait on forever (this runs in-process and
# Stop can't interrupt it).
def _src(ws: Workspace, p: Path) -> io.BytesIO:
    data = ws.read_bytes(p, MAX_FILE + 1)
    if len(data) > MAX_FILE:
        raise ToolError(f"{ws.rel(p)} is too large (over {MAX_FILE // 2**20} MB)")
    return io.BytesIO(data)


def _store(ws: Workspace, p: Path, save) -> None:
    buf = io.BytesIO()
    save(buf)
    ws.write_bytes(p, buf.getvalue())


def _clip(text: str) -> str:
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + f"\n[... truncated, {len(text):,} chars in total]"


_INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")


def _runs(paragraph, text: str) -> None:
    """Add text with light Markdown: **bold**, *italic*, `code`."""
    for part in _INLINE.split(str(text)):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            r = paragraph.add_run(part[1:-1])
            r.font.name = "Menlo"
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            paragraph.add_run(part[1:-1]).italic = True
        else:
            paragraph.add_run(part)


def _rows(value) -> list[list]:
    if not isinstance(value, list) or not all(isinstance(r, list) for r in value):
        raise ToolError("`rows` must be a list of rows, each a list of cells")
    return value


# ------------------------------------------------------------------ read
def office_read(args: dict, ws: Workspace):
    p = _path(ws, args, (".docx", ".xlsx", ".xlsm", ".pptx"), must_exist=True)
    ext = p.suffix.lower()
    out: list[str] = []
    if ext == ".docx":
        from docx import Document

        d = Document(_src(ws, p))
        for para in d.paragraphs:
            if para.text.strip():
                style = para.style.name if para.style is not None else ""
                prefix = "#" * int(style.split()[-1]) + " " if style.startswith("Heading") and style.split()[-1].isdigit() else ""
                out.append(prefix + para.text)
        for i, table in enumerate(d.tables, 1):
            out.append(f"\n[table {i}]")
            for row in table.rows:
                out.append(" | ".join(c.text.strip() for c in row.cells))
    elif ext in (".xlsx", ".xlsm"):
        out += _read_workbook(_src(ws, p).getvalue(), args)
    else:
        from pptx import Presentation

        prs = Presentation(_src(ws, p))
        for i, slide in enumerate(prs.slides, 1):
            out.append(f"## Slide {i}")
            for shape in slide.shapes:
                if shape.has_text_frame and shape.text_frame.text.strip():
                    out.append(shape.text_frame.text)
                if getattr(shape, "has_table", False) and shape.has_table:
                    for row in shape.table.rows:
                        out.append(" | ".join(c.text for c in row.cells))
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
                out.append(f"(notes) {slide.notes_slide.notes_text_frame.text}")
    return _clip("\n".join(out) or "(empty document)"), None


def _formula(value) -> str:
    """The formula text of a cell value read with data_only=False ('' for plain values)."""
    if isinstance(value, str):
        return value if value.startswith("=") else ""
    text = getattr(value, "text", None)  # ArrayFormula
    return text if isinstance(text, str) and text.startswith("=") else ""


def _saved_values(wb, data: bytes, name: str, n_rows: int) -> dict:
    """{(row, col): value Excel saved} for the first n_rows of a sheet: the results of its formulas. Read-only
    sheets parse rows on demand, so the same workbook re-reads them with data_only on (no second load, which
    would parse the shared strings again, and the whole sheet when the file has no <dimension>)."""
    from openpyxl import load_workbook

    src, other = wb, None
    wb._data_only = True  # what load_workbook(data_only=True) sets
    try:
        if not wb.data_only:  # openpyxl changed: load the values separately
            src = other = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        sv = src[name]
        sv.reset_dimensions()
        return {(r, c): v for r, row in enumerate(sv.iter_rows(max_row=n_rows, values_only=True), 1)
                for c, v in enumerate(row) if v is not None}
    finally:
        wb._data_only = False
        if other is not None:
            other.close()


def _read_workbook(data: bytes, args: dict) -> list[str]:
    """Excel as text. Read-only mode streams rows from the file, so a big workbook only costs the rows shown (a
    full load took ~1.5 GB and 20 s for an 8 MB export); the saved formula results are read only when the rows
    shown have formulas."""
    from openpyxl import load_workbook

    max_rows = max(1, int(args.get("max_rows") or 200))
    out: list[str] = []
    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=False)
    try:
        sheets = {s.title: s for s in wb.worksheets}
        charts = [c.title for c in wb.chartsheets]  # a chart moved to its own sheet: no cells to show
        want = str(args.get("sheet") or "")
        if want and want not in sheets:
            if want in charts:
                return [f"## Chart sheet {want!r}: it holds a chart, no cells"]
            raise ToolError(f"no sheet {want!r}; sheets: {', '.join(wb.sheetnames)}")
        budget = MAX_TEXT  # the text is clipped there anyway: don't hold more rows than that in memory
        for name in [want] if want else list(sheets):
            sf = sheets[name]
            n_rows, n_cols = sf.max_row, sf.max_column  # the file's own <dimension>, when it has one
            out.append(f"## Sheet {name!r} ({n_rows} rows × {n_cols} cols, range {sf.calculate_dimension()})"
                       if n_rows and n_cols else f"## Sheet {name!r}")
            sf.reset_dimensions()  # some writers store a wrong <dimension>: read the cells that are really there
            rows, cut = [], ""
            for r, row in enumerate(sf.iter_rows(max_row=max_rows + 1, values_only=True), 1):
                if r > max_rows:
                    cut = f"{n_rows - max_rows} more rows" if n_rows and n_rows > max_rows else "more rows"
                    break
                if budget <= 0:
                    cut = "output limit reached"
                    break
                rows.append(row)
                budget -= sum(len(str(v)) + 3 for v in row if v is not None)
            cached = _saved_values(wb, data, name, len(rows)) if any(_formula(v) for row in rows for v in row) else {}
            for r, row in enumerate(rows, 1):
                cells = []
                for c, v in enumerate(row):
                    if v is None:
                        cells.append("")
                    elif f := _formula(v):
                        cells.append(f"{f} → {cached[(r, c)]}" if (r, c) in cached else f)
                    else:
                        cells.append(str(v))
                if any(cells):
                    out.append(f"{r}: " + " | ".join(cells))
            if cut:
                out.append(f"... ({cut}; pass max_rows" + (" or sheet" if len(sheets) > 1 else "") + ")")
        if not want:
            out += [f"## Chart sheet {c!r}: it holds a chart, no cells" for c in charts]
    finally:
        wb.close()
    return out


# ------------------------------------------------------------------ Word
def word_write(args: dict, ws: Workspace):
    from docx import Document
    from docx.shared import Cm

    p = _path(ws, args, (".docx",))
    append = bool(args.get("append")) and p.is_file()
    d = Document(_src(ws, p)) if append else Document()
    if args.get("title") and not append:
        d.add_heading(str(args["title"]), level=0)
    blocks = args.get("blocks") or []
    if not isinstance(blocks, list):
        raise ToolError("`blocks` must be a list")
    for b in blocks:
        if isinstance(b, str):
            b = {"type": "paragraph", "text": b}
        kind = b.get("type", "paragraph")
        if kind == "heading":
            level = max(1, min(4, int(b.get("level", 1))))
            if _ensure_style(d, f"Heading {level}"):
                d.add_heading(str(b.get("text", "")), level=level)
            else:
                d.add_paragraph().add_run(str(b.get("text", ""))).bold = True
        elif kind == "paragraph":
            _runs(d.add_paragraph(), b.get("text", ""))
        elif kind in ("bullets", "numbered"):
            style = "List Bullet" if kind == "bullets" else "List Number"
            native = style in d.styles  # its numbering comes with the document; it can't be copied in alone
            for k, item in enumerate(b.get("items", []), 1):
                if native:
                    _runs(d.add_paragraph(style=style), item)
                else:  # Word/Google Docs files only carry the styles they use: draw the list by hand
                    para = d.add_paragraph()
                    para.paragraph_format.left_indent, para.paragraph_format.first_line_indent = Cm(1.0), Cm(-0.5)
                    _runs(para, ("• " if kind == "bullets" else f"{k}. ") + str(item))
        elif kind == "table":
            rows = _rows(b.get("rows"))
            if not rows:
                continue
            t = d.add_table(rows=len(rows), cols=max(len(r) for r in rows))
            if _ensure_style(d, "Table Grid"):
                t.style = "Table Grid"
            for i, row in enumerate(rows):
                for j, cell in enumerate(row):
                    t.cell(i, j).text = "" if cell is None else str(cell)
                    if i == 0 and b.get("header", True):
                        for r in t.cell(i, j).paragraphs[0].runs:
                            r.bold = True
        elif kind == "image":
            img = ws.resolve(str(b.get("path", "")))
            if not img.is_file():
                raise ToolError(f"image not found: {b.get('path')}")
            d.add_picture(_src(ws, img), width=Cm(float(b.get("width_cm", 15))))
            if b.get("caption"):
                d.add_paragraph().add_run(str(b["caption"])).italic = True
        elif kind == "page_break":
            d.add_page_break()
        else:
            raise ToolError(f"unknown block type {kind!r}")
    _store(ws, p, d.save)
    return f"{'Appended to' if append else 'Wrote'} {ws.rel(p)} ({len(blocks)} blocks).", ws.rel(p)


_DEFAULT_DOCX = None  # python-docx's own template: where missing styles are copied from


def _ensure_style(d, name: str) -> bool:
    """Make sure the document defines style `name`. Files made by Word, Google Docs or LibreOffice only carry the
    styles they use ('Table Grid', 'Heading 3'…): a missing one is copied from python-docx's default template.
    False when it can't be (list styles need numbering definitions the document doesn't have)."""
    import copy

    from docx import Document
    from docx.oxml.ns import qn

    if name in d.styles:
        return True
    global _DEFAULT_DOCX
    if _DEFAULT_DOCX is None:
        _DEFAULT_DOCX = Document()
    if name not in _DEFAULT_DOCX.styles:
        return False
    el = copy.deepcopy(_DEFAULT_DOCX.styles[name].element)
    if el.find(f"{qn('w:pPr')}/{qn('w:numPr')}") is not None:
        return False
    el.attrib.pop(qn("w:default"), None)
    for tag in ("w:link", "w:rsid"):  # its linked character style and revision id belong to the template
        for x in el.findall(qn(tag)):
            el.remove(x)
    have = {s.get(qn("w:styleId")) for s in d.styles.element.findall(qn("w:style"))}
    for tag in ("w:basedOn", "w:next"):  # only keep references to styles this document has
        x = el.find(qn(tag))
        if x is not None and x.get(qn("w:val")) not in have:
            el.remove(x)
    sid, n = el.get(qn("w:styleId")), 2
    while el.get(qn("w:styleId")) in have:  # the id is taken by another (e.g. localised) style
        el.set(qn("w:styleId"), f"{sid}{n}")
        n += 1
    d.styles.element.append(el)
    return True


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_DRAWN = {_W + "tab": "\t", _W + "ptab": "\t", _W + "cr": "\n", _W + "noBreakHyphen": "-"}  # as python-docx reads them
_NO_WIDTH = {_W + t for t in ("rPr", "lastRenderedPageBreak", "softHyphen", "delText", "delInstrText", "instrText",
                              "commentReference", "annotationRef")}


def _atoms(p_el) -> list[list]:
    """[element, text] for what one paragraph shows, in order: w:t nodes, tabs / line breaks / hyphens as the
    character Word draws, and '\\x00' for anything else in a run (picture, field, page break, footnote mark): no
    match may cross it. Nested paragraphs (text boxes) and tracked deletions are left out."""
    out = []
    for r in p_el.iter(_W + "r"):
        owner = next((a for a in r.iterancestors() if a is p_el or a.tag in (_W + "p", _W + "del", _W + "moveFrom")), None)
        if owner is not p_el:
            continue
        for el in r:
            if not isinstance(el.tag, str) or el.tag in _NO_WIDTH:
                continue
            if el.tag == _W + "t":
                out.append([el, el.text or ""])
            elif el.tag == _W + "br":
                out.append([el, "\n" if el.get(_W + "type") in (None, "textWrapping") else "\x00"])
            else:
                out.append([el, _DRAWN.get(el.tag, "\x00")])
    return out


def _set_text(t_el, text: str) -> None:
    t_el.text = text
    if text != text.strip():
        t_el.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def _replace_in_paragraph(p_el, find: str, repl: str) -> int:
    """Replace every `find` in one paragraph as Word shows it, also when Word split it across runs or it sits in
    a hyperlink. A match spanning several text nodes is written into the first one (keeping that run's formatting)
    and cut from the others, so nothing else moves or repeats. Tabs and line breaks count as '\\t' / '\\n' (in
    `find`, and in `repl`, where they become real tabs and breaks); a match never jumps over a tab, a break, a
    picture or a field that is not in `find`."""
    from docx.oxml import OxmlElement

    if not find or "\x00" in find:
        return 0
    atoms = _atoms(p_el)
    orig = [t for _, t in atoms]
    full = "".join(orig)
    starts, pos = [], 0
    for s in orig:
        starts.append(pos)
        pos += len(s)
    spans, i = [], full.find(find)
    while i != -1:
        end = i + len(find)
        spans.append((i, end, [k for k in range(len(atoms)) if starts[k] < end and starts[k] + len(orig[k]) > i]))
        i = full.find(find, end)
    W_T = _W + "t"
    for h, end, ks in reversed(spans):  # right to left: the offsets of earlier matches stay valid
        first, last = ks[0], ks[-1]
        f_el, l_el = atoms[first][0], atoms[last][0]
        tail = atoms[last][1][end - starts[last]:] if l_el.tag == W_T else ""  # what follows the match in its node
        if f_el.tag == W_T:
            anchor, head = f_el, atoms[first][1][:h - starts[first]]
        else:  # the match starts with a tab or a break: a new text node takes its place
            anchor, head = OxmlElement("w:t"), ""
            f_el.addprevious(anchor)
        pieces = [p for p in re.split(r"(\t|\n)", repl) if p]
        text0 = pieces.pop(0) if pieces and pieces[0] not in ("\t", "\n") else ""
        if first == last and f_el.tag == W_T and tail:  # the rest of this node goes after the replacement
            if pieces and pieces[-1] not in ("\t", "\n"):
                pieces[-1] += tail
            elif pieces:
                pieces.append(tail)
            else:
                text0 += tail
        _set_text(anchor, head + text0)
        atoms[first][1] = head + text0 if f_el.tag == W_T else ""
        at = anchor
        for piece in pieces:  # tabs and line breaks of the replacement, in the anchor's run (its formatting)
            el = OxmlElement({"\t": "w:tab", "\n": "w:br"}.get(piece, "w:t"))
            if piece not in ("\t", "\n"):
                _set_text(el, piece)
            at.addnext(el)
            at = el
        for k in ks[1:] if f_el.tag == W_T else ks:  # the rest of the match: cut from its nodes
            el = atoms[k][0]
            if el.tag != W_T:
                el.getparent().remove(el)
                atoms[k][1] = ""
            elif k == last and first != last:
                _set_text(el, tail)
                atoms[k][1] = tail
            elif k != first:
                _set_text(el, "")
                atoms[k][1] = ""
    return len(spans)


def word_replace(args: dict, ws: Workspace):
    from docx import Document
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from docx.oxml.ns import qn

    p = _path(ws, args, (".docx",), must_exist=True)
    pairs = args.get("replacements") or []
    if not pairs:
        raise ToolError("`replacements` is required: [{find, replace}]")
    d = Document(_src(ws, p))
    # every paragraph once, straight from the XML: body, tables (a merged cell is one cell, not one per grid
    # column), nested tables, text boxes, plus headers and footers
    roots, seen = [d.element.body], set()
    for rel in d.part.rels.values():
        if not rel.is_external and rel.reltype in (RT.HEADER, RT.FOOTER) and id(rel.target_part) not in seen:
            seen.add(id(rel.target_part))
            roots.append(rel.target_part.element)
    fallback = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"
    paragraphs = [(para, next(para.iterancestors(fallback), None) is None) for root in roots for para in root.iter(qn("w:p"))]
    counts = {}
    for pair in pairs:
        find, repl = str(pair.get("find", "")), str(pair.get("replace", ""))
        if not find:
            continue
        n = 0
        for para, counted in paragraphs:
            hits = _replace_in_paragraph(para, find, repl)
            n += hits if counted else 0  # a text box's legacy copy (mc:Fallback) is updated but not counted twice
        counts[find] = n
    _store(ws, p, d.save)
    return f"Replacements in {ws.rel(p)}: " + ", ".join(f"{k!r}×{v}" for k, v in counts.items()), ws.rel(p)


# ------------------------------------------------------------------ Excel
_BAD_SHEET = re.compile(r"[\\/*?:\[\]]")
MAX_FORMAT_CELLS = 100_000


def _load_xlsx(ws: Workspace, p: Path):
    """Open a workbook to edit it. An .xlsm keeps its macros (vbaProject.bin) and its macro-enabled type: without
    keep_vba, saving dropped the VBA and Excel refused to open the file."""
    from openpyxl import load_workbook

    return load_workbook(_src(ws, p), keep_vba=p.suffix.lower() == ".xlsm")


def _save_xlsx(ws: Workspace, wb, p: Path) -> None:
    _store(ws, p, wb.save)
    if wb.vba_archive is not None:  # the in-memory copy keep_vba made: close it now, not at garbage collection
        wb.vba_archive.close()


def _find_sheet(wb, name: str):
    """The worksheet called `name` (Excel compares sheet names case-insensitively), or None."""
    if any(c.title.lower() == name.lower() for c in wb.chartsheets):
        raise ToolError(f"{name!r} is a chart sheet (it holds a chart, no cells); sheets: {', '.join(wb.sheetnames)}")
    return next((s for s in wb.worksheets if s.title.lower() == name.lower()), None)


def _first_sheet(wb):
    """The active sheet, unless it is a chart sheet."""
    from openpyxl.worksheet.worksheet import Worksheet

    if isinstance(wb.active, Worksheet):
        return wb.active
    return wb.worksheets[0] if wb.worksheets else wb.create_sheet()


def _check_sheet_name(name: str) -> str:
    if len(name) > 31:  # openpyxl only warns; Excel 'repairs' the file and renames the sheet
        raise ToolError(f"Excel sheet names have at most 31 characters and {name!r} has {len(name)}: use a shorter `sheet`")
    if _BAD_SHEET.search(name) or name.startswith("'") or name.endswith("'"):
        raise ToolError(f"invalid sheet name {name!r}: Excel doesn't allow \\ / * ? : [ ] or an apostrophe at either end")
    return name


def _last_row(sh) -> int:
    """The last row holding a value. Formatted-but-empty cells (number_format on 'B2:B1000', templates formatted
    down to row 1000) don't count, so appended rows land right below the data."""
    return max((r for (r, _), c in sh._cells.items() if c.value is not None), default=0)


def _format_cells(sh, rng: str) -> list:
    """The cells of 'B2:B20', 'C5', 'B' or '2' as a flat list (a single cell is not a tuple in openpyxl). The size is
    checked first: looking a range up in openpyxl creates every cell in it ('A1:XFD1048576' is 17 billion cells, in
    the server's own process)."""
    from openpyxl.utils.cell import range_boundaries

    ref = rng.strip().replace("$", "").upper()
    bad = f"number_format: {rng!r} is not a cell or range (e.g. 'B2:B20' or 'C5')"
    try:
        c1, r1, c2, r2 = range_boundaries(ref)
    except Exception as exc:  # noqa: BLE001 - openpyxl raises several types for a bad reference
        raise ToolError(bad) from exc
    if c1 is None and r1 is None:
        raise ToolError(bad)
    # whole columns ('B', 'A:C') and whole rows ('2', '2:5') reach as far as the sheet's data does
    size = (c2 - c1 + 1 if c1 is not None else sh.max_column) * (r2 - r1 + 1 if r1 is not None else sh.max_row)
    limit = max(MAX_FORMAT_CELLS, sh.max_row * sh.max_column)  # never more new cells than that, or than the data has
    if size > limit:
        raise ToolError(f"number_format: {rng!r} has more than {limit:,} cells; format only the data range")
    try:
        sel = sh[ref]
    except Exception as exc:  # noqa: BLE001
        raise ToolError(bad) from exc
    if not isinstance(sel, tuple):
        return [sel]
    return [c for item in sel for c in (item if isinstance(item, tuple) else (item,))]


def excel_write(args: dict, ws: Workspace):
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter
    from openpyxl.utils.cell import coordinate_from_string, column_index_from_string

    p = _path(ws, args, (".xlsx", ".xlsm"))
    rows = _rows(args.get("rows"))
    exists = p.is_file()
    if not exists and p.suffix.lower() == ".xlsm":
        raise ToolError("a new .xlsm can't be created here (openpyxl can't write macros): write a .xlsx, "
                        "or edit an existing .xlsm (its macros are kept)")
    fmt = args.get("number_format") or {}
    if not isinstance(fmt, dict):
        raise ToolError("`number_format` must be an object like {'B2:B20': '#,##0.00 €', 'C5': '0%'}")
    wb = _load_xlsx(ws, p) if exists else Workbook()
    sheet = str(args.get("sheet") or "").strip()
    sh = _find_sheet(wb, sheet) if sheet else _first_sheet(wb)
    if sh is None:
        _check_sheet_name(sheet)
        if exists:
            sh = wb.create_sheet(sheet)
        else:
            sh = wb.active
            sh.title = sheet
    if args.get("append"):
        r0, c0 = _last_row(sh) + 1, 1
    else:
        col, r0 = coordinate_from_string(str(args.get("start", "A1")).upper())
        c0 = column_index_from_string(col)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            sh.cell(r0 + i, c0 + j, value)
    if args.get("header") and rows and not args.get("append"):
        for j in range(len(rows[0])):
            sh.cell(r0, c0 + j).font = Font(bold=True)
        sh.freeze_panes = sh.cell(r0 + 1, c0)
    for rng, f in fmt.items():
        for c in _format_cells(sh, str(rng)):
            c.number_format = str(f)
    if args.get("autofit", True):
        cols = {c0 + j for j in range(max((len(r) for r in rows), default=0))}
        widths: dict[int, int] = {}
        for (_, col_i), c in sh._cells.items():  # existing cells only: sh['B'] would create one per row
            if col_i in cols and c.value is not None:
                widths[col_i] = max(widths.get(col_i, 0), len(str(c.value)))
        for col_i in cols:
            sh.column_dimensions[get_column_letter(col_i)].width = min(60, max(8, widths.get(col_i, 8) + 2))
    _save_xlsx(ws, wb, p)
    return f"Wrote {len(rows)} rows to {ws.rel(p)} › {sh.title} (formulas are computed when the file is opened).", ws.rel(p)


def excel_chart(args: dict, ws: Workspace):
    from openpyxl.chart import BarChart, LineChart, PieChart, Reference, ScatterChart, Series
    from openpyxl.utils.cell import range_boundaries

    p = _path(ws, args, (".xlsx", ".xlsm"), must_exist=True)
    wb = _load_xlsx(ws, p)
    sh = _find_sheet(wb, str(args["sheet"])) if args.get("sheet") else _first_sheet(wb)
    if sh is None:
        raise ToolError(f"no sheet {args['sheet']!r}; sheets: {', '.join(wb.sheetnames)}")
    kind = str(args.get("type", "column"))
    try:
        c1, r1, c2, r2 = range_boundaries(str(args["data"]))
    except Exception as exc:
        raise ToolError("`data` must be a range like B1:C13 (first row = series names)") from exc
    data = Reference(sh, min_col=c1, min_row=r1, max_col=c2, max_row=r2)
    cats = None
    if args.get("categories"):
        a1, b1, a2, b2 = range_boundaries(str(args["categories"]))
        cats = Reference(sh, min_col=a1, min_row=b1, max_col=a2, max_row=b2)
    if kind == "scatter":
        chart = ScatterChart()
        if cats is None:
            raise ToolError("scatter needs `categories` (the X values)")
        for col in range(c1, c2 + 1):
            ys = Reference(sh, min_col=col, min_row=r1, max_row=r2)
            chart.series.append(Series(ys, cats, title_from_data=True))
    else:
        chart = {"bar": BarChart, "column": BarChart, "line": LineChart, "pie": PieChart}.get(kind)
        if chart is None:
            raise ToolError("`type` must be column, bar, line, pie or scatter")
        chart = chart()
        if kind in ("bar", "column"):
            chart.type = "bar" if kind == "bar" else "col"
        chart.add_data(data, titles_from_data=True)
        if cats is not None:
            chart.set_categories(cats)
    chart.title = args.get("title") or None
    if args.get("x_title") and hasattr(chart, "x_axis"):
        chart.x_axis.title = args["x_title"]
    if args.get("y_title") and hasattr(chart, "y_axis"):
        chart.y_axis.title = args["y_title"]
    chart.width, chart.height = 18, 9
    sh.add_chart(chart, str(args.get("anchor", "H2")))
    _save_xlsx(ws, wb, p)
    return f"Added a {kind} chart to {ws.rel(p)} › {sh.title} at {args.get('anchor', 'H2')}.", ws.rel(p)


# ------------------------------------------------------------------ PowerPoint
def _fit_layouts(prs, old_w: int, old_h: int) -> None:
    """The default template is 4:3. After resizing the slides, stretch the masters' and layouts' shapes too, so
    titles and text boxes span (and centre on) the 16:9 slide instead of ending ~10 cm before its right edge.
    Only shapes with their own position: the others inherit their master's, which is already stretched."""
    fx, fy = prs.slide_width / old_w, prs.slide_height / old_h
    for master in prs.slide_masters:
        for owner in [master, *master.slide_layouts]:
            for shape in owner.shapes:
                el = shape._element
                if getattr(el, "xfrm", None) is None or el.x is None or el.cx is None:
                    continue
                el.x, el.cx = round(el.x * fx), round(el.cx * fx)
                if fy != 1 and el.y is not None and el.cy is not None:
                    el.y, el.cy = round(el.y * fy), round(el.cy * fy)


def pptx_write(args: dict, ws: Workspace):
    from pptx import Presentation
    from pptx.util import Cm, Pt

    p = _path(ws, args, (".pptx",))
    append = bool(args.get("append")) and p.is_file()
    slides = args.get("slides") or []
    if not isinstance(slides, list) or not slides:
        raise ToolError("`slides` is required: a list of slides")
    # layout name -> index in the default template (image and table slides use 'title only' + a shape)
    layouts = {"title": 0, "content": 1, "section": 2, "two_content": 3, "title_only": 5, "blank": 6, "image": 5, "table": 5}
    for i, s in enumerate(slides, 1):  # check all first: an unknown layout used to drop its bullets silently
        if not isinstance(s, dict) or s.get("layout", "content") not in layouts:
            raise ToolError(f"slide {i}: `layout` must be one of {', '.join(layouts)}")
    prs = Presentation(_src(ws, p)) if append else Presentation()
    if not append:
        old_w, old_h = prs.slide_width, prs.slide_height
        prs.slide_width, prs.slide_height = Cm(33.867), Cm(19.05)  # 16:9
        _fit_layouts(prs, old_w, old_h)
    W, H = prs.slide_width, prs.slide_height
    for s in slides:
        kind = s.get("layout", "content")
        slide = prs.slides.add_slide(prs.slide_layouts[min(layouts[kind], len(prs.slide_layouts) - 1)])
        if slide.shapes.title is not None and s.get("title") is not None:
            slide.shapes.title.text = str(s.get("title", ""))
        ph = [x for x in slide.placeholders if x.placeholder_format.idx != 0]
        if kind == "title" and ph:
            ph[0].text = str(s.get("subtitle", ""))
        elif kind == "section" and ph:
            ph[0].text = str(s.get("subtitle", ""))
        elif kind in ("content", "two_content") and ph:
            cols = [s.get("bullets", [])] + ([s.get("right", [])] if kind == "two_content" else [])
            for frame_ph, items in zip(ph, cols):
                tf = frame_ph.text_frame
                tf.clear()
                for k, item in enumerate(items or []):
                    text, level = (item.get("text", ""), int(item.get("level", 0))) if isinstance(item, dict) else (str(item), 0)
                    para = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                    para.text, para.level = text, max(0, min(4, level))
        elif kind == "image":
            img = ws.resolve(str(s.get("image", "")))
            if not img.is_file():
                raise ToolError(f"image not found: {s.get('image')}")
            top = Cm(4)
            pic = slide.shapes.add_picture(_src(ws, img), 0, top, height=H - top - Cm(1))
            if pic.width > W - Cm(2):
                ratio = (W - Cm(2)) / pic.width
                pic.width, pic.height = int(pic.width * ratio), int(pic.height * ratio)
            pic.left = int((W - pic.width) / 2)
        elif kind == "table":
            rows = _rows(s.get("rows"))
            n_cols = max(len(r) for r in rows)
            gt = slide.shapes.add_table(len(rows), n_cols, Cm(1.5), Cm(4), W - Cm(3), Cm(1.2) * len(rows)).table
            for i, row in enumerate(rows):
                for j, cell in enumerate(row):
                    gt.cell(i, j).text = "" if cell is None else str(cell)
                    for para in gt.cell(i, j).text_frame.paragraphs:
                        for r in para.runs:
                            r.font.size = Pt(14)
        if s.get("notes"):
            slide.notes_slide.notes_text_frame.text = str(s["notes"])
    _store(ws, p, prs.save)
    return f"{'Appended' if append else 'Wrote'} {len(slides)} slides to {ws.rel(p)} ({len(prs.slides)} in total).", ws.rel(p)


# ------------------------------------------------------------------ tool specs
_PATH = {"path": {"type": "string", "description": "File path inside the workspace (relative)."}}

TOOLS: dict[str, tuple[ToolSpec, object]] = {
    "office_read": (ToolSpec(
        "office_read", "Read a Word (.docx), Excel (.xlsx) or PowerPoint (.pptx) file as text: Word paragraphs "
        "(headings as #) and tables; Excel cells per row with formulas (`=SUM(..) → value`); slides with notes.",
        {"type": "object", "properties": {**_PATH, "sheet": {"type": "string", "description": "Excel: only this sheet"},
                                          "max_rows": {"type": "integer", "description": "Excel: rows per sheet (default 200)"}},
         "required": ["path"]}), office_read),
    "word_write": (ToolSpec(
        "word_write", "Create a Word document (or append to one) from blocks. Block types: "
        "{type:'heading', text, level 1-4}, {type:'paragraph', text} (supports **bold**, *italic*, `code`), "
        "{type:'bullets'|'numbered', items:[...]}, {type:'table', rows:[[...]], header:true}, "
        "{type:'image', path, width_cm, caption}, {type:'page_break'}. A plain string is a paragraph.",
        {"type": "object", "properties": {**_PATH, "title": {"type": "string"}, "append": {"type": "boolean"},
                                          "blocks": {"type": "array", "items": {}}},
         "required": ["path", "blocks"]}), word_write),
    "word_replace": (ToolSpec(
        "word_replace", "Find and replace text in an existing Word document (body and tables), keeping formatting "
        "where possible. Good for filling templates ({{name}} → value).",
        {"type": "object", "properties": {**_PATH, "replacements": {"type": "array", "items": {
            "type": "object", "properties": {"find": {"type": "string"}, "replace": {"type": "string"}}}}},
         "required": ["path", "replacements"]}), word_replace),
    "excel_write": (ToolSpec(
        "excel_write", "Write rows to an Excel sheet (creates the file/sheet if needed). Cells starting with '=' are "
        "formulas (e.g. '=SUM(B2:B10)'). Use header:true to bold and freeze the first row; append:true to add rows "
        "below existing data; number_format like {'B2:B20': '#,##0.00 €', 'C2:C20': '0%'}.",
        {"type": "object", "properties": {**_PATH, "sheet": {"type": "string"}, "rows": {"type": "array", "items": {"type": "array", "items": {}}},
                                          "start": {"type": "string", "description": "Top-left cell, default A1"},
                                          "header": {"type": "boolean"}, "append": {"type": "boolean"},
                                          "number_format": {"type": "object"}, "autofit": {"type": "boolean"}},
         "required": ["path", "rows"]}), excel_write),
    "excel_chart": (ToolSpec(
        "excel_chart", "Add a native Excel chart (column, bar, line, pie, scatter) to a sheet. `data` is the range "
        "of the values including the series-name row (e.g. B1:C13); `categories` the labels/X range (e.g. A2:A13).",
        {"type": "object", "properties": {**_PATH, "sheet": {"type": "string"}, "type": {"type": "string", "enum": ["column", "bar", "line", "pie", "scatter"]},
                                          "data": {"type": "string"}, "categories": {"type": "string"}, "title": {"type": "string"},
                                          "x_title": {"type": "string"}, "y_title": {"type": "string"},
                                          "anchor": {"type": "string", "description": "Cell where the chart goes, default H2"}},
         "required": ["path", "type", "data"]}), excel_chart),
    "pptx_write": (ToolSpec(
        "pptx_write", "Create a 16:9 PowerPoint (or append slides). Each slide: {layout:'title', title, subtitle} | "
        "{layout:'content', title, bullets:[text or {text, level}]} | {layout:'two_content', title, bullets, right} | "
        "{layout:'section', title, subtitle} | {layout:'image', title, image:path} | {layout:'table', title, rows:[[...]]}, "
        "plus optional notes. For charts, render a PNG first (e.g. matplotlib) and use an image slide.",
        {"type": "object", "properties": {**_PATH, "append": {"type": "boolean"}, "slides": {"type": "array", "items": {"type": "object"}}},
         "required": ["path", "slides"]}), pptx_write),
}
