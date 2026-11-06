"""Nesicode's own Office connector: read and write Word (.docx), Excel (.xlsx) and PowerPoint (.pptx)
in-process (python-docx, openpyxl, python-pptx). No third-party server, nothing to install, and every
path is confined to the conversation's workspace (its cage).

Each tool is `fn(args, ws) -> (text_for_the_model, written_relpath | None)`.
"""

from __future__ import annotations

import re
from pathlib import Path

from .agent.workspace import ToolError, Workspace
from .llm import ToolSpec

MAX_TEXT = 40_000


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
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


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

        d = Document(str(p))
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
        from openpyxl import load_workbook

        max_rows = int(args.get("max_rows", 200))
        wb_f = load_workbook(str(p), data_only=False)
        wb_v = load_workbook(str(p), data_only=True)
        names = [args["sheet"]] if args.get("sheet") else wb_f.sheetnames
        for name in names:
            if name not in wb_f.sheetnames:
                raise ToolError(f"no sheet {name!r}; sheets: {', '.join(wb_f.sheetnames)}")
            sf, sv = wb_f[name], wb_v[name]
            out.append(f"## Sheet {name!r} ({sf.max_row} rows × {sf.max_column} cols, range {sf.dimensions})")
            for r, (row_f, row_v) in enumerate(zip(sf.iter_rows(), sv.iter_rows()), 1):
                if r > max_rows:
                    out.append(f"... ({sf.max_row - max_rows} more rows; pass max_rows)")
                    break
                cells = []
                for cf, cv in zip(row_f, row_v):
                    if cf.value is None:
                        cells.append("")
                    elif isinstance(cf.value, str) and cf.value.startswith("="):
                        cells.append(f"{cf.value} → {cv.value}" if cv.value is not None else cf.value)
                    else:
                        cells.append(str(cf.value))
                if any(cells):
                    out.append(f"{r}: " + " | ".join(cells))
    else:
        from pptx import Presentation

        prs = Presentation(str(p))
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


# ------------------------------------------------------------------ Word
def word_write(args: dict, ws: Workspace):
    from docx import Document
    from docx.shared import Cm

    p = _path(ws, args, (".docx",))
    append = bool(args.get("append")) and p.is_file()
    d = Document(str(p)) if append else Document()
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
            d.add_heading(str(b.get("text", "")), level=max(1, min(4, int(b.get("level", 1)))))
        elif kind == "paragraph":
            _runs(d.add_paragraph(), b.get("text", ""))
        elif kind in ("bullets", "numbered"):
            style = "List Bullet" if kind == "bullets" else "List Number"
            for item in b.get("items", []):
                _runs(d.add_paragraph(style=style), item)
        elif kind == "table":
            rows = _rows(b.get("rows"))
            if not rows:
                continue
            t = d.add_table(rows=len(rows), cols=max(len(r) for r in rows))
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
            d.add_picture(str(img), width=Cm(float(b.get("width_cm", 15))))
            if b.get("caption"):
                d.add_paragraph(str(b["caption"])).italic = True
        elif kind == "page_break":
            d.add_page_break()
        else:
            raise ToolError(f"unknown block type {kind!r}")
    d.save(str(p))
    return f"{'Appended to' if append else 'Wrote'} {ws.rel(p)} ({len(blocks)} blocks).", ws.rel(p)


def word_replace(args: dict, ws: Workspace):
    from docx import Document

    p = _path(ws, args, (".docx",), must_exist=True)
    pairs = args.get("replacements") or []
    if not pairs:
        raise ToolError("`replacements` is required: [{find, replace}]")
    d = Document(str(p))
    paragraphs = list(d.paragraphs)
    for table in d.tables:
        for row in table.rows:
            for cell in row.cells:
                paragraphs.extend(cell.paragraphs)
    counts = {}
    for pair in pairs:
        find, repl = str(pair.get("find", "")), str(pair.get("replace", ""))
        if not find:
            continue
        n = 0
        for para in paragraphs:
            if find not in para.text:
                continue
            hit = next((r for r in para.runs if find in r.text), None)
            if hit is not None:  # keep formatting when the text sits inside one run
                n += hit.text.count(find)
                hit.text = hit.text.replace(find, repl)
            else:  # spans several runs: rewrite the paragraph text in its first run
                n += para.text.count(find)
                text = para.text.replace(find, repl)
                for r in para.runs[1:]:
                    r.text = ""
                para.runs[0].text = text
        counts[find] = n
    d.save(str(p))
    return f"Replacements in {ws.rel(p)}: " + ", ".join(f"{k!r}×{v}" for k, v in counts.items()), ws.rel(p)


# ------------------------------------------------------------------ Excel
def excel_write(args: dict, ws: Workspace):
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter
    from openpyxl.utils.cell import coordinate_from_string, column_index_from_string

    p = _path(ws, args, (".xlsx", ".xlsm"))
    rows = _rows(args.get("rows"))
    exists = p.is_file()
    wb = load_workbook(str(p)) if exists else Workbook()
    sheet = str(args.get("sheet") or "").strip()
    if not exists and sheet:
        wb.active.title = sheet
    if sheet and sheet not in wb.sheetnames:
        wb.create_sheet(sheet)
    sh = wb[sheet] if sheet else wb.active
    if args.get("append"):
        r0 = sh.max_row + 1 if sh.max_row > 1 or sh.cell(1, 1).value is not None else 1
        c0 = 1
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
    if fmt := args.get("number_format"):
        for rng, f in (fmt.items() if isinstance(fmt, dict) else []):
            for row in sh[rng]:
                for c in (row if isinstance(row, tuple) else (row,)):
                    c.number_format = f
    if args.get("autofit", True):
        for j in range(max((len(r) for r in rows), default=0)):
            letter = get_column_letter(c0 + j)
            width = max((len(str(c.value)) for c in sh[letter] if c.value is not None), default=8)
            sh.column_dimensions[letter].width = min(60, max(8, width + 2))
    wb.save(str(p))
    return f"Wrote {len(rows)} rows to {ws.rel(p)} › {sh.title} (formulas are computed when the file is opened).", ws.rel(p)


def excel_chart(args: dict, ws: Workspace):
    from openpyxl import load_workbook
    from openpyxl.chart import BarChart, LineChart, PieChart, Reference, ScatterChart, Series
    from openpyxl.utils.cell import range_boundaries

    p = _path(ws, args, (".xlsx", ".xlsm"), must_exist=True)
    wb = load_workbook(str(p))
    sh = wb[args["sheet"]] if args.get("sheet") else wb.active
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
    wb.save(str(p))
    return f"Added a {kind} chart to {ws.rel(p)} › {sh.title} at {args.get('anchor', 'H2')}.", ws.rel(p)


# ------------------------------------------------------------------ PowerPoint
def pptx_write(args: dict, ws: Workspace):
    from pptx import Presentation
    from pptx.util import Cm, Pt

    p = _path(ws, args, (".pptx",))
    append = bool(args.get("append")) and p.is_file()
    prs = Presentation(str(p)) if append else Presentation()
    if not append:
        prs.slide_width, prs.slide_height = Cm(33.867), Cm(19.05)  # 16:9
    slides = args.get("slides") or []
    if not isinstance(slides, list) or not slides:
        raise ToolError("`slides` is required: a list of slides")
    layouts = {"title": 0, "content": 1, "section": 2, "two_content": 3, "title_only": 5, "blank": 6}
    W, H = prs.slide_width, prs.slide_height
    for s in slides:
        kind = s.get("layout", "content")
        if kind in ("image", "table"):
            slide = prs.slides.add_slide(prs.slide_layouts[5])
        else:
            slide = prs.slides.add_slide(prs.slide_layouts[layouts.get(kind, 1)])
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
            pic = slide.shapes.add_picture(str(img), 0, top, height=H - top - Cm(1))
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
    prs.save(str(p))
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
