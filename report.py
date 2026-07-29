"""PDF report generation for the bolt preload calculator.

Pure-Python (reportlab) so it deploys on Streamlit Community Cloud without any
system packages. ``build_pdf_report`` takes already-formatted strings (the app
owns unit conversion and rounding) plus optional PNG figures and returns the PDF
as bytes, ready for ``st.download_button``.
"""
import io
import re
import textwrap
from datetime import date
from typing import List, Optional, Tuple

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)
import logging

logger = logging.getLogger("bolt_calculator.report")

# A4 content width with the margins used below (18 mm each side).
_CONTENT_WIDTH = A4[0] - 2 * 18 * mm
_ACCENT = colors.HexColor("#1f4e79")


def _init_doc(buf: io.BytesIO, title: str, margins: float = 14 * mm) -> SimpleDocTemplate:
    """Helper to initialize a SimpleDocTemplate with consistent metadata."""
    return SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=margins, rightMargin=margins,
        topMargin=15 * mm, bottomMargin=15 * mm, title=title,
        author="Bolt Preload & Joint Analysis"
    )

def _kv_table(rows: List[Tuple[str, str]]) -> Optional[Table]:
    """A two-column key/value table with a light grid and zebra striping."""
    if not rows:
        return None
    table = Table(rows, colWidths=[_CONTENT_WIDTH * 0.55, _CONTENT_WIDTH * 0.45])
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#333333")),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]
    for i in range(len(rows)):
        if i % 2 == 1:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f4f6f9")))
    table.setStyle(TableStyle(style))
    return table


def _scaled_image(png_bytes: bytes, max_width: float) -> Image:
    """Build a reportlab Image scaled to ``max_width`` preserving aspect ratio."""
    reader = ImageReader(io.BytesIO(png_bytes))
    iw, ih = reader.getSize()
    ratio = ih / float(iw) if iw else 0.6
    return Image(io.BytesIO(png_bytes), width=max_width, height=max_width * ratio)


def build_fe_report(
    *,
    title: str,
    subtitle: str,
    columns: List[str],
    rows: List[List[str]],
    summary: Optional[List[str]] = None,
    notes: Optional[List[str]] = None,
    figures: Optional[List[bytes]] = None,
) -> bytes:
    """Standalone PDF for the FE-import results: a summary, optional result graphics
    and a multi-column table of per-bolt factors of safety. Independent of
    build_pdf_report so the FE feature has its own report. ``rows`` are
    already-formatted strings; ``figures`` are PNG bytes embedded after the summary."""
    buf = io.BytesIO()
    doc = _init_doc(buf, title, margins=14 * mm)
    styles = getSampleStyleSheet()
    h_title = ParagraphStyle("fe_title", parent=styles["Title"], fontSize=17,
                             textColor=_ACCENT, spaceAfter=2)
    h_sub = ParagraphStyle("fe_sub", parent=styles["Normal"], fontSize=9,
                           textColor=colors.HexColor("#555555"), spaceAfter=6)
    h_sec = ParagraphStyle("fe_sec", parent=styles["Heading2"], fontSize=12,
                           textColor=_ACCENT, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("fe_body", parent=styles["Normal"], fontSize=8.5, leading=11)

    story: list = [
        Paragraph(title, h_title),
        Paragraph(f"{subtitle} &nbsp;|&nbsp; Generated {date.today().isoformat()}", h_sub),
        HRFlowable(width="100%", thickness=1, color=_ACCENT, spaceAfter=6),
    ]
    if summary:
        story.append(Paragraph("Summary", h_sec))
        for s in summary:
            story.append(Paragraph(f"&bull;&nbsp; {s}", body))

    full_width = A4[0] - 2 * 14 * mm
    if figures:
        story.append(Paragraph("Result graphics", h_sec))
        for png in figures:
            story.append(Spacer(1, 4))
            story.append(_scaled_image(png, full_width))

    story.append(Paragraph("Per-bolt results", h_sec))
    if not rows:
        story.append(Paragraph("No bolt data available.", body))
    else:
        table = Table([columns] + rows, colWidths=[full_width / len(columns)] * len(columns), repeatRows=1)
        style = [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("BACKGROUND", (0, 0), (-1, 0), _ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]
        for i in range(1, len(rows) + 1):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f4f6f9")))
        table.setStyle(TableStyle(style))
        story.append(table)

    if notes:
        story.append(Paragraph("Notes", h_sec))
        for nline in notes:
            story.append(Paragraph(f"&bull;&nbsp; {nline}", body))

    doc.build(story)
    return buf.getvalue()


def build_pdf_report(
    *,
    title: str,
    subtitle: str,
    result_rows: List[Tuple[str, str]],
    warnings: Optional[List[str]] = None,
    bolt_group_rows: Optional[List[Tuple[str, str]]] = None,
    figures: Optional[List[bytes]] = None,
    assumptions: Optional[List[str]] = None,
) -> bytes:
    """Render a one/two-page analysis report and return it as PDF bytes."""
    buf = io.BytesIO()
    doc = _init_doc(buf, title, margins=18 * mm)

    styles = getSampleStyleSheet()
    h_title = ParagraphStyle("h_title", parent=styles["Title"], fontSize=18,
                             textColor=_ACCENT, spaceAfter=2)
    h_sub = ParagraphStyle("h_sub", parent=styles["Normal"], fontSize=9,
                           textColor=colors.HexColor("#555555"), spaceAfter=6)
    h_sec = ParagraphStyle("h_sec", parent=styles["Heading2"], fontSize=12,
                           textColor=_ACCENT, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9, leading=12)

    story: list = [
        Paragraph(title, h_title),
        Paragraph(f"{subtitle} &nbsp;|&nbsp; Generated {date.today().isoformat()}", h_sub),
        HRFlowable(width="100%", thickness=1, color=_ACCENT, spaceAfter=6),
        Paragraph("Results", h_sec),
    ]
    
    if result_rows:
        kv = _kv_table(result_rows)
        if kv: story.append(kv)

    if bolt_group_rows:
        kv = _kv_table(bolt_group_rows)
        if kv:
            story.append(Paragraph("Bolt Group / Pattern", h_sec))
            story.append(kv)

    if warnings:
        story.append(Paragraph("Warnings &amp; Notes", h_sec))
        for w in warnings:
            story.append(Paragraph(f"&bull;&nbsp; {w}", body))

    if figures:
        for png in figures:
            story.append(Spacer(1, 6))
            story.append(_scaled_image(png, _CONTENT_WIDTH))

    if assumptions:
        story.append(Paragraph("Methodology &amp; Assumptions", h_sec))
        for a in assumptions:
            story.append(Paragraph(f"&bull;&nbsp; {a}", body))

    doc.build(story)
    return buf.getvalue()


# =============================================================================
# Theory-manual PDF (rendered from theory.THEORY_BLOCKS)
# -----------------------------------------------------------------------------
# Display equations are rasterised with matplotlib mathtext; constructs mathtext
# cannot handle (e.g. \begin{cases}) fall back to a unicode/sub-super text form.
# Inline math in prose uses the same lightweight text conversion.
# =============================================================================

_GREEK = {
    r"\Delta": "Δ", r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\varepsilon": "ε", r"\epsilon": "ε", r"\theta": "θ", r"\mu": "μ", r"\pi": "π",
    r"\sigma": "σ", r"\tau": "τ", r"\phi": "φ", r"\Sigma": "Σ", r"\Phi": "Φ",
    r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥", r"\approx": "≈",
    r"\times": "×", r"\cdot": "·", r"\pm": "±", r"\to": "→", r"\Rightarrow": "⇒",
    r"\sqrt": "√", r"\sum": "Σ", r"\neq": "≠", r"\ne": "≠", r"\infty": "∞", r"\circ": "°",
}


def _tex_inline(s: str) -> str:
    """LaTeX snippet -> reportlab mini-HTML (unicode symbols + <sub>/<super>)."""
    s = s.replace(r"\begin{cases}", "{ ").replace(r"\end{cases}", " }")
    s = s.replace(r"\\", ";  ").replace("&", " ")
    s = s.replace(r"\tfrac", r"\frac").replace(r"\dfrac", r"\frac")
    for _ in range(3):
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    for cmd in (r"\left", r"\right", r"\displaystyle", r"\,", r"\!", r"\;"):
        s = s.replace(cmd, "")
    s = s.replace(r"\quad", "  ").replace(r"\qquad", "   ")
    for k, v in _GREEK.items():
        s = s.replace(k, v)
    s = s.replace(r"\text", "").replace(r"\mathrm", "").replace(r"\vec", "")
    # protect sub/super as placeholders, then escape, then restore real tags
    s = re.sub(r"\^\{([^{}]*)\}", "\x01S\x02\\1\x01s\x02", s)
    s = re.sub(r"\^(\w)", "\x01S\x02\\1\x01s\x02", s)
    s = re.sub(r"_\{([^{}]*)\}", "\x01B\x02\\1\x01b\x02", s)
    s = re.sub(r"_(\w)", "\x01B\x02\\1\x01b\x02", s)
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (s.replace("\x01S\x02", "<super>").replace("\x01s\x02", "</super>")
             .replace("\x01B\x02", "<sub>").replace("\x01b\x02", "</sub>"))


def _md_inline(text: str) -> str:
    """Inline markdown (**bold**, *italic*, `code`, $math$) -> reportlab markup."""
    out: List[str] = []
    for seg in re.split(r"(\$[^$]*\$)", text):
        if len(seg) >= 2 and seg.startswith("$") and seg.endswith("$"):
            out.append(_tex_inline(seg[1:-1]))
        else:
            t = seg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
            t = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", t)
            t = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', t)
            out.append(t)
    return "".join(out)


def _md_table(tbl_lines: List[str], styles: dict) -> Optional[Table]:
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in tbl_lines]
    data = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
    if not data:
        return None
    ncol = max(len(r) for r in data)
    table_data = [[Paragraph(_md_inline(c), styles["th"] if ri == 0 else styles["td"]) for c in r]
                  for ri, r in enumerate(data)]
    table = Table(table_data, colWidths=[_CONTENT_WIDTH / ncol] * ncol, repeatRows=1)
    style = [("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
             ("BACKGROUND", (0, 0), (-1, 0), _ACCENT),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    table.setStyle(TableStyle(style))
    return table


def _md_flowables(block: str, styles: dict) -> list:
    """Render one markdown block (prose / headings / lists / tables) to flowables."""
    lines = textwrap.dedent(block).strip("\n").split("\n")
    flow: list = []
    para: List[str] = []
    bullets: List[str] = []

    def flush_para() -> None:
        if para:
            flow.append(Paragraph(_md_inline(" ".join(para)), styles["body"]))
            para.clear()

    def flush_bullets() -> None:
        for b in bullets:
            flow.append(Paragraph("•&nbsp; " + _md_inline(b), styles["bullet"]))
        bullets.clear()

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line:
            flush_para()
            flush_bullets()
        elif line.startswith("#### "):
            flush_para()
            flush_bullets()
            flow.append(Paragraph(_md_inline(line[5:]), styles["h3"]))
        elif line.startswith("### "):
            flush_para()
            flush_bullets()
            flow.append(Paragraph(_md_inline(line[4:]), styles["h2"]))
        elif line.startswith("|"):
            flush_para()
            flush_bullets()
            tbl: List[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl.append(lines[i])
                i += 1
            tbl_flow = _md_table(tbl, styles)
            if tbl_flow:
                flow.append(tbl_flow)
            continue
        elif line.startswith("- "):
            flush_para()
            bullets.append(line[2:])
        elif bullets and raw.startswith("  "):
            bullets[-1] += " " + line
        else:
            para.append(line)
        i += 1
    flush_para()
    flush_bullets()
    return flow


# High DPI -> crisp equations; Computer Modern -> the classic LaTeX look.
_EQ_DPI = 400
_EQ_FONTSIZE = 12


def _cases_to_inline(latex: str) -> str:
    """Rewrite a \\begin{cases} block as one mathtext-renderable line.

    mathtext has no cases/array environment, so e.g.
        b = \\begin{cases} A & C_1 \\\\ B & C_2 \\end{cases}\\,\\text{[mm]}
    becomes  b = A\\ (C_1),\\ B\\ (C_2)\\ \\mathrm{[mm]} .
    """
    m = re.search(r"\\begin\{cases\}(.*?)\\end\{cases\}", latex, re.S)
    if not m:
        return latex
    lhs = latex[:m.start()].strip().rstrip("=").strip()
    post = latex[m.end():]
    unit_m = re.search(r"\\text\{(\[[^}]*\])\}", post)
    unit = unit_m.group(1) if unit_m else ""
    pieces = []
    for row in m.group(1).split(r"\\"):
        row = row.strip()
        if not row:
            continue
        if "&" in row:
            val, cond = row.split("&", 1)
            pieces.append(f"{val.strip()}\\ ({cond.strip()})")
        else:
            pieces.append(row)
    expr = f"{lhs} = " + ",\\ ".join(pieces)
    if unit:
        expr += f"\\ \\mathrm{{{unit}}}"
    return expr


def _eq_png(latex: str) -> Optional[bytes]:
    """Rasterise a display equation with matplotlib mathtext in Computer Modern at
    high DPI. Returns None only if mathtext cannot parse the (preprocessed) string."""
    expr = latex
    if r"\begin{cases}" in expr:
        expr = _cases_to_inline(expr)
    expr = expr.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    expr = re.sub(r"\\text\{([^{}]*)\}", r"\\mathrm{\1}", expr)
    # mathtext wants the long relational names (KaTeX accepts the short ones).
    expr = re.sub(r"\\le(?![a-zA-Z])", r"\\leq", expr)
    expr = re.sub(r"\\ge(?![a-zA-Z])", r"\\geq", expr)
    expr = re.sub(r"\\ne(?![a-zA-Z])", r"\\neq", expr)
    try:
        with matplotlib.rc_context({"mathtext.fontset": "cm"}):
            fig = Figure(figsize=(6.5, 0.7))
            FigureCanvasAgg(fig)
            fig.text(0.01, 0.5, f"${expr}$", fontsize=_EQ_FONTSIZE, va="center", color="#111111")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=_EQ_DPI, bbox_inches="tight",
                        pad_inches=0.05, facecolor="white")
        return buf.getvalue()
    except Exception as e:
        logger.warning("Failed to rasterize equation '%s': %s", latex, e)
        return None


def _equation_flowable(png: bytes) -> Image:
    """An equation PNG at its natural size, capped to the content width."""
    reader = ImageReader(io.BytesIO(png))
    iw, ih = reader.getSize()
    w = iw / _EQ_DPI * 72.0
    h = ih / _EQ_DPI * 72.0
    if w > _CONTENT_WIDTH:
        h *= _CONTENT_WIDTH / w
        w = _CONTENT_WIDTH
    img = Image(io.BytesIO(png), width=w, height=h)
    img.hAlign = "CENTER"
    return img


def build_theory_pdf(title: str, blocks: List[Tuple[str, str]]) -> bytes:
    """Render the theory manual (theory.THEORY_BLOCKS) to a standalone PDF."""
    buf = io.BytesIO()
    doc = _init_doc(buf, title, margins=14 * mm)
    base = getSampleStyleSheet()
    styles: dict = {
        "h2": ParagraphStyle("t_h2", parent=base["Heading2"], fontSize=13, textColor=_ACCENT,
                             spaceBefore=10, spaceAfter=4),
        "h3": ParagraphStyle("t_h3", parent=base["Heading3"], fontSize=11, textColor=_ACCENT,
                             spaceBefore=6, spaceAfter=3),
        "body": ParagraphStyle("t_body", parent=base["Normal"], fontSize=9, leading=12.5, spaceAfter=4),
        "bullet": ParagraphStyle("t_bul", parent=base["Normal"], fontSize=9, leading=12.5,
                                 leftIndent=10, spaceAfter=2),
        "th": ParagraphStyle("t_th", parent=base["Normal"], fontSize=8,
                             textColor=colors.white, fontName="Helvetica-Bold"),
        "td": ParagraphStyle("t_td", parent=base["Normal"], fontSize=8, leading=10),
        "mono": ParagraphStyle("t_mono", parent=base["Normal"], fontName="Courier", fontSize=8,
                               leading=11, spaceAfter=4),
    }
    h_title = ParagraphStyle("t_title", parent=base["Title"], fontSize=16,
                             textColor=_ACCENT, spaceAfter=2)
    h_sub = ParagraphStyle("t_sub", parent=base["Normal"], fontSize=8,
                           textColor=colors.HexColor("#555555"), spaceAfter=6)
    cap = ParagraphStyle("t_cap", parent=base["Normal"], fontSize=8,
                         textColor=colors.HexColor("#555555"), spaceAfter=4)

    story: list = [
        Paragraph(title, h_title),
        Paragraph(f"Generated {date.today().isoformat()}", h_sub),
        HRFlowable(width="100%", thickness=1, color=_ACCENT, spaceAfter=6),
    ]
    for kind, content in blocks:
        if kind == "md":
            story.extend(_md_flowables(content, styles))
        elif kind == "eq":
            png = _eq_png(content)
            if png is not None:
                story.append(Spacer(1, 2))
                story.append(_equation_flowable(png))
                story.append(Spacer(1, 2))
            else:
                story.append(Paragraph(_tex_inline(content), styles["mono"]))
        elif kind == "cap":
            story.append(Paragraph("<i>" + _md_inline(content) + "</i>", cap))
    doc.build(story)
    return buf.getvalue()
