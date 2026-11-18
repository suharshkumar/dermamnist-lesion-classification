import os, json, socket, datetime
socket.getfqdn = lambda *a, **k: "localhost"          # reportlab DNS stall on this Mac
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
                                PageBreak, HRFlowable, KeepTogether)
from PIL import Image as PILImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import sys, glob
_FD = (glob.glob(os.path.join(sys.prefix, "lib", "python3*", "site-packages", "matplotlib", "mpl-data", "fonts", "ttf")) or
       [os.path.join(os.path.dirname(__import__("matplotlib").__file__), "mpl-data", "fonts", "ttf")])[0]
for _n, _f in [("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
               ("DejaVu-Oblique", "DejaVuSans-Oblique.ttf"), ("DejaVu-BoldOblique", "DejaVuSans-BoldOblique.ttf"),
               ("DejaVuMono", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(_n, os.path.join(_FD, _f)))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique", boldItalic="DejaVu-BoldOblique")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUN = os.path.join(ROOT, "run"); ART = os.path.join(RUN, "artifacts")
P2 = os.path.join(ROOT, "DDP_Phase2"); FIGS = os.path.join(P2, "figures"); ASSETS = os.path.join(RUN, "report_assets")
OUT = None

# ------------------------------------------------------------------ helpers
def jload(p):
    try: return json.load(open(p))
    except Exception: return None
BASE_TOP1, BASE_STD, BASE_TOP3, BASE_BAL = 0.8000, 0.7840, 0.9736, 0.5487     # from run/eval.log (archived)
def pct(x, d=2): return "pending" if x is None else f"{100*x:.{d}f}%"
def num(x, d=3): return "pending" if x is None else f"{x:.{d}f}"
def g(dct, *keys):
    for k in keys:
        if dct is None: return None
        dct = dct.get(k) if isinstance(dct, dict) else None
    return dct

# ------------------------------------------------------------------ style (matches the July report)
MAROON = colors.HexColor("#7a1230"); NAVY = colors.HexColor("#1f3b6e"); LIGHT = colors.HexColor("#f3eef0")
GREY = colors.HexColor("#555555"); PINK = colors.HexColor("#fbe9ee"); GREEN = colors.HexColor("#e7f4e9")
PAGE_W, PAGE_H = A4; LM = RM = 2.0 * cm; USABLE_W = PAGE_W - LM - RM
ss = getSampleStyleSheet()
def style(name, parent=None, **kw): return ParagraphStyle(name, parent=parent or ss['Normal'], **kw)
H1 = style('H1', fontName='Helvetica-Bold', fontSize=15, textColor=MAROON, spaceBefore=16, spaceAfter=7, leading=18)
H2 = style('H2', fontName='Helvetica-Bold', fontSize=12, textColor=NAVY, spaceBefore=10, spaceAfter=4, leading=15)
H3 = style('H3', fontName='Helvetica-BoldOblique', fontSize=10.5, textColor=NAVY, spaceBefore=6, spaceAfter=2, leading=13)
BODY = style('BODY', fontName='DejaVu', fontSize=9.6, leading=14.5, alignment=TA_JUSTIFY, spaceAfter=6, textColor=colors.HexColor("#1a1a1a"))
BULLET = style('BULLET', parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)
CAP = style('CAP', fontName='DejaVu-Oblique', fontSize=8.2, alignment=TA_CENTER, textColor=GREY, spaceBefore=3, spaceAfter=10)
CELL = style('CELL', fontName='DejaVu', fontSize=8.2, leading=10.5)
EQ = style('EQ', fontName='DejaVu', fontSize=9.5, leading=13, alignment=TA_CENTER, spaceBefore=4, spaceAfter=6)
NOTE = style('NOTE', parent=BODY, fontSize=9, leading=13, textColor=GREY, leftIndent=10, rightIndent=10,
             borderColor=MAROON, borderWidth=0.6, borderPadding=6, spaceBefore=6, spaceAfter=10)
COVER_T = style('COVER_T', fontName='Helvetica-Bold', fontSize=20, alignment=TA_CENTER, textColor=MAROON, leading=25)
COVER_S = style('COVER_S', fontName='Helvetica', fontSize=12, alignment=TA_CENTER, textColor=colors.HexColor("#333333"), leading=17)

def P(t, s=BODY): return Paragraph(t, s)
def BL(t): return Paragraph(t, BULLET, bulletText="•")
def rule(c=MAROON, w=1.1): return HRFlowable(width="100%", thickness=w, color=c, spaceBefore=4, spaceAfter=8)
def fig(path, width_frac=1.0, caption=None):
    if not os.path.exists(path):
        return P(f"<i>[figure pending: {os.path.basename(path)}]</i>", CAP)
    iw, ih = PILImage.open(path).size; w = USABLE_W * width_frac; h = w * ih / iw
    if h > 0.72 * (PAGE_H - 4 * cm): h = 0.72 * (PAGE_H - 4 * cm); w = h * iw / ih
    flow = [Image(path, width=w, height=h)]
    if caption: flow.append(Paragraph(caption, CAP))
    return KeepTogether(flow)
def table(rows, widths, hl_rows=(), green_rows=(), font=8.5):
    data = [rows[0]] + [[Paragraph(str(c), CELL) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=widths, repeatRows=1)
    ts = [('BACKGROUND', (0, 0), (-1, 0), MAROON), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
          ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), font),
          ('ALIGN', (1, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
          ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
          ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
          ('TOPPADDING', (0, 0), (-1, -1), 3.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5), ('LEFTPADDING', (0, 0), (-1, -1), 5)]
    for r in hl_rows: ts += [('BACKGROUND', (0, r), (-1, r), PINK)]
    for r in green_rows: ts += [('BACKGROUND', (0, r), (-1, r), GREEN)]
    t.setStyle(TableStyle(ts)); return t


def tau_chart(sweep, title):
    """Native reportlab line chart (no matplotlib needed on the training machine)."""
    from reportlab.graphics.shapes import Drawing, String, Line
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.widgets.markers import makeMarker
    W, H = USABLE_W * 0.9, 7.2 * cm
    d = Drawing(W, H); lp = LinePlot(); lp.x, lp.y, lp.width, lp.height = 1.6 * cm, 1.2 * cm, W - 2.4 * cm, H - 2.4 * cm
    series = [("top1", NAVY), ("balanced", MAROON), ("macro_f1", colors.HexColor("#2e7d32"))]
    lp.data = [[(s["tau"], 100 * s[k]) for s in sweep] for k, _ in series]
    for i, (_, c) in enumerate(series):
        lp.lines[i].strokeColor = c; lp.lines[i].strokeWidth = 1.4; lp.lines[i].symbol = makeMarker("FilledCircle"); lp.lines[i].symbol.size = 3
        lp.lines[i].symbol.fillColor = c; lp.lines[i].symbol.strokeColor = c
    lp.xValueAxis.valueMin, lp.xValueAxis.valueMax, lp.xValueAxis.valueStep = 0, 1.5, 0.25
    lo = min(min(100 * s[k] for s in sweep) for k, _ in series); lp.yValueAxis.valueMin = 5 * int(lo // 5) - 5; lp.yValueAxis.valueMax = 100
    lp.xValueAxis.labels.fontName = lp.yValueAxis.labels.fontName = "DejaVu"; lp.xValueAxis.labels.fontSize = lp.yValueAxis.labels.fontSize = 7.5
    d.add(lp)
    d.add(String(W / 2, 0.15 * cm, "prior-correction exponent τ", fontName="DejaVu", fontSize=8, textAnchor="middle"))
    d.add(String(W / 2, H - 0.45 * cm, title, fontName="DejaVu-Bold", fontSize=8.5, textAnchor="middle"))
    for j, (lab, c) in enumerate([("top-1 accuracy", NAVY), ("balanced accuracy", MAROON), ("macro F1", colors.HexColor("#2e7d32"))]):
        x0 = 2.2 * cm + j * 4.6 * cm; yy = H - 1.0 * cm
        d.add(Line(x0, yy, x0 + 0.6 * cm, yy, strokeColor=c, strokeWidth=1.6)); d.add(String(x0 + 0.75 * cm, yy - 3, lab, fontName="DejaVu", fontSize=7.5))
    return d

CLASS_NAMES_R = ['Actinic Ker.', 'Basal Cell Ca.', 'Benign Ker.', 'Dermatofibroma', 'Melanoma', 'Nevi', 'Vascular']
def leaderboard_chart(rows):
    """Horizontal bar chart of published 28/64-px top-1 results with this work highlighted (reportlab native)."""
    from reportlab.graphics.shapes import Drawing, String, Rect, Line
    rows = list(rows); n = len(rows); W = USABLE_W * 0.98; rh = 0.42 * cm; H = n * rh + 1.4 * cm
    d = Drawing(W, H); x0 = 6.3 * cm; xw = W - x0 - 1.6 * cm; vmin, vmax = 68.0, 85.0
    for v in (70, 75, 80, 85):
        x = x0 + (v - vmin) / (vmax - vmin) * xw
        d.add(Line(x, 0.6 * cm, x, H - 0.8 * cm, strokeColor=colors.HexColor("#dddddd"), strokeWidth=0.5))
        d.add(String(x, 0.25 * cm, f"{v}%", fontName="DejaVu", fontSize=7, textAnchor="middle"))
    for i, (name, val) in enumerate(rows):
        yy = H - 0.8 * cm - (i + 1) * rh + 0.08 * cm; ours = "this work" in name
        d.add(String(x0 - 0.15 * cm, yy + 0.09 * cm, name, fontName="DejaVu-Bold" if ours else "DejaVu", fontSize=6.8, textAnchor="end"))
        d.add(Rect(x0, yy, (val - vmin) / (vmax - vmin) * xw, rh - 0.12 * cm, fillColor=MAROON if ours else colors.HexColor("#9fb3d1"), strokeColor=None))
        d.add(String(x0 + (val - vmin) / (vmax - vmin) * xw + 0.1 * cm, yy + 0.09 * cm, f"{val:.1f}", fontName="DejaVu", fontSize=6.8))
    d.add(String(W / 2, H - 0.45 * cm, "DermaMNIST official test split, 28/64-px source images: top-1 accuracy (%)", fontName="DejaVu-Bold", fontSize=8.5, textAnchor="middle"))
    return d

CLASS_NAMES_R = ['Actinic Ker.', 'Basal Cell Ca.', 'Benign Ker.', 'Dermatofibroma', 'Melanoma', 'Nevi', 'Vascular']
def leaderboard_chart(rows):
    """Horizontal bar chart of published 28/64-px top-1 results with this work highlighted (reportlab native)."""
    from reportlab.graphics.shapes import Drawing, String, Rect, Line
    rows = list(rows); n = len(rows); W = USABLE_W * 0.98; rh = 0.42 * cm; H = n * rh + 1.4 * cm
    d = Drawing(W, H); x0 = 6.3 * cm; xw = W - x0 - 1.6 * cm; vmin, vmax = 68.0, 85.0
    for v in (70, 75, 80, 85):
        x = x0 + (v - vmin) / (vmax - vmin) * xw
        d.add(Line(x, 0.6 * cm, x, H - 0.8 * cm, strokeColor=colors.HexColor("#dddddd"), strokeWidth=0.5))
        d.add(String(x, 0.25 * cm, f"{v}%", fontName="DejaVu", fontSize=7, textAnchor="middle"))
    for i, (name, val) in enumerate(rows):
        yy = H - 0.8 * cm - (i + 1) * rh + 0.08 * cm; ours = "this work" in name
        d.add(String(x0 - 0.15 * cm, yy + 0.09 * cm, name, fontName="DejaVu-Bold" if ours else "DejaVu", fontSize=6.8, textAnchor="end"))
        d.add(Rect(x0, yy, (val - vmin) / (vmax - vmin) * xw, rh - 0.12 * cm, fillColor=MAROON if ours else colors.HexColor("#9fb3d1"), strokeColor=None))
        d.add(String(x0 + (val - vmin) / (vmax - vmin) * xw + 0.1 * cm, yy + 0.09 * cm, f"{val:.1f}", fontName="DejaVu", fontSize=6.8))
    d.add(String(W / 2, H - 0.45 * cm, "DermaMNIST official test split, 28/64-px source images: top-1 accuracy (%)", fontName="DejaVu-Bold", fontSize=8.5, textAnchor="middle"))
    return d

