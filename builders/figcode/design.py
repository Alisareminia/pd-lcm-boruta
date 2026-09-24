# ============================== DESIGN SYSTEM ==============================
import glob, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

try:
    get_ipython()
    IN_NOTEBOOK = True
except NameError:
    IN_NOTEBOOK = False
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

warnings.filterwarnings("ignore")

try:
    from IPython.display import display
except ImportError:
    display = print

# ---- palette matched to the manuscript figures ---------------------------
INK    = "#1A1A1A"
MUTED  = "#5F6469"
FAINT  = "#9AA0A6"
GRIDC  = "#CFD4D9"
PAPER  = "#FFFFFF"

# Desaturated, print-first palette. Saturated primaries are the fastest way to
# make a figure look machine-made; these sit closer to what pigment ink does.
NAVY     = "#26405C"   # table headers, dark end of ramps
STEEL    = "#4C6E9C"   # primary blue  (control / down)
LIGHTBL  = "#A8BED6"
MINT     = "#B3C7B6"
DEEP_RED = "#A64236"   # brick        (PD / up)
CORAL    = "#C0785F"
FOREST   = "#5F7F63"   # sage
SLATE    = "#26405C"
PLUM     = "#6E5F7E"
AMBER    = "#B08442"   # ochre
ROWALT   = "#F2F4F6"
BORDER   = "#C9CFD5"

PANEL_COLORS = {
    "Optimal (exhaustive)":   DEEP_RED,
    "Overlap (Boruta n DEG)": CORAL,
    "Top-5 SHAP":             FOREST,
    "Top-5 DEG":              STEEL,
    "Boruta 20":              PLUM,
}
PANEL_SHORT = {
    "Optimal (exhaustive)":   "Optimal panel",
    "Overlap (Boruta n DEG)": "Overlap panel",
    "Top-5 SHAP":             "Top-5 SHAP",
    "Top-5 DEG":              "Top-5 DEG",
    "Boruta 20":              "Boruta 20",
}

# ---- palette lifted from the manuscript's own figures --------------------
# Sampled straight out of ML LCM PARKINSON5.pdf: the SHAP summary panel uses the
# azure-to-magenta ramp, the volcano uses tan / blue / crimson. Using the paper's
# own colours keeps these figures in the same visual world as the manuscript
# without borrowing its layouts.
SHAP_LOW   = "#008BFB"      # azure   - low expression, protective direction
SHAP_HIGH  = "#FF0051"      # magenta - high expression, drives the PD call
SHAP_CMAP  = LinearSegmentedColormap.from_list(
    "ms_shap", ["#008BFB", "#5261E1", "#7E48CA", "#962DB3", "#B10EA4",
                "#D5008F", "#FF0051"])
# the manuscript's directional SHAP panel uses the sober RdBu pair, not the
# azure-magenta ramp - that ramp is for the beeswarm's expression scale only
MS_UP      = "#B2182B"      # deep red   - upregulated, drives the PD call
MS_DOWN    = "#2166AC"      # steel blue - downregulated, protective
VOL_BG     = "#D9DEE4"      # every other gene
VOL_DEG    = "#EAB67A"      # differentially expressed only
VOL_BOR    = "#95AFD2"      # Boruta only
VOL_BOTH   = "#B62436"      # both tracks
# the volcano tints are set for fills; text needs a darker weight of each
VOL_DEG_INK = "#A06A14"
VOL_BOR_INK = "#40699C"

DIVERGING  = LinearSegmentedColormap.from_list("pd_div", [STEEL, "#EEF2F6", DEEP_RED])
SLATE_SEQ  = LinearSegmentedColormap.from_list("pd_slate", [LIGHTBL, NAVY])
SEQUENTIAL = LinearSegmentedColormap.from_list("pd_seq",
                                               ["#F4F8FC", LIGHTBL, STEEL, "#123A5C"])

# ---- type scale: large and bold, as in the manuscript --------------------
# Grotesque stack: Helvetica where present, else Arial-metric substitutes.
# DejaVu is the last resort and is the classic "default matplotlib" tell.
FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans",
              "Nimbus Sans", "FreeSans", "DejaVu Sans"]

plt.rcParams.update({
    "figure.facecolor":  PAPER,
    "axes.facecolor":    PAPER,
    "savefig.facecolor": PAPER,
    "font.family":       "sans-serif",
    "font.sans-serif":   FONT_STACK,
    "font.size":         10.5,
    "axes.titlesize":    11,
    "axes.titleweight":  "regular",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9.5,
    "axes.labelcolor":   INK,
    "text.color":        INK,
    "xtick.color":       "#3F3F3F",
    "ytick.color":       "#3F3F3F",
    "axes.edgecolor":    "#3F3F3F",
    "axes.linewidth":    0.9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "lines.linewidth":   1.7,
    "legend.frameon":    False,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "figure.dpi":        100,
})

import matplotlib.font_manager as _fm
_avail = {f.name for f in _fm.fontManager.ttflist}
_resolved = next((f for f in FONT_STACK if f in _avail), "DejaVu Sans")
print(f"Typeface in use: {_resolved}")

# Grotesques such as Helvetica carry no dingbats: a check mark or star drawn as
# a text glyph silently becomes a hollow box. Verify before drawing anything.
def _glyph_ok(ch):
    try:
        from fontTools.ttLib import TTFont
        _p = _fm.findfont(_fm.FontProperties(family=_resolved),
                          fallback_to_default=False)
        _t = TTFont(_p, fontNumber=0)
        return any(ord(ch) in tb.cmap for tb in _t["cmap"].tables)
    except Exception:
        return True                      # cannot verify - assume fine

for _ch, _name in [("\u2022", "bullet"), ("\u2013", "en dash")]:
    if not _glyph_ok(_ch):
        print(f"  WARNING: {_name} missing from {_resolved}; marks may render as boxes")
for _ch, _name in [("\u2713", "check mark"), ("\u2605", "star")]:
    if not _glyph_ok(_ch):
        print(f"  note: {_name} absent from {_resolved} - not used, by design")

ON_KAGGLE = Path("/kaggle/working").exists()
OUT_DIR   = Path("/kaggle/working/outputs") if ON_KAGGLE else Path("./fig_outputs")
FIG_DIR   = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FIGURES = []

# Journal figures do not carry editorial headlines - the caption does that work.
# suptitle() therefore records the sentence for the caption block instead of
# printing it across the top of the artwork.
_PENDING_CAPTION = {"text": None}

def suptitle(fig, text, y=None):
    """Record the figure's one-line summary; it is printed with the caption."""
    _PENDING_CAPTION["text"] = text

def head(ax, letter, text, size=10.5, pad=8):
    """Bold panel letter followed by a short, regular-weight descriptor."""
    if letter:
        ax.set_title(f"$\\bf{{{letter}}}$   {text}", fontsize=size,
                     fontweight="regular", loc="left", color=INK, pad=pad)
    else:
        ax.set_title(text, fontsize=size, fontweight="regular",
                     loc="left", color=INK, pad=pad)

def big_title(ax, text, pad=10):
    """Terse title for a single-panel figure; the caption carries the message."""
    ax.set_title(text, fontsize=11, fontweight="regular", loc="left",
                 color=INK, pad=pad)

def framed_legend(ax, **kw):
    """Frameless legend - a box around a key is a needless rule on the page."""
    kw.setdefault("frameon", False)
    kw.setdefault("borderpad", 0.3)
    kw.setdefault("labelspacing", 0.42)
    kw.setdefault("handletextpad", 0.6)
    return ax.legend(**kw)

def dashgrid(ax, axis="both"):
    """A whisper of a grid; heavy dashes are chartjunk."""
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, ls="-", lw=0.5, color="#E4E7EA", alpha=1.0)

def roc_frame(ax, chance=True):
    """Standard ROC axes formatting, used across every ROC figure."""
    if chance:
        ax.plot([0, 1], [0, 1], ls=(0, (3, 3)), lw=0.9, color="#B0B4B8", zorder=1)
    ax.set_xlabel("1 $-$ Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    dashgrid(ax)

def styled_table(ax, cells, header, col_colors=None, col_bg=None,
                 fontsize=12.5, row_h=0.085, head_h=0.115,
                 bbox=(0.0, 0.0, 1.0, 0.92), col_widths=None):
    """Journal-style table: navy header, striped rows, hairline borders.

    col_colors maps a column index to a function(value, row) -> colour, so a
    column can be emphasised without hard-coding it here.
    """
    ax.axis("off")
    # matplotlib splits width evenly unless told otherwise, which clips a wide
    # first column against narrow numeric ones
    table = ax.table(cellText=cells, colLabels=header, cellLoc="center",
                     loc="center", bbox=list(bbox), colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        cell.set_edgecolor(BORDER)
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.set_height(head_h)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_height(row_h)
            cell.set_facecolor(ROWALT if row % 2 == 0 else "#FFFFFF")
            val = cells[row - 1][col]
            if col_bg and col in col_bg:
                bg = col_bg[col](val, cells[row - 1])
                if bg:
                    cell.set_facecolor(bg)
            if col_colors and col in col_colors:
                c, bold = col_colors[col](val, cells[row - 1])
                cell.set_text_props(color=c, fontweight="bold" if bold else "normal")
    return table


def smart_labels(ax, items, fontsize=9.8, gap_pt=7.0, pad_pt=2.0,
                 step_pt=3.0, max_steps=110, lw=0.7, avoid=()):
    """Label points so that no two labels can touch.

    items = [(x, y, text, colour, bold), ...] in data coordinates. Each label is
    tried to the right of its anchor, then to the left; if both are taken it is
    pushed vertically - outward, alternating up and down - until it clears every
    label already placed and stays inside the axes. A hairline leader is drawn
    whenever a label ends up away from its point, so the association survives the
    move.

    `avoid` takes (x, y, radius_in_points) triples - typically the emphasised
    markers - which are treated as occupied space, so a label cannot be parked
    on top of a point the figure is trying to show.

    All packing is done in FIGURE-FRACTION units, never pixels: pixel geometry
    is measured at screen dpi and would be reinterpreted at save dpi, which
    collapses every label towards the origin.
    """
    fig = ax.figure
    FW, FH = fig.get_figwidth(), fig.get_figheight()
    inv = fig.transFigure.inverted()
    gap, pad, step = gap_pt / 72 / FW, pad_pt / 72 / FH, step_pt / 72 / FH
    (ax0, ay0), (ax1, ay1) = inv.transform(ax.get_window_extent().get_points())
    placed = []
    for ox, oy, orad in avoid:
        cx, cy = inv.transform(ax.transData.transform((ox, oy)))
        rx, ry = orad / 72 / FW, orad / 72 / FH
        placed.append((cx - rx, cy - ry, cx + rx, cy + ry))

    def free(b):
        if b[0] < ax0 or b[2] > ax1 or b[1] < ay0 or b[3] > ay1:
            return False
        return not any(b[0] < q[2] + pad and q[0] < b[2] + pad and
                       b[1] < q[3] + pad and q[1] < b[3] + pad for q in placed)

    for x, y, txt, col, bold in sorted(items, key=lambda it: -it[1]):
        px, py = inv.transform(ax.transData.transform((x, y)))
        w = len(str(txt)) * fontsize * 0.60 / 72 / FW
        h = fontsize * 1.25 / 72 / FH
        best = None
        for side in (+1, -1):
            bx = px + gap if side > 0 else px - gap - w
            for s in range(max_steps):
                for dy in ((0.0,) if s == 0 else (s * step, -s * step)):
                    box = (bx, py + dy - h / 2, bx + w, py + dy + h / 2)
                    if free(box):
                        cost = abs(dy) + (0 if side > 0 else 4 * step)
                        if best is None or cost < best[0]:
                            best = (cost, box, side)
                        break
                if best is not None:
                    break
            if best is not None and best[0] < step:
                break
        if best is None:
            continue
        _, box, side = best
        placed.append(box)
        cy = (box[1] + box[3]) / 2
        moved = abs(cy - py) > step * 1.5
        ax.annotate(txt, xy=(x, y), xycoords="data",
                    xytext=(box[0] if side > 0 else box[2], cy),
                    textcoords="figure fraction",
                    ha="left" if side > 0 else "right", va="center",
                    fontsize=fontsize, color=col,
                    fontweight="bold" if bold else "regular", zorder=9,
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=col, lw=lw, alpha=0.5,
                                    shrinkA=1.5, shrinkB=3.0,
                                    connectionstyle="arc3,rad=0.06") if moved
                               else None)
    return placed


def rule(fig, x0, x1, y, lw=0.8, color=BORDER):
    """A hairline across the page - the only divider a journal page needs."""
    fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure,
                          color=color, lw=lw, zorder=0))


def ramp(fig, rect, cmap, lo="low", hi="high", note=None, fontsize=9.3):
    """An inline colour ramp; a framed colorbar is furniture, not information."""
    cax = fig.add_axes(rect)
    cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cmap)
    cax.set_xticks([]); cax.set_yticks([])
    for sp in cax.spines.values():
        sp.set_edgecolor(BORDER); sp.set_linewidth(0.6)
    cax.text(-0.05, 0.5, lo, transform=cax.transAxes, ha="right", va="center",
             fontsize=fontsize, color=MUTED)
    cax.text(1.05, 0.5, hi, transform=cax.transAxes, ha="left", va="center",
             fontsize=fontsize, color=MUTED)
    if note:
        cax.text(0.5, 1.9, note, transform=cax.transAxes, ha="center",
                 va="bottom", fontsize=fontsize, color=MUTED)
    return cax


def finish(fig, name, caption):
    """Save vector + raster, record the caption, and render inline."""
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=400,
                    bbox_inches="tight", facecolor=PAPER)
    headline = _PENDING_CAPTION.pop("text", None) or caption
    _PENDING_CAPTION["text"] = None
    num = len(FIGURES) + 1
    FIGURES.append({"figure": name, "caption": caption, "headline": headline})
    print(f"  [saved] {name}.pdf + .png")
    print(f"  Figure {num}. {headline}")
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)

print(f"Design system loaded. Figures -> {FIG_DIR}")