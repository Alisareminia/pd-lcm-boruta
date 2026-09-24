import matplotlib
try:
    get_ipython(); IN_NB = True
except NameError:
    IN_NB = False; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

INK, MUTED, RULE = "#1B1D20", "#5E656D", "#C9CED4"
PD_C, CT_C = "#7A2533", "#6F829A"
CORE_C, PANEL_C, NEURO_C = "#23324A", "#8E1B2E", "#9AA1A9"
SETS = {"both": ("#8E1B2E", "#5E0F1C", "Boruta & DEG"), "boruta": ("#2E5A87", "#1B3A5C", "Boruta only")}
EFFECT = LinearSegmentedColormap.from_list("effect", ["#1E3350", "#4C6583", "#9AAABB", "#EEECE7", "#C3A09E", "#8C4B52", "#551C28"])
FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "Nimbus Sans", "FreeSans", "DejaVu Sans"]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT_STACK, "font.size": 6.3,
                     "axes.linewidth": 0.5, "axes.edgecolor": "#30343A", "axes.labelcolor": INK, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "xtick.labelsize": 5.9, "ytick.labelsize": 5.9,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.1, "ytick.major.size": 2.1,
                     "xtick.major.pad": 1.7, "ytick.major.pad": 1.7, "xtick.color": "#30343A", "ytick.color": "#30343A",
                     "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150, "savefig.facecolor": "white",
                     "figure.facecolor": "white", "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic"})

class Canvas:
    """A print-size figure drawn on a millimetre grid."""
    def __init__(self, w, h):
        self.W, self.H = w, h
        self.fig = plt.figure(figsize=(w / 25.4, h / 25.4))
        self.M = self.fig.add_axes([0, 0, 1, 1]); self.M.set_xlim(0, w); self.M.set_ylim(0, h); self.M.axis("off")
    def ax(self, x, y, w, h):
        return self.fig.add_axes([x / self.W, y / self.H, w / self.W, h / self.H])
    def letter(self, x, y, L, title):
        self.M.text(x, y, L, ha="left", va="baseline", fontsize=9, fontweight="bold", color=INK)
        self.M.text(x + 4.2, y, title, ha="left", va="baseline", fontsize=7.2, color=INK)
    def rule(self, x0, x1, y, lw=0.4, color=RULE):
        self.M.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt")
    def text(self, x, y, s, **kw):
        kw.setdefault("va", "center"); kw.setdefault("fontsize", 6.0)
        return self.M.text(x, y, s, **kw)
    def save(self, stem):
        for ext in ("pdf", "png"):
            self.fig.savefig(OUT / f"{stem}.{ext}", dpi=600 if ext == "png" else None)
        print("saved", stem)
        plt.show() if IN_NB else plt.close(self.fig)
f2 = lambda v: f"{v:.2f}"
def pfmt(p): return f"= {p:.3f}" if p >= 0.001 else "< 0.001"   # "P = 0.003", "P < 0.0001"
