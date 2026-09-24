"""Builds PD_LCM_rf_volcano_venn.ipynb - secondary notebook: volcano, Venn and cohort agreement.

Nothing is recomputed: the notebook reads the tables of pd-lcm-rf-boruta-panel (differential
expression, the 30-gene Boruta panel). No model is refitted.

Drawn at journal print size (183 mm, double column; 6-7.5 pt type; hairline strokes; 600 dpi).
Colour means one thing everywhere - the gene set - in the manuscript's volcano palette:
burgundy = Boruta and DEG, navy = Boruta only, ochre = DEG only (p < 0.01), grey = other genes.
  a - volcano; every gene a point, the three sets coloured, Boruta genes and the strongest
      DEG-only genes labelled in two aligned columns (italic symbols).
  b - classic exact area-proportional Venn: outlined circles, regions in flat tints, counts inside.
  c - every Boruta gene in every cohort: PD-versus-control effect size (Hedges' g), plus the pooled estimate.
"""
import pathlib
from kaggle_nb import write_nb
HERE = pathlib.Path(__file__).parent
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md("""# Volcano, Venn and per-cohort effects for the Boruta + Random Forest panel

Reads **`pd-lcm-rf-boruta-panel`** (differential expression for 5,622 genes across 63 people;
the 30-gene Boruta + Random Forest panel) and the per-person matrix of **`pd-lcm-rf-core`** for panel c's
per-cohort effect sizes. No model is refitted. The figure is drawn at journal
print size - 183 mm wide - so type and strokes are the sizes a reader will see on the page.

Colour means the gene set, in every panel: **burgundy** Boruta and DEG (p < 0.01), **navy**
Boruta only, **ochre** DEG only, **grey** all other genes.""")

code(r'''import os, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
try:
    get_ipython(); IN_NB = True
except NameError:
    IN_NB = False; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from scipy.optimize import brentq
warnings.filterwarnings("ignore")

ON_KAGGLE = Path("/kaggle/input").exists()
FIG_DIR = Path("/kaggle/working/figures") if ON_KAGGLE else Path(os.environ.get("SMOKE_OUT", "smoke_fig")) / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def find_any(pattern):
    roots = ["/kaggle/input"] if ON_KAGGLE else os.environ.get("FIG_EXTRA", ".").split(":")
    for root in roots:
        hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True), key=len)
        if hits:
            return hits[0]
    raise FileNotFoundError(pattern)

# ---- the manuscript's volcano palette: a fill for marks, a darker weight for strokes and text ----
SETS = {"both":   dict(fill="#8E1B2E", ink="#5E0F1C", name="Boruta & DEG"),     # burgundy
        "boruta": dict(fill="#2E5A87", ink="#1B3A5C", name="Boruta only"),      # navy
        "de":     dict(fill="#BF8A34", ink="#7E5714", name="DEG only"),         # ochre
        "none":   dict(fill="#CDD1D6", ink="#8C9299", name="other genes")}
def tint(c, t):
    """c mixed with white; t = 0 is c itself, t = 1 is white (opaque, so regions never blend)."""
    r = np.array(to_rgba(c)[:3]); return tuple(r * (1 - t) + t)
INK, MUTED, RULE = "#1B1D20", "#5E656D", "#C9CED4"
MM = 1 / 25.4
FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "Nimbus Sans", "FreeSans", "DejaVu Sans"]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT_STACK, "font.size": 7,
                     "axes.linewidth": 0.6, "axes.edgecolor": "#30343A", "axes.labelcolor": INK,
                     "axes.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "xtick.color": "#30343A", "ytick.color": "#30343A",
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.6, "ytick.major.size": 2.6,
                     "xtick.major.pad": 2, "ytick.major.pad": 2, "xtick.direction": "out", "ytick.direction": "out",
                     "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150,
                     "savefig.facecolor": "white", "figure.facecolor": "white"})
print("Kaggle" if ON_KAGGLE else "LOCAL SMOKE RUN")''')

md("## 1. The tables this figure reads")

code(r'''DE  = pd.read_csv(find_any("03_de_results_full.csv"))
BOR = pd.read_csv(find_any("04_boruta_selected_genes.csv"))
COH = pd.read_csv(find_any("01_cohort_by_dataset.csv"))
P_NOM = 0.01

DE["nlp"] = -np.log10(DE["pvalue"])
DE["sig"] = DE["pvalue"] < P_NOM
DE["up"] = DE["log2FC"] > 0
DE["boruta"] = DE["gene"].isin(set(BOR["gene"]))
DE["agree"] = DE["same_direction_datasets"].astype(int)
N_COH = int(DE["agree"].max())
DE["set"] = np.select([DE.boruta & DE.sig, DE.boruta, DE.sig], ["both", "boruta", "de"], "none")
N_GENES, N_PEOPLE = len(DE), int(COH["people"].sum())
CNT = DE["set"].value_counts().reindex(["both", "boruta", "de", "none"]).fillna(0).astype(int)
n_both, n_bor, n_de = CNT["both"], CNT["boruta"], CNT["de"]
n_up_sig, n_dn_sig = int((DE.sig & DE.up).sum()), int((DE.sig & ~DE.up).sum())
print(f"{N_GENES:,} genes, {N_PEOPLE} people | " + ", ".join(f"{SETS[k]['name']} {v}" for k, v in CNT.items())
      + f" | p < {P_NOM}: {n_up_sig} higher, {n_dn_sig} lower in PD | lowest q = {DE.padj.min():.2f}")

AGREE = DE.assign(all_=DE.agree == N_COH).groupby("set")["all_"].agg(["sum", "count"]).reindex(["both", "boruta", "de"])
AGREE.loc["all"] = [int((DE.agree == N_COH).sum()), N_GENES]
AGREE["share"] = AGREE["sum"] / AGREE["count"]
print(AGREE.round(3).to_string())

# ---- per-cohort effect of every Boruta gene (from the core notebook's matrix; no model is fitted) ----
cz = np.load(find_any("core_data.npz"), allow_pickle=True)
Xc, yc, dsc = cz["X"], cz["y"].astype(int), cz["ds"].astype(str)
gidx = {g: i for i, g in enumerate(cz["genes"].astype(str))}
def hedges_g(v, yy):
    a, b = v[yy == 1], v[yy == 0]; n1, n0 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n0 - 1) * b.var(ddof=1)) / (n1 + n0 - 2))
    return (1 - 3 / (4 * (n1 + n0) - 9)) * (a.mean() - b.mean()) / sp
COHORTS = [d for d in pd.Series(dsc).value_counts().index]                   # largest cohort first
PANEL = DE[DE.boruta].copy()
PANEL["grp"] = np.where(PANEL.set == "both", 0, 1)
PANEL = PANEL.sort_values(["grp", "hedges_g_meta"], ascending=[True, False]).reset_index(drop=True)
GMAT = np.array([[hedges_g(Xc[dsc == d, gidx[g]], yc[dsc == d]) for g in PANEL.gene] for d in COHORTS])
NCOH = {d: (int((yc[dsc == d] == 1).sum()), int((yc[dsc == d] == 0).sum())) for d in COHORTS}
from scipy.stats import ttest_ind
PMAT = np.array([[ttest_ind(Xc[(dsc == d) & (yc == 1), gidx[g]], Xc[(dsc == d) & (yc == 0), gidx[g]], equal_var=False).pvalue
                  for g in PANEL.gene] for d in COHORTS])                     # within-cohort Welch test
same = (np.sign(GMAT) == np.sign(PANEL.hedges_g_meta.to_numpy())[None, :]).sum(0)
print(f"per-cohort sign agreement matches the DE table for {int((same == PANEL.agree.to_numpy()).sum())} of {len(PANEL)} genes")''')

md("## 2. Venn geometry: exact areas")

code(r'''def lens_area(d, r1, r2):
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return np.pi * min(r1, r2) ** 2
    a1 = r1 ** 2 * np.arccos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    a2 = r2 ** 2 * np.arccos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    return a1 + a2 - 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))

N_B, N_D = n_bor + n_both, n_de + n_both
rB, rD = np.sqrt(N_B / np.pi), np.sqrt(N_D / np.pi)            # one unit of area per gene
if n_both == 0:
    dist = rB + rD + 0.8
elif n_both >= min(N_B, N_D):
    dist = abs(rB - rD) + 1e-6
else:
    dist = brentq(lambda d: lens_area(d, rB, rD) - n_both, abs(rB - rD) + 1e-9, rB + rD - 1e-9)
cB, cD = np.array([0.0, 0.0]), np.array([dist, 0.0])
xi = (dist ** 2 + rB ** 2 - rD ** 2) / (2 * dist)
yi = np.sqrt(max(rB ** 2 - xi ** 2, 0))

def label_point(kind):
    """The point of a region farthest from every circle edge - where its count sits most comfortably."""
    xs = np.linspace(cB[0] - rB, cD[0] + rD, 500); ys = np.linspace(-rD, rD, 330)
    G = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
    dB, dD = np.hypot(*(G - cB).T), np.hypot(*(G - cD).T)
    inside = {"boruta": (dB < rB) & (dD > rD), "both": (dB < rB) & (dD < rD), "de": (dD < rD) & (dB > rB)}[kind]
    margin = np.minimum(np.abs(dB - rB), np.abs(dD - rD))
    margin[~inside] = -1
    return G[np.argmax(margin)]
LABEL_AT = {k: label_point(k) for k in ("boruta", "both", "de")}
from scipy.stats import hypergeom
N_ALL = len(DE)
EXPECTED = N_B * N_D / N_ALL                                   # overlap if the two lists were unrelated
FOLD = n_both / EXPECTED
P_OVER = hypergeom.sf(n_both - 1, N_ALL, N_D, N_B)
BOR_ONLY = DE[DE.set == "boruta"].sort_values("nlp", ascending=False).symbol.tolist()
def sci(p_):
    """1.2e-40 written the way a caption would write it."""
    e = int(np.floor(np.log10(p_))); m = p_ / 10 ** e
    return f"{m:.0f} $\\times$ 10$^{{{e}}}$"
print(f"radii {rB:.2f} / {rD:.2f}; centre distance {dist:.2f}; lens area {lens_area(dist, rB, rD):.2f} = {n_both} genes; "
      f"overlap {n_both} vs {EXPECTED:.2f} expected = {FOLD:.0f}x, P = {P_OVER:.1e}; panel only: {', '.join(BOR_ONLY)}")''')

md("## 3. The figure")

code(r'''W, Hh = 183 * MM, 138 * MM
fig = plt.figure(figsize=(W, Hh))
fig.canvas.draw(); R = fig.canvas.get_renderer()

# =============================== a: volcano ===============================
TOP_Y, BOT_Y = 0.955, 0.395                                                    # shared top/bottom of a and b
axV = fig.add_axes([0.062, BOT_Y, 0.608, TOP_Y - BOT_Y])
XT = np.ceil(DE.log2FC.abs().max() * 2) / 2
XCOL = XT + 0.10
XL = XT + 0.66
YT = np.ceil(DE.nlp.max() * 2) / 2 + 0.3
axV.set_xlim(-XL, XL); axV.set_ylim(0, YT)
axV.spines["bottom"].set_bounds(-XT, XT)
axV.spines["left"].set_bounds(0, np.floor(YT))
axV.set_xticks(np.arange(-XT, XT + 1e-9, 0.5)); axV.set_yticks(np.arange(0, np.floor(YT) + 1e-9, 1))
axV.axvline(0, color="#E3E6EA", lw=0.5, zorder=0)
thr = -np.log10(P_NOM)
axV.plot([-XT, XT], [thr, thr], color="#7D848C", lw=0.5, ls=(0, (3.5, 2.2)), zorder=1)

style = {"none": dict(s=1.6, lw=0, alpha=0.85, z=2), "de": dict(s=7, lw=0.35, alpha=1, z=3),
         "boruta": dict(s=14, lw=0.45, alpha=1, z=5), "both": dict(s=14, lw=0.45, alpha=1, z=6)}
for k in ["none", "de", "boruta", "both"]:
    d = DE[DE.set == k]
    axV.scatter(d.log2FC, d.nlp, s=style[k]["s"], color=SETS[k]["fill"], edgecolor=SETS[k]["ink"] if k != "none" else "none",
                linewidth=style[k]["lw"], alpha=style[k]["alpha"], zorder=style[k]["z"])
axV.set_xlabel("log$_2$ fold change (PD / control)", labelpad=3)
axV.set_ylabel("$-$log$_{10}$ $P$", labelpad=3)

# direction counts, quiet
axV.text(-XT, 0.10, rf"$\leftarrow$ {n_dn_sig} lower in PD", ha="left", va="bottom", fontsize=6.3, color=MUTED)
axV.text(XT, 0.10, rf"{n_up_sig} higher in PD $\rightarrow$", ha="right", va="bottom", fontsize=6.3, color=MUTED)


# ---- labels in two aligned columns ------------------------------------------------------------
LAB = DE[DE.boruta].copy()
extra = DE[DE.set == "de"].sort_values("pvalue").groupby("up").head(3)      # the strongest DEG-only genes, per side
LAB = pd.concat([LAB, extra])
probe = axV.text(0, 0, "Xg", fontsize=6.3, fontstyle="italic"); h_px = probe.get_window_extent(R).height; probe.remove()
GAP = 1.18 * h_px * YT / axV.get_window_extent(R).height

def column_positions(y, lo, hi, gap):
    o = np.argsort(y); p = np.asarray(y, float)[o].copy()
    gap = min(gap, (hi - lo) / max(len(p) - 1, 1))
    for i in range(1, len(p)):
        p[i] = max(p[i], p[i - 1] + gap)
    if len(p) and p[-1] > hi:
        p -= p[-1] - hi
    for i in range(len(p) - 2, -1, -1):
        p[i] = min(p[i], p[i + 1] - gap)
    if len(p) and p[0] < lo:
        p += lo - p[0]
    out = np.empty_like(p); out[o] = p
    return out

for side, sub in ((1, LAB[LAB.up]), (-1, LAB[~LAB.up])):
    ys = column_positions(sub.nlp.to_numpy(), 0.45, YT - 0.08, GAP)
    xc = side * XCOL
    for (_, r), yl in zip(sub.iterrows(), ys):
        knee = xc - side * 0.07
        axV.plot([r.log2FC, knee, xc - side * 0.012], [r.nlp, yl, yl], color="#A9AFB6", lw=0.35, zorder=2.5,
                 solid_joinstyle="round", solid_capstyle="round")
        axV.text(xc, yl, r.symbol, ha="left" if side > 0 else "right", va="center", fontsize=6.3, fontstyle="italic",
                 color=SETS[r.set]["ink"], zorder=8)

# =============================== b: Venn ===============================
# the classic form: two outlined circles, each region a flat tint of its set colour, counts inside
axN = fig.add_axes([0.682, 0.598, 0.313, TOP_Y - 0.598]); axN.set_aspect("equal"); axN.axis("off")
axN.set_xlim(cB[0] - rB - 0.45, cD[0] + rD + 0.45); axN.set_ylim(-rD - 2.55, rD + 1.60)
from matplotlib.colors import LinearSegmentedColormap as LSC
def wash(centre, r, c, t_hi, t_lo, z, clip=None, angle=(0.35, 1.0)):
    """A circle filled with a soft diagonal wash of its own colour, light at the top left."""
    cmap = LSC.from_list("w", [tint(c, t_hi), tint(c, t_lo)])
    gx, gy = np.meshgrid(np.linspace(0, 1, 256), np.linspace(1, 0, 256))
    img = axN.imshow(angle[0] * gx + angle[1] * gy, cmap=cmap, vmin=0, vmax=angle[0] + angle[1], zorder=z,
                     extent=(centre[0] - r, centre[0] + r, centre[1] - r, centre[1] + r), interpolation="bilinear")
    img.set_clip_path(clip if clip is not None else Circle(centre, r, transform=axN.transData))
    return img
for c_, r_, col in ((cD, rD, SETS["de"]["fill"]), (cB, rB, SETS["boruta"]["fill"])):
    axN.add_patch(Circle(c_, r_ + 0.09, facecolor=tint(col, 0.96), edgecolor="none", zorder=0.5))   # a faint halo
wash(cD, rD, SETS["de"]["fill"], 0.93, 0.78, 1)
wash(cB, rB, SETS["boruta"]["fill"], 0.92, 0.76, 2)
# the intersection is neither circle, so it needs its own outline: B's near arc, then D's far arc
thB = np.arctan2(yi, xi)
thD = np.arctan2(yi, xi - dist)
t1 = np.linspace(thB, -thB, 400)
t2 = np.linspace(2 * np.pi - thD, thD, 400)      # the far circle's arc that runs through its left side
LENS = np.vstack([np.c_[cB[0] + rB * np.cos(t1), cB[1] + rB * np.sin(t1)],
                  np.c_[cD[0] + rD * np.cos(t2), cD[1] + rD * np.sin(t2)]])
lens_clip = Polygon(LENS, closed=True, transform=axN.transData, facecolor="none", edgecolor="none")
axN.add_patch(lens_clip)
wash(cD, rD, SETS["both"]["fill"], 0.88, 0.72, 3, clip=lens_clip)
axN.add_patch(Circle(cD, rD, facecolor="none", edgecolor=SETS["de"]["ink"], lw=0.9, zorder=4))
axN.add_patch(Circle(cB, rB, facecolor="none", edgecolor=SETS["boruta"]["ink"], lw=0.9, zorder=5))

# ---- each region: how many genes, and how they split between higher and lower in PD ------------------
UP, DN = "$\\uparrow$", "$\\downarrow$"
def split_counts(kind):
    d_ = DE[DE.set == kind]
    return int((d_.log2FC > 0).sum()), int((d_.log2FC <= 0).sum())
def region(kind, x, y, count, label, bar_w, fs=10.5, bar=True, stack=False):
    """The count, what the region means, and a bar split into higher / lower in PD."""
    col, fill = SETS[kind]["ink"], SETS[kind]["fill"]
    up, dn = split_counts(kind)
    axN.text(x, y, f"{count}", ha="center", va="center", fontsize=fs, color=col, zorder=7)
    axN.text(x, y - (0.80 if stack else 0.66), label, ha="center", va="center", fontsize=5.1, color=col,
             zorder=7, linespacing=1.15)
    if not bar:
        txt = f"{up} {UP}\n{dn} {DN}" if stack else f"{up} {UP}   {dn} {DN}"
        axN.text(x, y - (1.85 if stack else 1.28), txt, ha="center", va="center", fontsize=5.1, color=col,
                 zorder=7, linespacing=1.3)
        return
    h, yb = 0.32, y - 1.48
    frac = up / max(up + dn, 1)
    axN.add_patch(Rectangle((x - bar_w / 2, yb), bar_w * frac, h, facecolor=fill, edgecolor="none", zorder=7))
    axN.add_patch(Rectangle((x - bar_w / 2 + bar_w * frac, yb), bar_w * (1 - frac), h,
                            facecolor=tint(fill, 0.68), edgecolor="none", zorder=7))
    axN.add_patch(Rectangle((x - bar_w / 2, yb), bar_w, h, facecolor="none", edgecolor=col, lw=0.4, zorder=7.5))
    axN.text(x - bar_w / 2 - 0.14, yb + h / 2, f"{up} {UP}", ha="right", va="center", fontsize=5.1, color=col, zorder=7)
    axN.text(x + bar_w / 2 + 0.14, yb + h / 2, f"{dn} {DN}", ha="left", va="center", fontsize=5.1, color=col, zorder=7)
region("both", (dist - rD + rB) / 2, 0.85, n_both, "in both", 0.0, bar=False)
region("de", (rB + dist + rD) / 2, 0.85, n_de, "DE only", 0.0, bar=False)
region("boruta", (-rB + dist - rD) / 2 + 0.22, 1.30, n_bor, "panel\nonly", 0.0, fs=9.0, bar=False, stack=True)

# the two sets, as a small colour key above the circles
for i_, (k, title, n) in enumerate((("boruta", "Boruta panel", N_B), ("de", f"DEG, $P$ < {P_NOM:g}", N_D))):
    yk = 1.0 - i_ * 0.050
    axN.scatter([0.018], [yk - 0.012], s=16, color=tint(SETS[k]["fill"], 0.45), edgecolor=SETS[k]["ink"],
                linewidth=0.7, transform=axN.transAxes, clip_on=False, zorder=6)
    axN.text(0.052, yk, title, transform=axN.transAxes, ha="left", va="top", fontsize=6.3, fontweight="bold",
             color=SETS[k]["ink"], zorder=6)
    axN.text(1.0, yk, f"{n} genes", transform=axN.transAxes, ha="right", va="top", fontsize=6.0, color=MUTED, zorder=6)

# what the overlap is worth
XL_ = cB[0] - rB - 0.45
BASE = -rD - 0.75
axN.plot([XL_, cD[0] + rD + 0.45], [BASE + 0.62, BASE + 0.62], color="#C9CED4", lw=0.45, zorder=4)
axN.text(XL_, BASE, f"{UP} higher in PD    {DN} lower in PD", ha="left", va="top", fontsize=5.6, color=MUTED, zorder=6)
axN.text(XL_, BASE - 0.80, f"{n_both} of the panel's {N_B} genes are also differentially expressed:", ha="left",
         va="top", fontsize=5.8, color=INK, zorder=6)
axN.text(XL_, BASE - 1.55, f"{FOLD:.0f}$\\times$ the overlap chance gives ({EXPECTED:.1f} genes), $P$ = {sci(P_OVER)}",
         ha="left", va="top", fontsize=5.6, color=MUTED, zorder=6)

# ---- the key for every panel, set as a small ruled table under the Venn --------------------------------
axK = fig.add_axes([0.712, BOT_Y, 0.270, 0.170]); axK.axis("off"); axK.set_xlim(0, 1); axK.set_ylim(0, 1)
axK.plot([0, 1], [0.985, 0.985], color="#3A3F45", lw=0.6)
axK.plot([0, 1], [0.215, 0.215], color="#3A3F45", lw=0.4)
rows_k = [("both", "Boruta & DEG"), ("boruta", "Boruta only"), ("de", f"DEG only  ($P$ < {P_NOM:g})"), ("none", "other genes")]
for i, (k, lab) in enumerate(rows_k):
    yy = 0.87 - i * 0.135
    axK.scatter([0.035], [yy], s=15 if k in ("both", "boruta") else (8 if k == "de" else 5), color=SETS[k]["fill"],
                edgecolor=SETS[k]["ink"] if k != "none" else "none", linewidth=0.45, clip_on=False)
    axK.text(0.085, yy, lab, ha="left", va="center", fontsize=6.6, color=INK)
    axK.text(0.985, yy, f"{CNT[k]:,}", ha="right", va="center", fontsize=6.6, color=INK)
yy = 0.87 - 4 * 0.135
axK.plot([0.005, 0.065], [yy, yy], color="#7D848C", lw=0.6, ls=(0, (3.5, 2.2)))
axK.text(0.085, yy, f"$P$ = {P_NOM:g}  (panel a)", ha="left", va="center", fontsize=6.6, color=INK)
axK.text(0.0, 0.165, f"No gene passes a 5% false-discovery rate\n(lowest $q$ = {DE.padj.min():.2f}).", ha="left",
         va="top", fontsize=6.0, color=MUTED, linespacing=1.35)

# =============================== c: every Boruta gene, cohort by cohort ===============================
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
GCMAP = LinearSegmentedColormap.from_list("effect", ["#1E3350", "#4C6583", "#9AAABB", "#EEECE7",   # muted, dark ends
                                                    "#C3A09E", "#8C4B52", "#551C28"])
GMAX = 1.5
norm = Normalize(-GMAX, GMAX)
axH = fig.add_axes([0.140, 0.080, 0.770, 0.172]); axH.axis("off")
nG = len(PANEL)
ROWY = list(range(len(COHORTS))) + [len(COHORTS) + 0.35]                      # pooled row set apart
axH.set_xlim(-0.5, nG - 0.5); axH.set_ylim(ROWY[-1] + 0.55, -1.55)
VALS = np.vstack([GMAT, PANEL.hedges_g_meta.to_numpy()[None, :]])
for r, yv in enumerate(ROWY):
    for j in range(nG):
        rgb = GCMAP(norm(np.clip(VALS[r, j], -GMAX, GMAX)))
        axH.add_patch(Rectangle((j - 0.5, yv - 0.5), 1, 1, facecolor=rgb, edgecolor="white", lw=0.9))
        if r < len(COHORTS) and PMAT[r, j] < 0.05:                          # significant inside this cohort
            lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            axH.plot(j, yv, "o", ms=1.9, color="white" if lum < 0.55 else "#222629", mew=0)
lab_rows = [f"{d}  ({sum(NCOH[d])})" for d in COHORTS] + ["pooled"]
for yv, lab in zip(ROWY, lab_rows):
    axH.text(-0.75, yv, lab, ha="right", va="center", fontsize=6.2, color=INK if lab != "pooled" else INK,
             fontweight="bold" if lab == "pooled" else "regular")
for j, r in PANEL.iterrows():                                               # gene names under the columns
    axH.text(j, ROWY[-1] + 0.62, r.symbol, ha="center", va="top", rotation=90, fontsize=6.0, fontstyle="italic",
             color=SETS[r.set]["ink"])
for grp, k in ((0, "both"), (1, "boruta")):                                 # which set each column belongs to
    js = np.where(PANEL.grp.to_numpy() == grp)[0]
    if len(js):
        axH.add_patch(Rectangle((js.min() - 0.45, -1.02), js.max() - js.min() + 0.9, 0.26, facecolor=SETS[k]["fill"],
                                edgecolor="none"))
        axH.text((js.min() + js.max()) / 2, -1.12, f"{SETS[k]['name']}  ({len(js)})", ha="center", va="bottom",
                 fontsize=6.3, color=SETS[k]["ink"])
cax = fig.add_axes([0.930, 0.150, 0.009, 0.085])
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=GCMAP), cax=cax, ticks=[-GMAX, 0, GMAX])
cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.8, width=0.5, length=2)
cb.ax.set_yticklabels([f"−{GMAX:g}", "0", f"+{GMAX:g}"])
cax.text(0.5, 1.14, "Hedges' $g$", transform=cax.transAxes, ha="center", va="bottom", fontsize=6.0, color=INK)
fig.add_artist(Line2D([0.925], [0.118], marker="o", ms=2.2, color="#222629", mew=0, transform=fig.transFigure))
fig.text(0.932, 0.118, "$P$ < 0.05", ha="left", va="center", fontsize=5.8, color=INK)
fig.text(0.925, 0.100, "in cohort", ha="left", va="center", fontsize=5.8, color=MUTED)

# =============================== panel letters and titles ===============================
for x, y, letter, ttl in ((0.012, 0.975, "a", "Differential expression and the Boruta panel"),
                          (0.690, 0.975, "b", "Overlap of the gene sets"),
                          (0.012, 0.305, "c", "Every Boruta gene in every cohort: PD versus control effect size")):
    fig.text(x, y, letter, ha="left", va="center", fontsize=9, fontweight="bold", color=INK)
    fig.text(x + 0.022, y, ttl, ha="left", va="center", fontsize=7.2, color=INK)

for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"Figure_volcano_venn.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.03)
print("saved", sorted(p.name for p in FIG_DIR.iterdir()))
plt.show() if IN_NB else plt.close(fig)''')

md("""## 4. Suggested legend for the manuscript""")

code(r'''print(f"""Differential expression and the Boruta + Random Forest panel in {N_PEOPLE} people from four laser-capture cohorts.
(a) Volcano plot of {N_GENES:,} genes (log2 fold change, PD versus control, against -log10 P of the pooled test). Burgundy,
genes selected by Boruta + Random Forest that are also nominally differentially expressed (DEG, P < {P_NOM:g}; n = {n_both});
navy, Boruta-only genes (n = {n_bor}); ochre, DEG-only genes (n = {n_de}); grey, all other genes. Every Boruta gene and the three
strongest DEG-only genes on each side are labelled. No gene passes a 5% false-discovery rate (lowest q = {DE.padj.min():.2f}).
(b) Area-proportional Venn diagram of the two gene sets. (c) Effect size (Hedges' g, PD minus control; oxblood higher,
slate lower in PD) of each of the {len(PANEL)} Boruta genes within each cohort (cohort size in brackets) and pooled across
cohorts (fixed-effect meta-analysis); dots mark P < 0.05 within the cohort (Welch t-test); columns grouped by set. Of the Boruta genes, {int((PANEL.agree == N_COH).sum())} change
in the same direction in all {N_COH} cohorts, compared with {100 * AGREE.loc['all', 'share']:.0f}% of all genes.""")
out = DE.loc[DE.set != "none", ["gene", "symbol", "set", "log2FC", "pvalue", "agree"]].sort_values(["set", "pvalue"])
out.to_csv(FIG_DIR.parent / "venn_membership.csv", index=False)
AGREE.to_csv(FIG_DIR.parent / "cohort_agreement_by_set.csv")
eff = PANEL[["gene", "symbol", "set", "log2FC", "pvalue", "hedges_g_meta", "agree"]].copy()
for d, row in zip(COHORTS, GMAT):
    eff[f"g_{d}"] = row
eff.to_csv(FIG_DIR.parent / "boruta_genes_effect_by_cohort.csv", index=False)
print(eff.round(3).to_string(index=False))''')

write_nb(HERE / "PD_LCM_rf_volcano_venn.ipynb", CELLS, "vv")
