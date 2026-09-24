"""Builds PD_LCM_rf_figures34_print.ipynb - the two SHAP figures redrawn in the print style of Figures 1, 2 and 5.

A separate notebook: the earlier large-format versions (pd-lcm-rf-shap-figures) are left untouched so
the two can be compared. Nothing is recomputed - the SHAP values are the out-of-fold ones saved by
pd-lcm-rf-shap-figures (ten Random Forest runs, every person explained by forests that never saw them);
selection stability and single-gene AUC come from pd-lcm-rf-boruta-panel.

  Figure 3 - each Boruta gene's mean |SHAP| as a blade on the side of the call that higher expression
             pushes towards, SD across runs as a halo; beside it, how often Boruta kept the gene inside
             the training folds and the gene's AUC on its own.
  Figure 4 - every person's SHAP value for every gene, PD above each line and controls below, coloured
             by that gene's expression; a dumbbell joins the two class means.
"""
import pathlib
from kaggle_nb import write_nb
HERE = pathlib.Path(__file__).parent
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md("""# Figures 3 and 4 - the SHAP figures in print style

Reads **`pd-lcm-rf-shap-figures`** (out-of-fold SHAP of the Boruta genes, ten Random Forest runs) and
**`pd-lcm-rf-boruta-panel`** (selection stability, single-gene AUC, differential expression). Nothing is
recomputed. The large-format versions stay in `pd-lcm-rf-shap-figures`, so the two can be compared.

* **Figure 3** - mean |SHAP| of each Boruta gene as a blade on the side of the call that higher expression pushes
  towards; halo = SD across runs; bold names with a node on the spine are also nominally differentially expressed.
  Beside it: how often Boruta kept the gene inside the 25 training folds, and the gene's AUC on its own.
* **Figure 4** - one point per person per gene; PD above each line, controls below; colour = that gene's expression;
  the dumbbell joins the two class means.""")

code(r'''import os, glob, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
try:
    get_ipython(); IN_NB = True
except NameError:
    IN_NB = False; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings("ignore")

ON_KAGGLE = Path("/kaggle/input").exists()
FIG_DIR = Path("/kaggle/working/figures") if ON_KAGGLE else Path(os.environ.get("SMOKE_OUT", "smoke_fig")) / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SOURCES = {"shap": ("shap-figures", "figures_rf"), "panel": ("boruta-panel", "outputs_rf/panel")}
def find_any(pattern, source):
    """The file from one named notebook - several inputs carry files with the same name."""
    roots = ["/kaggle/input"] if ON_KAGGLE else os.environ.get("FIG_EXTRA", ".").split(":")
    hits = [h for r in roots for h in glob.glob(f"{r}/**/{pattern}", recursive=True)]
    hits = sorted((h for h in hits if any(k in h for k in SOURCES[source])), key=len)
    if not hits:
        raise FileNotFoundError(f"{pattern} from {source}")
    return hits[0]

# ---- the article's type and palette (Figures 1, 2 and 5) --------------------------------------------
INK, MUTED, FAINT, RULE = "#1B1D20", "#5E656D", "#9AA1A9", "#C9CED4"
UP, DOWN = "#7A2533", "#2F4A6B"                            # raises / lowers the PD call
PD_RIM, CT_RIM = "#1B1D20", "#A3AAB2"
EXPR = LinearSegmentedColormap.from_list("expr", ["#2C4A6E", "#8FA3B8", "#ECE9E3", "#C49A9A", "#7A2533"])
FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "Nimbus Sans", "FreeSans", "DejaVu Sans"]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT_STACK, "font.size": 6.5,
                     "axes.linewidth": 0.5, "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "figure.dpi": 150, "savefig.facecolor": "white", "figure.facecolor": "white"})
print("Kaggle" if ON_KAGGLE else "LOCAL SMOKE RUN")''')

md("## 1. The numbers the figures state")

code(r'''IMP = pd.read_csv(find_any("shap_oof_importance.csv", "shap"))
SV  = pd.read_csv(find_any("shap_oof_values.csv", "shap"))
ANN = pd.read_csv(find_any("13_gene_annotation_master.csv", "panel")).set_index("gene")
IMP["deg"] = IMP.gene.map(ANN["is_deg"]).fillna(False).astype(bool)
IMP["fold_freq"] = IMP.gene.map(ANN["boruta_fold_frequency"])
IMP["alone_auc"] = IMP.gene.map(ANN["single_gene_auc"])
N_RUNS = 10
n_pd = int(SV.groupby("person").true_label.first().sum()); n_ct = SV.person.nunique() - n_pd
print(f"{len(IMP)} genes | {int((IMP.direction_sign > 0).sum())} raise, {int((IMP.direction_sign < 0).sum())} lower the PD call | "
      f"{int(IMP.deg.sum())} also nominally DE | {SV.person.nunique()} people ({n_ct} control, {n_pd} PD)")''')

md("## 2. Figure 3 - SHAP ranking")

code(r'''FW, FH = 183.0, 118.0
fig = plt.figure(figsize=(FW / 25.4, FH / 25.4))
M = fig.add_axes([0, 0, 1, 1]); M.set_xlim(0, FW); M.set_ylim(0, FH); M.axis("off")

up = IMP[IMP.direction_sign > 0].sort_values("mean_abs_shap", ascending=False)
dn = IMP[IMP.direction_sign < 0].sort_values("mean_abs_shap", ascending=False)
PITCH, GAP, Y0 = 2.8, 3.6, 106.0
rows = [(r, Y0 - i * PITCH) for i, (_, r) in enumerate(up.iterrows())]
y_dn0 = Y0 - len(up) * PITCH - GAP
rows += [(r, y_dn0 - i * PITCH) for i, (_, r) in enumerate(dn.iterrows())]
SPINE = 64.0
reach_up = (up.mean_abs_shap + up.shap_std).max(); reach_dn = (dn.mean_abs_shap + dn.shap_std).max()
SCALE = min(56.0 / reach_up, 44.0 / reach_dn)              # millimetres per unit of mean |SHAP|
y_top, y_bot = Y0 + PITCH * 0.9, rows[-1][1] - PITCH * 0.9

def blade(x0, y0, reach, hw):
    tip, c1, c2 = x0 + reach, x0 + reach * 0.34, x0 + reach * 0.80
    V = [(x0, y0), (c1, y0 + hw), (c2, y0 + hw * 0.62), (tip, y0), (c2, y0 - hw * 0.62), (c1, y0 - hw), (x0, y0)]
    return MPath(V, [MPath.MOVETO] + [MPath.CURVE4] * 6)

# scale beneath, gridlines up through the rows
AX_Y = y_bot - 3.0
step = 0.01
ks = np.arange(-np.floor(reach_dn / step), np.floor(reach_up / step) + 1)
for k in ks:
    xv = SPINE + k * step * SCALE
    if k != 0:
        M.plot([xv, xv], [AX_Y, y_top], color="#EEF0F2", lw=0.4, zorder=0)
    M.plot([xv, xv], [AX_Y, AX_Y - 0.8], color="#6B7178", lw=0.45)
    M.text(xv, AX_Y - 2.0, f"{abs(k * step):g}", ha="center", va="center", fontsize=5.8, color=MUTED)
M.plot([SPINE + ks[0] * step * SCALE, SPINE + ks[-1] * step * SCALE], [AX_Y, AX_Y], color="#6B7178", lw=0.45)
M.text(SPINE, AX_Y - 4.6, "mean |SHAP|  (contribution to the out-of-fold PD probability)", ha="center", va="center",
       fontsize=6.0, color=INK)
M.plot([SPINE, SPINE], [AX_Y, y_top + 1.0], color="#30343A", lw=0.7, zorder=5)

for r, yy in rows:
    s = r.direction_sign
    col = UP if s > 0 else DOWN
    reach, sd = s * r.mean_abs_shap * SCALE, r.shap_std * SCALE
    M.add_patch(PathPatch(blade(SPINE, yy, reach + s * sd, 1.18), facecolor=col, alpha=0.16, edgecolor="none", zorder=2))
    M.add_patch(PathPatch(blade(SPINE, yy, reach, 0.98), facecolor=col, alpha=0.92 if r.deg else 0.55, edgecolor="none",
                          zorder=3))
    if r.deg:
        M.scatter([SPINE], [yy], s=5.5, color="white", edgecolor=col, linewidth=0.6, zorder=6)
    lx = SPINE + reach + s * (sd + 1.3)
    M.text(lx, yy, r.symbol, ha="left" if s > 0 else "right", va="center", fontsize=6.2, fontstyle="italic",
           fontweight="bold" if r.deg else "normal", color=INK if r.deg else MUTED, zorder=6)
M.text(SPINE + 1.6, y_top + 1.4, f"raises the PD call  ·  {len(up)} genes", ha="left", va="center", fontsize=6.3,
       color=UP, fontweight="bold")
M.text(SPINE - 1.6, y_dn0 + PITCH * 0.95, f"{len(dn)} genes  ·  lowers it", ha="right", va="center", fontsize=6.3,
       color=DOWN, fontweight="bold")

# ---- stability and single-gene AUC, one row per gene ----
FX0, FX1 = 138.0, 160.0                                    # kept-in-folds bar, 0-100%
M.text((FX0 + FX1) / 2, y_top + 1.4, "kept by Boruta in\nthe 25 training folds", ha="center", va="center", fontsize=5.8,
       color=INK, linespacing=1.1)
M.text(181.0, y_top + 1.4, "AUC\nalone", ha="right", va="center", fontsize=5.8, color=INK, linespacing=1.1)
for r, yy in rows:
    col = UP if r.direction_sign > 0 else DOWN
    M.add_patch(Rectangle((FX0, yy - 0.55), FX1 - FX0, 1.1, facecolor="#F0F1F3", edgecolor="none", zorder=1))
    M.add_patch(Rectangle((FX0, yy - 0.55), (FX1 - FX0) * r.fold_freq, 1.1, facecolor=col, alpha=0.75, edgecolor="none",
                          zorder=2))
    M.text(FX1 + 1.2, yy, f"{100 * r.fold_freq:.0f}%", ha="left", va="center", fontsize=5.7, color=MUTED)
    M.text(181.0, yy, f"{r.alone_auc:.2f}", ha="right", va="center", fontsize=5.9, color=INK)
for v in (0, 0.5, 1.0):
    M.text(FX0 + (FX1 - FX0) * v, AX_Y - 2.0, f"{100 * v:.0f}%", ha="center", va="center", fontsize=5.6, color=MUTED)

# ---- key ----
KY = 4.0
M.scatter([6.0], [KY], s=5.5, color="white", edgecolor=INK, linewidth=0.6)
M.text(8.0, KY, "bold name, node on the spine: also nominally differentially expressed (P < 0.01)", ha="left", va="center",
       fontsize=5.8, color=INK)
M.add_patch(PathPatch(blade(118.0, KY, 9.0, 1.18), facecolor=INK, alpha=0.16, edgecolor="none"))
M.add_patch(PathPatch(blade(118.0, KY, 7.0, 0.98), facecolor=INK, alpha=0.8, edgecolor="none"))
M.text(128.5, KY, f"blade = mean |SHAP|, halo = SD over {N_RUNS} runs", ha="left", va="center", fontsize=5.8, color=INK)

for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"Figure03_shap_ranking_print.{ext}", dpi=600)
plt.show() if IN_NB else plt.close(fig)
print("Figure 3 saved")''')

md("## 3. Figure 4 - every person, every gene")

code(r'''order = IMP.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
nG = len(order)
FW, FH = 183.0, 145.0
fig = plt.figure(figsize=(FW / 25.4, FH / 25.4))
M = fig.add_axes([0, 0, 1, 1]); M.set_xlim(0, FW); M.set_ylim(0, FH); M.axis("off")
TOP, PITCH = 133.0, 4.05
XL, XR = 26.0, 150.0
vmax = np.ceil(SV.shap_value.abs().max() * 100) / 100
sx = lambda v: XL + (v + vmax) / (2 * vmax) * (XR - XL)
YB = TOP - (nG - 1) * PITCH - 3.0
TICKS = np.arange(-np.floor(vmax / 0.02 + 1e-9), np.floor(vmax / 0.02 + 1e-9) + 1) * 0.02   # always includes 0
for v in TICKS:
    M.plot([sx(v)] * 2, [YB, TOP + 2.5], color="#EEF0F2" if abs(v) > 1e-9 else "#30343A", lw=0.4 if abs(v) > 1e-9 else 0.7,
           zorder=0 if abs(v) > 1e-9 else 4)
    M.plot([sx(v)] * 2, [YB, YB - 0.8], color="#6B7178", lw=0.45)
    lab_ = "0" if abs(v) < 1e-9 else (f"{v:.2f}" if v > 0 else "−" + f"{abs(v):.2f}")
    M.text(sx(v), YB - 2.0, lab_, ha="center", va="center", fontsize=5.8, color=MUTED)
M.plot([XL, XR], [YB, YB], color="#6B7178", lw=0.45)
M.text((XL + XR) / 2, YB - 4.8, "SHAP value  (contribution to that person's out-of-fold PD probability)", ha="center",
       va="center", fontsize=6.0, color=INK)

rng = np.random.default_rng(0)
vmax_imp = order.mean_abs_shap.max()
for i, r in order.iterrows():
    yy = TOP - i * PITCH
    M.plot([XL, XR], [yy, yy], color="#E3E6EA", lw=0.4, zorder=1)
    d = SV[SV.gene == r.gene]
    rank = d.feature_value.rank(pct=True).to_numpy()
    lab = d.true_label.to_numpy(int); val = d.shap_value.to_numpy()
    for cls, sgn, rim in ((1, 1, PD_RIM), (0, -1, CT_RIM)):
        m = lab == cls
        M.scatter(sx(val[m]), yy + sgn * rng.uniform(0.35, 1.55, m.sum()), s=3.2, c=EXPR(rank[m]), edgecolor=rim,
                  linewidth=0.22, zorder=3)
    mp, mc = val[lab == 1].mean(), val[lab == 0].mean()
    M.plot([sx(mc), sx(mp)], [yy, yy], color="#30343A", lw=0.8, zorder=5, solid_capstyle="butt")
    M.scatter([sx(mc)], [yy], s=7, color="white", edgecolor="#5E656D", linewidth=0.7, zorder=6)
    M.scatter([sx(mp)], [yy], s=7, color=INK, edgecolor="white", linewidth=0.4, zorder=6)
    M.text(XL - 1.5, yy, r.symbol, ha="right", va="center", fontsize=6.2, fontstyle="italic",
           fontweight="bold" if r.deg else "normal", color=INK if r.deg else MUTED)
    w = 16.0 * r.mean_abs_shap / vmax_imp
    M.add_patch(Rectangle((156.0, yy - 0.6), w, 1.2, facecolor="#B9C3CE", edgecolor="none"))
    M.text(181.0, yy, f"{r.mean_abs_shap:.3f}", ha="right", va="center", fontsize=5.8, color=INK)

# ---- header: what the marks mean ----
HY = TOP + 7.5
M.text(181.0, HY, "mean |SHAP|", ha="right", va="center", fontsize=5.9, color=INK, fontweight="bold")
cax = fig.add_axes([XL / FW, (HY - 0.9) / FH, 22.0 / FW, 1.8 / FH])
cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=EXPR); cax.set_xticks([]); cax.set_yticks([])
for s_ in cax.spines.values():
    s_.set_linewidth(0.3); s_.set_color("#B7BDC4")
M.text(XL - 1.0, HY, "low", ha="right", va="center", fontsize=5.7, color=MUTED)
M.text(XL + 23.0, HY, "high   expression of that gene", ha="left", va="center", fontsize=5.7, color=MUTED)
M.scatter([96.0], [HY], s=3.2, color="#C8CDD3", edgecolor=PD_RIM, linewidth=0.3)
M.text(97.5, HY, f"PD, above each line (n = {n_pd})", ha="left", va="center", fontsize=5.7, color=INK)
M.scatter([96.0], [HY - 3.0], s=3.2, color="#C8CDD3", edgecolor=CT_RIM, linewidth=0.3)
M.text(97.5, HY - 3.0, f"control, below (n = {n_ct})", ha="left", va="center", fontsize=5.7, color=INK)
M.scatter([128.0], [HY], s=7, color=INK, edgecolor="white", linewidth=0.4)
M.scatter([128.0], [HY - 3.0], s=7, color="white", edgecolor="#5E656D", linewidth=0.7)
M.text(129.8, HY, "PD mean", ha="left", va="center", fontsize=5.7, color=INK)
M.text(129.8, HY - 3.0, "control mean", ha="left", va="center", fontsize=5.7, color=INK)
M.text((XL + XR) / 2, YB - 9.0, "every person explained by forests that never saw them  \u00b7  bold names: also nominally "
       "differentially expressed (P < 0.01)", ha="center", va="center", fontsize=5.7, color=MUTED)

for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"Figure04_shap_beeswarm_print.{ext}", dpi=600)
plt.show() if IN_NB else plt.close(fig)
print("Figure 4 saved")''')

md("## 4. Suggested legends for the manuscript")

code(r'''print(f"""Figure 3. SHAP importance of the {len(IMP)} Boruta genes. Blade length is the mean absolute SHAP value over {N_RUNS} Random
Forest runs (every person explained by forests trained without them); the halo is its standard deviation across runs. Blades to
the right: higher expression raises the predicted probability of PD ({len(up)} genes); to the left: it lowers it ({len(dn)} genes).
Bold names with a node on the spine are also nominally differentially expressed (P < 0.01). Beside each gene: how often Boruta
confirmed it inside the 25 training folds, and the gene's out-of-fold AUC on its own.

Figure 4. SHAP value of every person for every Boruta gene, from forests that never saw that person. PD above each line
(n = {n_pd}), controls below (n = {n_ct}); colour, that gene's expression (rank within the cohort); the dumbbell joins the two class
means. Rows are ordered by mean |SHAP| (right).""")''')

write_nb(HERE / "PD_LCM_rf_figures34_print.ipynb", CELLS, "f34")
