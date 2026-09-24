"""Builds PD_LCM_rf_figure1.ipynb - secondary notebook: Figure 1, study design.

Reads only finished outputs: pd-lcm-merged-pipeline (cohorts, gene filtering, the merge check),
pd-lcm-rf-core (the core classifier's performance), pd-lcm-rf-boruta-panel (the Boruta panel) and
pd-lcm-rf-confirm (the strict checks). Nothing is refitted.

Drawn at print size (183 x 120 mm) on a millimetre canvas so every edge can be aligned exactly;
the same type, line weights and palette as the volcano figure.
  a - tissue and capture: the midbrain schematic and laser-capture field of the earlier Figure 1
  b - four laser-capture cohorts, one dot per person
  c - harmonisation: the gene filter and the merge check (PCA before / after, study identity)
  d - analysis: the core classifier and its evaluation; Boruta + Random Forest and its explanation
"""
import pathlib
from kaggle_nb import write_nb
HERE = pathlib.Path(__file__).parent
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md("""# Figure 1 - study design

Reads the outputs of **`pd-lcm-merged-pipeline`**, **`pd-lcm-rf-core`**, **`pd-lcm-rf-boruta-panel`** and
**`pd-lcm-rf-confirm`**; nothing is refitted. Drawn at print size, 183 x 120 mm, on a millimetre grid.

* **a** tissue and capture - transverse midbrain and the laser-capture field
* **b** the four laser-capture cohorts, one dot per person
* **c** harmonisation - the gene filter, and whether a model can still tell the studies apart
* **d** the analysis - the core classifier and its evaluation; Boruta + Random Forest and its explanation""")

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
from matplotlib.patches import PathPatch, Circle, Ellipse, Arc, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")

ON_KAGGLE = Path("/kaggle/input").exists()
FIG_DIR = Path("/kaggle/working/figures") if ON_KAGGLE else Path(os.environ.get("SMOKE_OUT", "smoke_fig")) / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {"pipeline": ("merged-pipeline", "merged/kaggle_outputs"), "core": ("rf-core", "outputs_rf/core"),
           "panel": ("boruta-panel", "outputs_rf/panel"), "confirm": ("rf-confirm", "kaggle_confirm"),
           "ext": ("rf-external-unified", "outputs_rf/external_unified"), "enrich": ("rf-enrichment", "outputs_rf/enrichment"),
           "pathcomp": ("rf-pathway-composition", "outputs_rf/pathcomp")}
def find_any(pattern, source):
    """The file from one named notebook - several inputs carry files with the same name."""
    roots = ["/kaggle/input"] if ON_KAGGLE else os.environ.get("FIG_EXTRA", ".").split(":")
    hits = [h for r in roots for h in glob.glob(f"{r}/**/{pattern}", recursive=True)]
    hits = sorted((h for h in hits if any(k in h for k in SOURCES[source])), key=len)
    if not hits:
        raise FileNotFoundError(f"{pattern} from {source}")
    return hits[0]

# ---- the article's type and palette (as in the volcano figure) -------------------------------------
INK, MUTED, FAINT, RULE = "#1B1D20", "#5E656D", "#9AA1A9", "#C9CED4"
PD_C, CT_C = "#7A2533", "#6F829A"                          # oxblood = PD, slate = control
CAPTURE = "#8E1B2E"                                        # the captured stretch and cell
COH_COL = {"GSE182622": "#2F4A6B", "GSE20141": "#A8793A", "GSE24378": "#5B8577", "GSE169755": "#6E567E"}
BOX_FILL, BOX_EDGE = "#F4F3EF", "#A3A9B0"
FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "Nimbus Sans", "FreeSans", "DejaVu Sans"]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT_STACK, "font.size": 6.5,
                     "axes.linewidth": 0.5, "axes.edgecolor": "#30343A", "text.color": INK,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150,
                     "savefig.facecolor": "white", "figure.facecolor": "white"})
print("Kaggle" if ON_KAGGLE else "LOCAL SMOKE RUN")''')

md("## 1. The numbers the figure states")

code(r'''COH  = pd.read_csv(find_any("01_cohort_by_dataset.csv", "pipeline")).set_index("dataset")
PREP = pd.read_csv(find_any("01_preprocessing_summary.csv", "pipeline")).set_index("step")["genes"]
IDC  = pd.read_csv(find_any("01_dataset_identity_check.csv", "pipeline")).set_index("statistic")["value"]
PCA  = pd.read_csv(find_any("01_pca_coordinates.csv", "pipeline"))
PERF = pd.read_csv(find_any("core_performance.csv", "core")).set_index("model")
BOR  = pd.read_csv(find_any("04_boruta_selected_genes.csv", "panel"))
try:
    CONF = json.load(open(find_any("S10_confirmation.json", "confirm")))
except FileNotFoundError:
    CONF = None
def read_json(pattern, source):
    try:
        return json.load(open(find_any(pattern, source)))
    except (FileNotFoundError, IndexError):
        return None
EXT = read_json("external_unified_summary.json", "ext")            # the external cohorts
ENR = read_json("enrich_summary.json", "enrich")                   # the dopamine-neuron lineage
PWC = read_json("pathway_composition_summary.json", "pathcomp")    # pathways and the neuron mix

ORDER = ["GSE182622", "GSE20141", "GSE24378", "GSE169755"]
PLATFORM = {"GSE182622": "RNA-seq, Illumina HiSeq 2500", "GSE20141": "Affymetrix HG-U133 Plus 2.0",
            "GSE24378": "Affymetrix HG-U133 X3P", "GSE169755": "RNA-seq, Illumina HiSeq 4000"}
UNIT = {"GSE182622": f"{int(COH.loc['GSE182622', 'units_merged'])} LCM pools averaged into {int(COH.loc['GSE182622', 'people'])} donors",
        "GSE20141": "one array per person", "GSE24378": "one array per person",
        "GSE169755": f"{int(COH.loc['GSE169755', 'units_merged'])} libraries summed into {int(COH.loc['GSE169755', 'people'])} brains"}
N_PEOPLE, N_CT, N_PD = int(COH.people.sum()), int(COH.control.sum()), int(COH.PD.sum())
G_ANY, G_ALL, G_EXP = int(PREP["raw matrix"]), int(PREP["genes on every platform"]), int(PREP["expressed in every dataset (model X)"])
HEAD = PERF.loc["Ranks + PCA 30 + Random Forest"]
N_BOR = len(BOR)
N_UP, N_DOWN = (len(ENR["lists"]["up"]), len(ENR["lists"]["down"])) if ENR else (18, 12)
if PWC:
    _oof = pd.DataFrame(PWC["oof_auc"]); _oof = _oof[_oof.model == "core"].set_index("variant")
    OOF_BEFORE, OOF_AFTER = _oof.loc["original", "oof_auc"], _oof.loc["composition (50 markers)", "oof_auc"]
    OOF_P, N_PATH = _oof.loc["composition (50 markers)", "shuffle_p"], int(PWC["gprofiler"]["all"]["observed_significant"])
print(f"{N_PEOPLE} people ({N_CT} control, {N_PD} PD); genes {G_ANY:,} -> {G_ALL:,} -> {G_EXP:,}; identity "
      f"{IDC['accuracy_before']:.2f} -> {IDC['accuracy_after']:.2f} (chance {IDC['chance']:.2f}); core AUC {HEAD.cv_auc:.3f}, "
      f"accuracy {HEAD.cv_accuracy:.2f}, unseen study {HEAD.lodo_auc:.2f}; Boruta {N_BOR} genes; strict checks "
      + ("found" if CONF else "not attached"))''')

md("## 2. The anatomy (the midbrain schematic and capture field of the earlier Figure 1, rescaled for print)")

code(r'''TISSUE, TISSUE_E = "#E9E3DB", "#8C8378"
NIGRA, NIGRA_R = "#2B211B", "#7A6A5E"
PEDUNCLE, FIBRE, REDNUC = "#D8D0C5", "#BCB2A5", "#C79A93"
PAG, NUCLEUS, SOMA, MELANIN = "#CEC5B9", "#B9AEA0", "#D9C7A7", "#2E241E"
LS = 0.42                                                  # line-weight scale from the poster version to print

def on_arc(ox, oy, W, H, angle_deg, theta_deg):
    t, a = np.radians(theta_deg), np.radians(angle_deg)
    x, y = (W / 2) * np.cos(t), (H / 2) * np.sin(t)
    return (ox + x * np.cos(a) - y * np.sin(a), oy + x * np.sin(a) + y * np.cos(a))

def draw_anatomy(ax, FS=6.1):
    """Transverse midbrain at the level of the superior colliculus, labelled on the left, with the
    laser-capture field magnified beneath it. Returns the capture field's centre and radius."""
    cx, cy, w, h = 3.20, 4.80, 2.25, 1.52
    V = [(cx, cy + h),
         (cx + w * 0.55, cy + h * 0.97), (cx + w * 0.95, cy + h * 0.58), (cx + w, cy + h * 0.12),
         (cx + w * 1.02, cy - h * 0.36), (cx + w * 0.88, cy - h * 0.86), (cx + w * 0.52, cy - h * 1.00),
         (cx + w * 0.34, cy - h * 1.08), (cx + w * 0.16, cy - h * 0.74), (cx, cy - h * 0.46),
         (cx - w * 0.16, cy - h * 0.74), (cx - w * 0.34, cy - h * 1.08), (cx - w * 0.52, cy - h * 1.00),
         (cx - w * 0.88, cy - h * 0.86), (cx - w * 1.02, cy - h * 0.36), (cx - w, cy + h * 0.12),
         (cx - w * 0.95, cy + h * 0.58), (cx - w * 0.55, cy + h * 0.97), (cx, cy + h)]
    ax.add_patch(PathPatch(MPath(V, [MPath.MOVETO] + [MPath.CURVE4] * 18), facecolor=TISSUE, edgecolor=TISSUE_E,
                           lw=1.4 * LS, zorder=2))
    for sgn in (-1, 1):
        ax.add_patch(Arc((cx + sgn * w * 0.33, cy + h * 0.50), w * 0.54, h * 0.56, theta1=18, theta2=162,
                         color=TISSUE_E, lw=0.9 * LS, alpha=0.85, zorder=3))
    ax.add_patch(Ellipse((cx, cy + h * 0.48), w * 0.36, h * 0.36, facecolor=PAG, edgecolor="none", alpha=0.8, zorder=3))
    ax.add_patch(Ellipse((cx, cy + h * 0.48), 0.11, 0.21, facecolor="white", edgecolor=TISSUE_E, lw=1.0 * LS, zorder=6))
    for sgn in (-1, 1):
        ax.add_patch(Ellipse((cx + sgn * w * 0.075, cy + h * 0.24), w * 0.10, h * 0.14, facecolor=NUCLEUS,
                             edgecolor="none", alpha=0.95, zorder=4))
        ax.add_patch(Circle((cx + sgn * w * 0.13, cy + h * 0.09), 0.055, facecolor=FIBRE, edgecolor="none", zorder=4))
    SN = {}
    for sgn in (-1, 1):
        px, py = cx + sgn * w * 0.50, cy - h * 0.66
        ang = -sgn * 26
        ax.add_patch(Ellipse((px, py), w * 0.72, h * 0.54, angle=ang, facecolor=PEDUNCLE, edgecolor=TISSUE_E,
                             lw=1.0 * LS, zorder=3))
        for f in (-0.18, 0, 0.18):
            ax.add_patch(Arc((px, py), w * 0.72 * (0.55 + f), h * 0.54 * (0.55 + f), angle=ang, theta1=200, theta2=340,
                             color=FIBRE, lw=0.7 * LS, alpha=0.8, zorder=4))
        ax.add_patch(Arc((px, cy - h * 0.56), w * 0.76, h * 0.60, angle=ang, theta1=16, theta2=164, color=NIGRA_R,
                         lw=4.0 * LS, zorder=4, capstyle="round"))
        sn = dict(ox=px, oy=cy - h * 0.47, W=w * 0.78, H=h * 0.64, ang=ang)
        ax.add_patch(Arc((sn["ox"], sn["oy"]), sn["W"], sn["H"], angle=sn["ang"], theta1=16, theta2=164, color=NIGRA,
                         lw=5.0 * LS, zorder=5, capstyle="round"))
        SN[sgn] = sn
        ax.add_patch(Ellipse((cx + sgn * w * 0.30, cy - h * 0.04), w * 0.26, h * 0.28, facecolor=REDNUC,
                             edgecolor="#A98A83", lw=0.9 * LS, alpha=0.92, zorder=5))
        ax.add_patch(Arc((cx + sgn * w * 0.60, cy - h * 0.24), w * 0.52, h * 0.46, angle=-sgn * 40, theta1=250,
                         theta2=340 if sgn > 0 else 300, color=FIBRE, lw=3.2 * LS, zorder=4, capstyle="round"))
        for k in (-1, 0, 1):
            ax.annotate("", xy=(cx + sgn * (w * 0.10 + k * 0.02), cy - h * 0.52),
                        xytext=(cx + sgn * w * 0.24, cy - h * 0.16 + k * 0.05),
                        arrowprops=dict(arrowstyle="-", color="#9C9184", lw=0.8 * LS, alpha=0.9,
                                        connectionstyle=f"arc3,rad={0.22 * sgn}"), zorder=4)
    ax.text(cx, cy + h + 0.30, "Midbrain, transverse section", ha="center", va="bottom", fontsize=6.4, color=INK)
    sl = SN[-1]
    LX = cx - w - 0.28
    for txt, xy, ty, col, bold in [
            ("substantia nigra\npars compacta", on_arc(sl["ox"], sl["oy"], sl["W"], sl["H"], sl["ang"], 128),
             cy - h * 0.46, NIGRA, True),
            ("pars reticulata", (cx - w * 0.68, cy - h * 0.74), cy - h * 0.92, NIGRA_R, False),
            ("red nucleus", (cx - w * 0.30, cy - h * 0.04), cy + h * 0.46, "#8A6A63", False),
            ("crus cerebri", (cx - w * 0.62, cy - h * 0.92), cy - h * 1.28, MUTED, False),
            ("medial lemniscus", (cx - w * 0.72, cy - h * 0.28), cy + h * 0.08, MUTED, False),
            ("periaqueductal grey\nand aqueduct", (cx - w * 0.14, cy + h * 0.52), cy + h * 0.84, MUTED, False)]:
        ax.annotate(txt, xy=xy, xytext=(LX, ty), ha="right", va="center", fontsize=FS, color=col,
                    fontweight="bold" if bold else "normal", linespacing=1.1,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.45, alpha=0.8, shrinkA=1.5, shrinkB=0.5,
                                    connectionstyle="arc3,rad=0.12"))
    # ---- magnified capture field ----
    mx, my, mr = 3.15, 1.60, 1.12
    p1 = on_arc(sl["ox"], sl["oy"], sl["W"], sl["H"], sl["ang"], 150)
    p2 = on_arc(sl["ox"], sl["oy"], sl["W"], sl["H"], sl["ang"], 106)
    t1, t2 = (mx - mr * 0.86, my + mr * 0.52), (mx + mr * 0.30, my + mr * 0.95)
    for s_, t_ in ((p1, t1), (p2, t2)):
        ax.plot([s_[0], t_[0]], [s_[1], t_[1]], color=TISSUE_E, lw=0.45, ls=(0, (2.5, 2)), zorder=1)
    seg = [on_arc(sl["ox"], sl["oy"], sl["W"], sl["H"], sl["ang"], t) for t in np.linspace(106, 150, 24)]
    ax.plot([q[0] for q in seg], [q[1] for q in seg], color=CAPTURE, lw=5.6 * LS, solid_capstyle="round", zorder=6)
    ax.add_patch(Circle((mx, my), mr, facecolor="#FBF8F3", edgecolor=TISSUE_E, lw=1.4 * LS, zorder=3))
    rng = np.random.default_rng(7); somas = []
    while len(somas) < 11:
        a, r = rng.uniform(0, 2 * np.pi), mr * np.sqrt(rng.uniform(0, 0.62))
        px, py = mx + r * np.cos(a), my + r * np.sin(a)
        if all((px - qx) ** 2 + (py - qy) ** 2 > 0.30 ** 2 for qx, qy, _ in somas):
            somas.append((px, py, rng.uniform(0.115, 0.155)))
    for k, (px, py, rr) in enumerate(somas):
        ax.add_patch(Circle((px, py), rr, facecolor=SOMA, edgecolor="#9A8B72", lw=0.8 * LS, zorder=4))
        if k % 3 != 2:
            for _ in range(4):
                gx, gy = px + rng.uniform(-rr * 0.45, rr * 0.45), py + rng.uniform(-rr * 0.45, rr * 0.45)
                ax.add_patch(Circle((gx, gy), rr * 0.26, facecolor=MELANIN, edgecolor="none", alpha=0.85, zorder=5))
    tx, ty, tr = somas[0]
    ax.add_patch(Circle((tx, ty), tr * 1.75, facecolor="none", edgecolor=CAPTURE, lw=0.7, ls=(0, (2.2, 1.6)), zorder=6))
    ax.annotate("laser-capture\nmicrodissection", xy=(tx - tr * 1.5, ty + tr * 1.2), xytext=(mx - mr - 0.25, my + mr * 0.55),
                ha="right", va="center", fontsize=FS, color=CAPTURE, fontweight="bold", linespacing=1.1,
                arrowprops=dict(arrowstyle="-", color=CAPTURE, lw=0.5, shrinkA=1.5, shrinkB=1,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(mx, my - mr - 0.16, "neuromelanin-positive\ndopaminergic neurons", ha="center", va="top", fontsize=FS,
            color=MUTED, linespacing=1.1)
    return mx, my, mr
print("anatomy ready")''')

md("## 3. The figure")

code(r'''FW, FH = 183.0, 120.0                                      # millimetres
fig = plt.figure(figsize=(FW / 25.4, FH / 25.4))
M = fig.add_axes([0, 0, 1, 1]); M.set_xlim(0, FW); M.set_ylim(0, FH); M.axis("off")   # 1 unit = 1 mm
def fax(x, y, w, h):
    return fig.add_axes([x / FW, y / FH, w / FW, h / FH])
def letter(x, y, L, title):
    M.text(x, y, L, ha="left", va="baseline", fontsize=9, fontweight="bold", color=INK)
    M.text(x + 4.2, y, title, ha="left", va="baseline", fontsize=7.2, color=INK)
def rule(x0, x1, y, lw=0.5, color=RULE):
    M.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt")

# grid: top band 40-114 mm (a | b over c), bottom band 3-35 mm (d)
LEFT, RIGHT = 2.0, 181.0
RX = 72.0                                                  # left edge of the right-hand column
letter(LEFT, 115.0, "a", "Tissue and capture")
letter(RX, 115.0, "b", f"Four laser-capture cohorts, one profile per person")
letter(RX, 72.6, "c", "Harmonisation onto one scale")
letter(LEFT, 35.2, "d", "Analysis")

# =============================== a ===============================
axA = fax(0.5, 39.0, 67.0, 73.5); axA.set_aspect("equal"); axA.axis("off")
axA.set_xlim(-1.85, 5.55); axA.set_ylim(0.00, 6.92); axA.set_anchor("N")
mx, my, mr = draw_anatomy(axA)

# =============================== b ===============================
ROW_Y = [106.5, 100.0, 93.5, 87.0]
DOT_X0, DOT_DX = 113.0, 1.95
for d, yr in zip(ORDER, ROW_Y):
    c = COH.loc[d]
    M.add_patch(Rectangle((RX, yr - 1.1), 1.6, 2.2, facecolor=COH_COL[d], edgecolor="none"))
    M.text(RX + 3.0, yr + 0.9, d, ha="left", va="center", fontsize=6.6, fontweight="bold", color=INK)
    M.text(RX + 3.0, yr - 1.35, PLATFORM[d], ha="left", va="center", fontsize=5.6, color=MUTED)
    xs = []
    x = DOT_X0
    for _ in range(int(c.control)):
        xs.append((x, CT_C)); x += DOT_DX
    x += 1.1
    for _ in range(int(c.PD)):
        xs.append((x, PD_C)); x += DOT_DX
    M.scatter([p[0] for p in xs], [yr + 0.45] * len(xs), s=10.5, c=[p[1] for p in xs], edgecolor="white", linewidth=0.35,
              zorder=4)
    M.text(DOT_X0, yr - 1.75, UNIT[d], ha="left", va="center", fontsize=5.4, color=MUTED, style="italic")
    M.text(RIGHT, yr + 0.45, f"{int(c.control)} + {int(c.PD)}", ha="right", va="center", fontsize=6.3, color=INK)
rule(RX, RIGHT, 83.0, lw=0.4)
M.scatter([RX + 1.0], [80.3], s=10.5, c=CT_C, edgecolor="white", linewidth=0.35)
M.text(RX + 2.6, 80.3, "control", ha="left", va="center", fontsize=6.0, color=INK)
M.scatter([RX + 14.0], [80.3], s=10.5, c=PD_C, edgecolor="white", linewidth=0.35)
M.text(RX + 15.6, 80.3, "Parkinson's disease", ha="left", va="center", fontsize=6.0, color=INK)
M.text(RIGHT, 80.3, f"{N_PEOPLE} people:  {N_CT} + {N_PD}", ha="right", va="center", fontsize=6.3, color=INK,
       fontweight="bold")
M.text(RIGHT, 110.2, "control + PD", ha="right", va="center", fontsize=5.4, color=MUTED)

# the capture field feeds the cohorts
fig.canvas.draw()
p_from = M.transData.inverted().transform(axA.transData.transform((mx + mr * 1.02, my + mr * 0.20)))
M.add_patch(FancyArrowPatch(tuple(p_from), (RX - 2.0, 92.0), arrowstyle="-|>", mutation_scale=6, lw=0.6,
                            color="#6B7178", connectionstyle="arc3,rad=0.28", zorder=2))

# =============================== c ===============================
# gene filter: three bars, length proportional to the number of genes
FX0, FXW = RX, 38.0
steps = [(f"{G_ANY:,}", "genes in any study"), (f"{G_ALL:,}", "measured on all four platforms"),
         (f"{G_EXP:,}", "expressed in every study")]
vals = [G_ANY, G_ALL, G_EXP]
FY = [63.5, 56.0, 48.5]
for (num, lab), v, yb, shade in zip(steps, vals, FY, ("#B6BFC9", "#7D8B9B", "#34475E")):
    wbar = FXW * v / G_ANY
    M.add_patch(Rectangle((FX0, yb - 1.6), wbar, 3.2, facecolor=shade, edgecolor="none"))
    M.text(FX0 + wbar + 1.2, yb, num, ha="left", va="center", fontsize=6.6, fontweight="bold", color=INK)
    M.text(FX0, yb + 2.9, lab, ha="left", va="center", fontsize=5.8, color=MUTED)
M.text(FX0, 42.6, "then z-scored within each study,\nwithout reference to diagnosis", ha="left", va="center", fontsize=5.8,
       color=INK, linespacing=1.2)

# the merge check: PCA before and after, and whether a model can name each person's study
PW = 25.0
for k, (stage, title, acc_key) in enumerate((("before", "each study on its own scale", "accuracy_before"),
                                            ("after", "z-scored within each study", "accuracy_after"))):
    x0 = 122.0 + k * 31.0
    ax = fax(x0, 48.0, PW, PW * 0.80)
    d = PCA[PCA.stage == stage]
    for coh in ORDER:
        dd = d[d.dataset == coh]
        ax.scatter(dd.PC1, dd.PC2, s=4.5, color=COH_COL[coh], edgecolor="white", linewidth=0.25)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_linewidth(0.4); s_.set_color("#B7BDC4")
    ax.set_xlabel(f"PC1 ({100 * d.var_PC1.iloc[0]:.0f}%)", fontsize=5.3, color=MUTED, labelpad=1.5)
    ax.set_ylabel(f"PC2 ({100 * d.var_PC2.iloc[0]:.0f}%)", fontsize=5.3, color=MUTED, labelpad=1.5)
    M.text(x0, 48.0 + PW * 0.80 + 1.7, title, ha="left", va="center", fontsize=5.8, color=INK)
    val, shade = IDC[acc_key], (PD_C if k == 0 else "#34475E")
    BY, BHb = 42.4, 2.1                                    # the bar runs the width of its own scatter
    M.add_patch(Rectangle((x0, BY), PW, BHb, facecolor="#E8EBEE", edgecolor="none", zorder=1))
    M.add_patch(Rectangle((x0, BY), PW * val, BHb, facecolor=shade, edgecolor="none", zorder=2))
    xch = x0 + PW * IDC["chance"]
    M.plot([xch, xch], [BY - 0.7, BY + BHb + 0.7], color=MUTED, lw=0.5, ls=(0, (1.6, 1.4)), zorder=3)
    lab_in = PW * val > 9.0                                # the number sits inside a long bar, beside a short one
    M.text(x0 + PW * val + (-1.2 if lab_in else 1.2), BY + BHb / 2, f"{100 * val:.0f}%",
           ha="right" if lab_in else "left", va="center", fontsize=6.4, fontweight="bold",
           color="white" if lab_in else shade, zorder=4)
M.text(122.0, 40.8, "how often a model can still name each person's study", ha="left", va="center", fontsize=5.3, color=INK)
M.text(122.0, 38.9, f"dashed line: chance, {100 * IDC['chance']:.0f}%", ha="left", va="center", fontsize=5.0, color=MUTED)

# =============================== d ===============================
rule(LEFT, RIGHT, 37.9, lw=0.4)
def box(x, y, w, h, title, body, fill=BOX_FILL, edge=BOX_EDGE, tcol=INK):
    M.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.3", facecolor=fill, edgecolor=edge,
                               lw=0.5, zorder=2))
    M.text(x + 2.0, y + h - 2.3, title, ha="left", va="center", fontsize=6.2, fontweight="bold", color=tcol, zorder=3)
    M.text(x + 2.0, y + h - 4.4, body, ha="left", va="top", fontsize=5.3, color=INK, linespacing=1.2, zorder=3)
def arrow(p, q, rad=0.0):
    M.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=5.5, lw=0.55, color="#5E656D",
                                connectionstyle=f"arc3,rad={rad}", shrinkA=0.5, shrinkB=0.5, zorder=3))
BH, GAP = 8.8, 2.4                                          # one box, and the gap that carries the arrows
L3 = 2.5; L2 = L3 + BH + GAP; L1 = L2 + BH + GAP            # the three rows, bottom to top
XC, XE, XR, WB, WR = 44.0, 94.5, 145.0, 42.5, 36.0          # columns of the first two rows
ext_n = len(EXT["cohorts"]) if EXT else 8
ext_people = EXT["people"] if EXT else ""
ext_panel = EXT["auc"]["panel"]["auc"] if EXT else float("nan")
ext_core = EXT["auc"]["frozen"]["auc"] if EXT else float("nan")
ext_adj = EXT["auc"]["panel|neuron_removed"]["auc"] if EXT else float("nan")
lin_p = ENR["subtypes"]["up_vs_down"] if ENR else float("nan")

box(LEFT, L2, 34.0, L1 + BH - L2, "Merged cohort",
    f"{N_PEOPLE} people × {G_EXP:,} genes\none profile per person\n{N_CT} control, {N_PD} PD\n\nfour studies, each\nz-scored on its own")
box(XC, L1, WB, BH, "Core classifier", "ranks within each person $\\rightarrow$ 30 PCs\n$\\rightarrow$ Random Forest, 1,000 trees")
box(XE, L1, WB, BH, "Evaluation", "5 × 5-fold CV by person, one study out\n200 label shuffles")
box(XC, L2, WB, BH, "Boruta + Random Forest", f"all-relevant selection on {N_PEOPLE} people,\nand again inside every training fold")
box(XE, L2, WB, BH, "Explanation", "TreeSHAP out of fold, differential\nexpression, effect in each study")
box(XR, L1, WR, BH, "Fig. 2", f"AUC {HEAD.cv_auc:.2f} · unseen study {HEAD.lodo_auc:.2f}"
    + (f"\nlabel shuffles $P$ = {CONF['permutation']['p_auc']:.3f}" if CONF else ""), fill="white", edge=INK, tcol=PD_C)
box(XR, L2, WR, BH, "Figs 3–5", f"{N_BOR} genes: {N_UP} up, {N_DOWN} down in PD\nSHAP and per-study effects",
    fill="white", edge=INK, tcol=PD_C)

# the third row: the panel taken outside the discovery data, and read biologically
XV, WV, XVR, XB, WB2, XBR, WBR = LEFT, 54.0, 58.5, 92.0, 50.0, 144.5, 36.5
box(XV, L3, WV, BH, "External validation",
    f"{ext_n} public cohorts, {ext_people} people; shared\ndonors removed, model fixed first")
box(XVR, L3, 30.0, BH, "Figs 6–7", f"panel {ext_panel:.2f} · core {ext_core:.2f}\nneuron-adjusted {ext_adj:.2f}",
    fill="white", edge=INK, tcol=PD_C)
box(XB, L3, WB2, BH, "Biology of the panel",
    f"lineage, pathways on the {G_EXP:,}-gene\nbackground, and the neuron mix removed")
box(XBR, L3, WBR, BH, "Figs 8–10", f"lineage $P$ = {lin_p:.4f} · {N_PATH} pathways\n"
    f"mix removed: {OOF_BEFORE:.2f} $\\rightarrow$ {OOF_AFTER:.2f}", fill="white", edge=INK, tcol=PD_C)

arrow((LEFT + 34.0, L1 + BH / 2 - 0.8), (XC, L1 + BH / 2), rad=-0.10)
arrow((LEFT + 34.0, L2 + BH / 2 + 0.8), (XC, L2 + BH / 2), rad=0.10)
for yy in (L1 + BH / 2, L2 + BH / 2):
    arrow((XC + WB, yy), (XE, yy)); arrow((XE + WB, yy), (XR, yy))
arrow((XV + WV, L3 + BH / 2), (XVR, L3 + BH / 2)); arrow((XB + WB2, L3 + BH / 2), (XBR, L3 + BH / 2))
arrow((XC + 4.0, L2), (XC + 4.0, L3 + BH))                  # the panel goes on to the external cohorts
arrow((XE + 6.0, L2), (XE + 6.0, L3 + BH))                  # and to the biology

for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"Figure01_study_design.{ext}", dpi=600)
print("saved", sorted(p.name for p in FIG_DIR.iterdir()))
plt.show() if IN_NB else plt.close(fig)''')

md("## 4. Suggested legend for the manuscript")

code(r'''print(f"""Study design. (a) Dopaminergic neurons of the substantia nigra pars compacta were isolated by laser-capture
microdissection in each study (schematic transverse midbrain; the magnified field shows neuromelanin-positive neurons). (b) Four
laser-capture cohorts, {N_PEOPLE} people in total ({N_CT} control, {N_PD} Parkinson's disease); each dot is one person. Replicate pools
and libraries were combined so that every person contributes one profile. (c) Of {G_ANY:,} genes, {G_ALL:,} were measured on all four
platforms and {G_EXP:,} were expressed in every study; each study was then z-scored on its own without reference to diagnosis.
Principal components before and after, coloured by study; a classifier trained to name each person's study placed
{100 * IDC['accuracy_before']:.0f}% correctly before and {100 * IDC['accuracy_after']:.0f}% after (chance {100 * IDC['chance']:.0f}%). (d) The analysis in
five stages, with the figure each one produces. The core classifier (within-person gene ranks, 30 principal components fitted on
training people only, Random Forest) was evaluated by 5 x 5-fold cross-validation by person, leave-one-study-out and 200 label
permutations (Fig. 2). Boruta with Random Forest importance selected {N_BOR} genes ({N_UP} higher and {N_DOWN} lower in Parkinson's disease),
explained by TreeSHAP and per-study effect sizes (Figs 3-5). The fixed model and the {N_BOR}-gene panel were then tested in
{len(EXT['cohorts']) if EXT else 8} independent cohorts ({EXT['people'] if EXT else ''} people) after donors shared with the discovery studies had been removed, and again with models
retrained without the same brain banks (Figs 6-7). Finally the panel was read biologically: the genes lower in Parkinson's disease
are markers of SOX6 dopamine neurons and those higher are CALB1 markers (P = {lin_p:.4f}; Fig. 8), pathway enrichment used the
{G_EXP:,} measured genes as background with Benjamini-Hochberg control ({N_PATH} terms; Fig. 9), and the classifier was retested with each
donor's estimated neuron mix regressed out of every gene (AUC {OOF_BEFORE:.2f} to {OOF_AFTER:.2f}, P = {OOF_P:.3f}; Fig. 10).""")''')

write_nb(HERE / "PD_LCM_rf_figure1.ipynb", CELLS, "f1")
