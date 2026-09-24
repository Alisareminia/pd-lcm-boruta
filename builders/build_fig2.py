"""Builds PD_LCM_rf_figure2.ipynb - secondary notebook: Figure 2, how well the core classifier works.

Reads only finished outputs: pd-lcm-rf-core (fold-by-fold scores of the core model and of a plain
forest), pd-lcm-rf-boruta-panel (every model on the same folds) and pd-lcm-rf-confirm (model choice
inside folds, the label-permutation null). Nothing is refitted.

Print size (183 x 118 mm) on a millimetre canvas, in the type and palette of Figure 1.
  a - ROC of the core Random Forest: five repeats and their average, a plain forest for reference;
      the classification metrics beneath it as a small ruled table
  b - the core model against simpler forests on the same folds
  c - every person's out-of-fold probability of PD, by study
  d - leave one study out
  e - random labels
"""
import pathlib
from kaggle_nb import write_nb
HERE = pathlib.Path(__file__).parent
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md("""# Figure 2 - how well the core classifier works

Reads **`pd-lcm-rf-core`**, **`pd-lcm-rf-boruta-panel`** and **`pd-lcm-rf-confirm`**; nothing is refitted.
Print size, 183 x 118 mm, in the style of Figure 1.

* **a** ROC of the core Random Forest (five repeats of 5-fold cross-validation by person), with its classification metrics
* **b** the core model against simpler forests, scored on the same folds
* **c** every person's out-of-fold probability of PD, by study
* **d** leave one study out - trained on three studies, tested on the fourth
* **e** random labels - the whole procedure re-run on 200 shuffles of the diagnoses""")

code(r'''import os, glob, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
try:
    get_ipython(); IN_NB = True
except NameError:
    IN_NB = False; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_curve, roc_auc_score
warnings.filterwarnings("ignore")

ON_KAGGLE = Path("/kaggle/input").exists()
FIG_DIR = Path("/kaggle/working/figures") if ON_KAGGLE else Path(os.environ.get("SMOKE_OUT", "smoke_fig")) / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SOURCES = {"core": ("rf-core", "outputs_rf/core"), "panel": ("boruta-panel", "outputs_rf/panel"),
           "confirm": ("rf-confirm", "kaggle_confirm")}
def find_any(pattern, source):
    """The file from one named notebook - several inputs carry files with the same name."""
    roots = ["/kaggle/input"] if ON_KAGGLE else os.environ.get("FIG_EXTRA", ".").split(":")
    hits = [h for r in roots for h in glob.glob(f"{r}/**/{pattern}", recursive=True)]
    hits = sorted((h for h in hits if any(k in h for k in SOURCES[source])), key=len)
    if not hits:
        raise FileNotFoundError(f"{pattern} from {source}")
    return hits[0]

# ---- the article's type and palette (Figure 1) -------------------------------------------------------
INK, MUTED, FAINT, RULE = "#1B1D20", "#5E656D", "#9AA1A9", "#C9CED4"
PD_C, CT_C = "#7A2533", "#6F829A"                          # oxblood = PD, slate = control
CORE = "#23324A"                                           # the core model
REF = "#9AA1A9"                                            # the comparison forests
COH_COL = {"GSE182622": "#2F4A6B", "GSE20141": "#A8793A", "GSE24378": "#5B8577", "GSE169755": "#6E567E"}
FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "Nimbus Sans", "FreeSans", "DejaVu Sans"]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT_STACK, "font.size": 6.5,
                     "axes.linewidth": 0.5, "axes.edgecolor": "#30343A", "axes.labelcolor": INK, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2.2, "ytick.major.size": 2.2, "xtick.major.pad": 1.8, "ytick.major.pad": 1.8,
                     "xtick.color": "#30343A", "ytick.color": "#30343A",
                     "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150,
                     "savefig.facecolor": "white", "figure.facecolor": "white"})
print("Kaggle" if ON_KAGGLE else "LOCAL SMOKE RUN")''')

md("## 1. The numbers the figure states")

code(r'''OOF  = pd.read_csv(find_any("core_oof_scores.csv", "core"))            # one row per person, in matrix order
REC  = json.load(open(find_any("core_fold_records.json", "core")))
PERF = pd.read_csv(find_any("performance_all_models.csv", "panel")).set_index("model")
CONF = json.load(open(find_any("S10_confirmation.json", "confirm")))
NULL = pd.read_csv(find_any("S10_permutation_null.csv", "confirm"))
y = OOF.y.to_numpy(int); DS = OOF.dataset.to_numpy(str)
ORDER = ["GSE182622", "GSE20141", "GSE24378", "GSE169755"]
HEAD_NAME = "Ranks + PCA 30 + Random Forest"
H = PERF.loc[HEAD_NAME]

cv = [r for r in REC if r["kind"] == "cv"]
REPS = sorted({r["rep"] for r in cv})
def rep_scores(key, rep):
    rr = [r for r in cv if r["rep"] == rep]
    return np.concatenate([r["test"] for r in rr]), np.concatenate([r[key]["score"] for r in rr])
FGRID = np.linspace(0, 1, 401)
def roc_reps(key):
    out = []
    for rep in REPS:
        idx, s_ = rep_scores(key, rep)
        f, t, _ = roc_curve(y[idx], s_); out.append((f, t))
    return out
def vertical_average(curves):
    """Mean true-positive rate at each false-positive rate across repeats."""
    return np.mean([np.interp(FGRID, f, t) for f, t in curves], axis=0)
ROC_REPS, ROC_RF = roc_reps("head"), roc_reps("rf_all")
T_HEAD, T_RF = vertical_average(ROC_REPS), vertical_average(ROC_RF)
def mean_oof(key):
    acc, cnt = np.zeros(len(y)), np.zeros(len(y))
    for r in cv:
        acc[r["test"]] += r[key]["score"]; cnt[r["test"]] += 1
    return acc / np.maximum(cnt, 1)
S_HEAD, S_RF = mean_oof("head"), mean_oof("rf_all")
AUC_MEAN, AUC_RF = roc_auc_score(y, S_HEAD), roc_auc_score(y, S_RF)
STUDY_AUC = {d: roc_auc_score(y[DS == d], S_HEAD[DS == d]) for d in ORDER}
LODO = {d: float(H[f"lodo_auc_{d}"]) for d in ORDER}
print(f"core: CV AUC {H.cv_auc:.3f} +/- {H.cv_auc_sd:.3f}, averaged-score AUC {AUC_MEAN:.3f}, plain forest {AUC_RF:.3f}; "
      f"LODO {H.lodo_auc:.3f} " + str({k: round(v, 2) for k, v in LODO.items()}))
print("per study (out of fold):", {k: round(v, 2) for k, v in STUDY_AUC.items()})
print(f"nested {CONF['nested']['auc']:.3f}; permutation observed {CONF['permutation']['observed_auc']:.3f}, "
      f"null mean {NULL.auc.mean():.3f}, P = {CONF['permutation']['p_auc']:.4f}")''')

md("## 2. The figure")

code(r'''FW, FH = 183.0, 118.0
fig = plt.figure(figsize=(FW / 25.4, FH / 25.4))
M = fig.add_axes([0, 0, 1, 1]); M.set_xlim(0, FW); M.set_ylim(0, FH); M.axis("off")   # 1 unit = 1 mm
def fax(x, y_, w, h):
    return fig.add_axes([x / FW, y_ / FH, w / FW, h / FH])
def letter(x, y_, L, title):
    M.text(x, y_, L, ha="left", va="baseline", fontsize=9, fontweight="bold", color=INK)
    M.text(x + 4.2, y_, title, ha="left", va="baseline", fontsize=7.2, color=INK)
def rule(x0, x1, y_, lw=0.5, color=RULE):
    M.plot([x0, x1], [y_, y_], color=color, lw=lw, solid_capstyle="butt")
LEFT, RIGHT, RX = 2.0, 181.0, 76.0
TOPY = 114.0

# =============================== a: ROC =====================================
letter(LEFT, TOPY, "a", "Out-of-fold ROC of the core classifier")
axR = fax(12.0, 57.0, 50.0, 50.0)
axR.plot([0, 1], [0, 1], color="#B7BDC4", lw=0.5, ls=(0, (2, 2)))
for f, t in ROC_REPS:
    axR.step(f, t, where="post", color="#A9B3C1", lw=0.45, alpha=0.9)
axR.plot(FGRID, T_RF, color=REF, lw=0.8, ls=(0, (3, 1.6)))
axR.fill_between(FGRID, 0, T_HEAD, color=CORE, alpha=0.06, lw=0)
axR.plot(FGRID, T_HEAD, color=CORE, lw=1.3)
axR.set_xlim(-0.01, 1.01); axR.set_ylim(-0.01, 1.01); axR.set_aspect("equal")
axR.set_xticks([0, 0.5, 1]); axR.set_yticks([0, 0.5, 1]); axR.set_xticklabels(["0", "0.5", "1"]); axR.set_yticklabels(["0", "0.5", "1"])
axR.set_xlabel("False-positive rate", fontsize=6.5, labelpad=2); axR.set_ylabel("True-positive rate", fontsize=6.5, labelpad=2)
axR.text(0.97, 0.20, f"AUC {H.cv_auc:.2f} ± {H.cv_auc_sd:.2f}", transform=axR.transAxes, ha="right", va="center",
         fontsize=7.4, fontweight="bold", color=CORE)
axR.text(0.97, 0.12, "core classifier, 5 × 5-fold CV", transform=axR.transAxes, ha="right", va="center", fontsize=5.8, color=MUTED)
axR.text(0.97, 0.05, f"plain forest on all genes {PERF.loc['Random Forest, all genes', 'cv_auc']:.2f}",
         transform=axR.transAxes, ha="right", va="center", fontsize=5.8, color=REF)
# key to the curves
for i, (lab, col, ls, lw) in enumerate((("mean of the five repeats", CORE, "-", 1.3), ("each repeat", "#A9B3C1", "-", 0.6),
                                        ("plain forest, all genes", REF, (0, (3, 1.6)), 0.8))):
    yy = 0.43 - i * 0.065
    axR.plot([0.47, 0.55], [yy, yy], color=col, ls=ls, lw=lw, transform=axR.transAxes)
    axR.text(0.575, yy, lab, transform=axR.transAxes, ha="left", va="center", fontsize=5.8, color=INK)

# classification metrics as a small ruled table
rows_t = [("Accuracy", f"{H.cv_accuracy:.2f} ± {H.cv_accuracy_sd:.2f}"), ("Balanced accuracy", f"{H.cv_bal_accuracy:.2f}"),
          ("Sensitivity (PD)", f"{H.cv_sensitivity:.2f}"), ("Specificity (control)", f"{H.cv_specificity:.2f}"),
          ("Matthews correlation", f"{H.cv_mcc:.2f}"), ("Brier score", f"{H.cv_brier:.2f}"),
          ("Model choice inside folds, AUC", f"{CONF['nested']['auc']:.2f} ± {CONF['nested']['auc_sd']:.2f}")]
TX0, TX1, TY = 12.0, 64.0, 44.5
rule(TX0, TX1, TY, lw=0.6, color="#3A3F45")
M.text(TX0, TY - 2.3, "at a 0.5 cut, mean of five repeats", ha="left", va="center", fontsize=5.6, color=MUTED)
rule(TX0, TX1, TY - 4.3, lw=0.35, color="#8C9299")
for i, (lab, val) in enumerate(rows_t):
    yy = TY - 7.0 - i * 4.1
    M.text(TX0, yy, lab, ha="left", va="center", fontsize=6.3, color=INK)
    M.text(TX1, yy, val, ha="right", va="center", fontsize=6.3, color=INK)
rule(TX0, TX1, TY - 7.0 - (len(rows_t) - 1) * 4.1 - 2.6, lw=0.6, color="#3A3F45")

# =============================== b: against simpler forests =====================================
letter(RX, TOPY, "b", "The core classifier against simpler forests, same folds")
MODELS = [(HEAD_NAME, "gene ranks $\\rightarrow$ 30 components", True), ("Boruta (perc 99) + Random Forest", "Boruta genes, perc 99", False),
          ("Random Forest, all genes", "all 5,622 genes", False), ("Boruta (perc 100) + Random Forest", "Boruta genes, perc 100", False)]
BY = [104.0, 98.5, 93.0, 87.5]
LO, HI = 0.4, 0.8
COLS = {"cv": (113.0, 139.0), "lodo": (148.0, 168.0)}
def sx(v, col):
    a, b = COLS[col]; return a + (np.clip(v, LO, HI) - LO) / (HI - LO) * (b - a)
for col, head in (("cv", "cross-validated AUC"), ("lodo", "unseen study AUC")):
    a, b = COLS[col]
    for v in (0.4, 0.5, 0.6, 0.7, 0.8):
        M.plot([sx(v, col)] * 2, [84.8, 107.2], color="#E6E8EB" if v != 0.5 else "#B9BFC6", lw=0.4,
               ls="-" if v != 0.5 else (0, (2, 1.5)), zorder=0)
        if v in (0.4, 0.6, 0.8):
            M.text(sx(v, col), 83.0, f"{v:g}", ha="center", va="center", fontsize=5.6, color=MUTED)
    M.text((a + b) / 2, 109.2, head, ha="center", va="center", fontsize=5.9, color=INK)
M.text(RIGHT, 109.2, "accuracy", ha="right", va="center", fontsize=5.9, color=INK)
M.text(RX, 109.2, "features given to the Random Forest", ha="left", va="center", fontsize=5.9, color=INK)
for (name, lab, core), yy in zip(MODELS, BY):
    r = PERF.loc[name]
    col = CORE if core else "#6B7178"
    M.text(RX, yy, lab, ha="left", va="center", fontsize=6.3, color=col, fontweight="bold" if core else "normal")
    lo_, hi_ = sx(r.cv_auc - r.cv_auc_sd, "cv"), sx(r.cv_auc + r.cv_auc_sd, "cv")
    M.plot([lo_, hi_], [yy, yy], color=col, lw=0.7, solid_capstyle="butt", zorder=2)
    M.scatter([sx(r.cv_auc, "cv")], [yy], s=16 if core else 11, color=col, edgecolor="white", linewidth=0.4, zorder=3)
    M.scatter([sx(r.lodo_auc, "lodo")], [yy], s=16 if core else 11, color=col, edgecolor="white", linewidth=0.4, zorder=3,
              marker="D")
    M.text(RIGHT, yy, f"{r.cv_accuracy:.2f}", ha="right", va="center", fontsize=6.3, color=col,
           fontweight="bold" if core else "normal")
rule(RX, RIGHT, 80.2, lw=0.4)

# =============================== c: every person =====================================
letter(RX, 74.8, "c", "Every person, scored by models that never saw them")
CY = [65.5, 59.5, 53.5, 47.5]
PX0, PX1 = 104.0, 168.0
px = lambda p: PX0 + p * (PX1 - PX0)
M.plot([px(0.5)] * 2, [44.2, 68.8], color="#B9BFC6", lw=0.45, ls=(0, (2, 1.5)), zorder=0)
rng = np.random.default_rng(3)
for d, yy in zip(ORDER, CY):
    M.add_patch(Rectangle((RX, yy - 1.1), 1.6, 2.2, facecolor=COH_COL[d], edgecolor="none"))
    M.text(RX + 3.0, yy, d, ha="left", va="center", fontsize=6.3, color=INK)
    M.plot([PX0, PX1], [yy, yy], color="#E3E6EA", lw=0.5, zorder=0)
    m = DS == d
    for lab, off, colr in ((1, 1.05, PD_C), (0, -1.05, CT_C)):
        s = S_HEAD[m & (y == lab)]
        M.scatter(px(s), yy + off + rng.uniform(-0.35, 0.35, len(s)), s=6.5, color=colr, edgecolor="white", linewidth=0.3,
                  zorder=3)
    M.text(RIGHT, yy, f"{STUDY_AUC[d]:.2f}", ha="right", va="center", fontsize=6.3, color=INK)
for v in (0, 0.5, 1):
    M.text(px(v), 42.6, f"{v:g}", ha="center", va="center", fontsize=5.6, color=MUTED)
M.text((PX0 + PX1) / 2, 40.3, "out-of-fold probability of PD (averaged over five repeats)", ha="center", va="center",
       fontsize=5.9, color=INK)
M.text(RIGHT, 70.6, "AUC", ha="right", va="center", fontsize=5.9, color=INK)
M.scatter([RX + 1.0], [40.3], s=6.5, color=CT_C, edgecolor="white", linewidth=0.3)
M.text(RX + 2.5, 40.3, "control", ha="left", va="center", fontsize=5.9, color=INK)
M.scatter([RX + 12.5], [40.3], s=6.5, color=PD_C, edgecolor="white", linewidth=0.3)
M.text(RX + 14.0, 40.3, "PD", ha="left", va="center", fontsize=5.9, color=INK)
rule(RX, RIGHT, 37.3, lw=0.4)

# =============================== d: leave one study out =====================================
letter(RX, 32.2, "d", "Leave one study out")
DY = [25.0, 20.8, 16.6, 12.4]
DX0, DX1, DLO, DHI = 100.0, 124.0, 0.3, 1.0
dx = lambda v: DX0 + (np.clip(v, DLO, DHI) - DLO) / (DHI - DLO) * (DX1 - DX0)
M.plot([dx(0.5)] * 2, [5.0, 28.0], color="#B9BFC6", lw=0.45, ls=(0, (2, 1.5)), zorder=0)
for d, yy in zip(ORDER, DY):
    M.add_patch(Rectangle((RX, yy - 0.9), 1.4, 1.8, facecolor=COH_COL[d], edgecolor="none"))
    M.text(RX + 2.6, yy, f"{d} held out", ha="left", va="center", fontsize=5.9, color=INK)
    M.plot([dx(0.5), dx(LODO[d])], [yy, yy], color="#8C95A3", lw=0.7, solid_capstyle="butt")
    M.scatter([dx(LODO[d])], [yy], s=9, color=COH_COL[d], edgecolor="white", linewidth=0.3, zorder=3)
    M.text(dx(LODO[d]) + (1.2 if LODO[d] >= 0.5 else -1.2), yy, f"{LODO[d]:.2f}", ha="left" if LODO[d] >= 0.5 else "right",
           va="center", fontsize=5.6, color=MUTED)
yy = 7.6
M.text(RX + 2.6, yy, "all four, pooled", ha="left", va="center", fontsize=5.9, color=INK, fontweight="bold")
M.scatter([dx(H.lodo_auc)], [yy], s=16, color=CORE, marker="D", edgecolor="white", linewidth=0.4, zorder=3)
M.text(dx(H.lodo_auc) + 1.4, yy, f"{H.lodo_auc:.2f}", ha="left", va="center", fontsize=6.0, color=CORE, fontweight="bold")
for v in (0.3, 0.5, 0.7, 1.0):
    M.text(dx(v), 3.0, f"{v:g}", ha="center", va="center", fontsize=5.6, color=MUTED)

# =============================== e: random labels =====================================
letter(134.0, 32.2, "e", "Random labels")
axE = fax(138.0, 7.0, 43.0, 20.0)
obs, p = CONF["permutation"]["observed_auc"], CONF["permutation"]["p_auc"]
bins = np.linspace(0.2, 0.8, 31)
axE.hist(NULL.auc, bins=bins, color="#C3C9D0", edgecolor="white", linewidth=0.3)
top = axE.get_ylim()[1] * 1.55; axE.set_ylim(0, top)
axE.plot([obs, obs], [0, top * 0.97], color=CORE, lw=1.1)
axE.text(obs - 0.012, top * 0.93, f"real labels {obs:.2f}", ha="right", va="top", fontsize=5.9, color=CORE, fontweight="bold")
axE.text(obs - 0.012, top * 0.78, f"$P$ = {p:.3f}", ha="right", va="top", fontsize=5.9, color=CORE)
axE.text(0.205, top * 0.93, f"{len(NULL)} shuffles", ha="left", va="top", fontsize=5.6, color=MUTED)
axE.set_xlim(0.2, 0.8); axE.set_yticks([]); axE.spines["left"].set_visible(False)
axE.set_xticks([0.2, 0.5, 0.8]); axE.set_xlabel("AUC", fontsize=5.9, labelpad=1.5)

for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"Figure02_classifier_performance.{ext}", dpi=600)
print("saved", sorted(p_.name for p_ in FIG_DIR.iterdir()))
plt.show() if IN_NB else plt.close(fig)''')

md("## 3. Suggested legend for the manuscript")

code(r'''print(f"""Performance of the core classifier (within-person gene ranks, 30 principal components fitted on training people
only, Random Forest) in {len(y)} people. (a) Out-of-fold ROC over five repeats of 5-fold cross-validation by person (thin lines)
and for the scores averaged over repeats (bold); dashed, a Random Forest on all 5,622 genes. AUC {H.cv_auc:.2f} +/- {H.cv_auc_sd:.2f}
(mean +/- SD over repeats). Beneath, classification metrics at a 0.5 probability cut, and the AUC when the choice among the
six best model variants is itself made inside each training fold ({CONF['nested']['auc']:.2f}). (b) The core classifier against
Random Forests on Boruta-selected genes (selection redone inside every training fold) and on all genes, all on the same folds:
cross-validated AUC (dot, +/- SD), AUC on a study left out of training (diamond) and accuracy. (c) Each person's out-of-fold
probability of Parkinson's disease, by study (PD above each line, controls below), with the AUC within each study. (d) Trained
on three studies and tested on the fourth; pooled over the four held-out studies, AUC {H.lodo_auc:.2f}. (e) AUC of the whole
procedure on {len(NULL)} within-study shuffles of the diagnoses, against the real labels (P = {CONF['permutation']['p_auc']:.3f}).""")''')

write_nb(HERE / "PD_LCM_rf_figure2.ipynb", CELLS, "f2")
