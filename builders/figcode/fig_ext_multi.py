import os, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

FIG_IN = Path(os.environ.get("FIG_IN", "/kaggle/working")); FIG_OUT = Path(os.environ.get("FIG_OUT", "/kaggle/working"))
R = pd.read_csv(FIG_IN / "external_multi_results.csv").set_index(["cohort", "model"])
OVR = pd.read_csv(FIG_IN / "donor_overlap.csv").set_index("cohort")
SUMM = json.load(open(FIG_IN / "external_multi_summary.json"))
GMF = pd.read_csv(FIG_IN / "external_multi_gene_meta.csv")

INK, MUTED, RULE = "#1B1D20", "#5E656D", "#C9CED4"
CORE_C, PANEL_C, NEURO_C = "#23324A", "#8E1B2E", "#9AA1A9"
SETC = {True: "#8E1B2E", False: "#2E5A87"}
FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "Nimbus Sans", "FreeSans", "DejaVu Sans"]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT_STACK, "font.size": 6.3,
                     "axes.linewidth": 0.5, "axes.edgecolor": "#30343A", "axes.labelcolor": INK, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "xtick.labelsize": 5.8, "ytick.labelsize": 5.8,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
                     "xtick.major.pad": 1.6, "ytick.major.pad": 1.6, "xtick.color": "#30343A", "ytick.color": "#30343A",
                     "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150, "savefig.facecolor": "white", "figure.facecolor": "white",
                     "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic"})

FW, FH = 183.0, 150.0
fig = plt.figure(figsize=(FW / 25.4, FH / 25.4))
M = fig.add_axes([0, 0, 1, 1]); M.set_xlim(0, FW); M.set_ylim(0, FH); M.axis("off")
def fax(x, y_, w, h): return fig.add_axes([x / FW, y_ / FH, w / FW, h / FH])
def letter(x, y_, L, title):
    M.text(x, y_, L, ha="left", va="baseline", fontsize=9, fontweight="bold", color=INK)
    M.text(x + 4.2, y_, title, ha="left", va="baseline", fontsize=7.2, color=INK)
def rule(x0, x1, y_, lw=0.4, color=RULE): M.plot([x0, x1], [y_, y_], color=color, lw=lw, solid_capstyle="butt")
def mark(x, y_, color, filled=True, size=11, marker="o"):
    M.scatter([x], [y_], s=size, marker=marker, facecolor=color if filled else "white", edgecolor=color, linewidth=0.8, zorder=5, clip_on=False)
f2 = lambda v: f"{v:.2f}"

# ======================= a: forest plot =======================
POOL_U, POOL_A = SUMM["pooled"]["unseen"], SUMM["pooled"]["all"]
ORDER = ["GSE20292", "GSE20163", "GSE20164", "GSE8397", "GSE49036", "GSE114517", "GSE168496", "GSE7621"]
PLAT = {"GSE20292": "array (U133A)", "GSE20163": "array (U133A)", "GSE20164": "array (U133A)", "GSE8397": "array (U133A)",
        "GSE49036": "array (U133 Plus 2)", "GSE114517": "RNA-seq", "GSE168496": "RNA-seq", "GSE7621": "array (U133 Plus 2)"}
letter(2.0, 145.0, "a", "The frozen models in eight independent bulk substantia nigra cohorts")
Y0, STEP = 131.0, 4.9
ROWY = {c: Y0 - i * STEP for i, c in enumerate(ORDER)}
ROWY["GSE7621"] -= 1.2
PY = {"unseen": ROWY["GSE7621"] - 7.0, "all": ROWY["GSE7621"] - 7.0 - STEP}
YB, YT = PY["all"] - 3.2, Y0 + 3.2
AX = {"core": (76.0, 34.0), "panel": (128.0, 34.0)}
NUMX = {"core": (114.0, 120.0), "panel": (166.0, 172.0)}
XNEU = 177.5

# column headers
HY = YT + 2.4
for x, t, ha in ((4.0, "cohort", "left"), (21.0, "platform", "left"), (52.5, "control / PD", "center"), (65.5, "shared donors\nremoved", "center")):
    M.text(x, HY, t, ha=ha, va="bottom", fontsize=5.9, color=MUTED, linespacing=1.05)
for key, (x0, w) in AX.items():
    M.text(x0 + w / 2, HY + 3.4, "core classifier" if key == "core" else "Boruta-panel forest", ha="center", va="bottom",
           fontsize=6.6, color=CORE_C if key == "core" else PANEL_C, fontweight="bold")
    M.text(x0 + w / 2, HY, "AUC", ha="center", va="bottom", fontsize=5.9, color=MUTED)
    c = CORE_C if key == "core" else PANEL_C
    mark(NUMX[key][0], HY + 1.0, c, True, 9); mark(NUMX[key][1], HY + 1.0, c, False, 9)
M.scatter([XNEU], [HY + 1.0], s=16, marker="|", color=NEURO_C, linewidth=1.1, clip_on=False)
M.text(XNEU, HY + 3.0, "neuron\nmarkers", ha="center", va="bottom", fontsize=5.6, color=MUTED, linespacing=1.0)
rule(2.0, 181.0, YT + 0.9, lw=0.6, color="#30343A")

# forest axes
AXES = {}
for key, (x0, w) in AX.items():
    ax = fax(x0, YB, w, YT - YB); ax.set_xlim(0.12, 1.02); ax.set_ylim(YB, YT)
    ax.spines["left"].set_visible(False); ax.set_yticks([])
    ax.set_xticks([0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0.25", "0.5", "0.75", "1"])
    ax.axvline(0.5, color="#B7BDC4", lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax.patch.set_alpha(0)
    AXES[key] = ax
for i, c in enumerate(ORDER[:-1]):                                   # light zebra rows across the whole table
    if i % 2 == 0:
        M.add_patch(Rectangle((2.0, ROWY[c] - STEP / 2), 179.0, STEP, color="#F5F4F1", lw=0, zorder=0))

def cohort_row(c, yy):
    r = R.loc[(c, "frozen")]
    name = c + ("$^{\\dagger}$" if c == "GSE7621" else "")
    M.text(4.0, yy, name, ha="left", va="center", fontsize=6.2)
    M.text(21.0, yy, PLAT[c], ha="left", va="center", fontsize=5.9, color=MUTED)
    M.text(52.5, yy, f"{int(r.control)} / {int(r.PD)}", ha="center", va="center", fontsize=6.0)
    k = int(OVR.loc[c, "shared_with_discovery"])
    M.text(65.5, yy, str(k) if k else "–", ha="center", va="center", fontsize=6.0, color=INK if k else MUTED,
           fontweight="bold" if k else "normal")
    for key, (m_rep, m_str) in (("core", ("frozen", "strict")), ("panel", ("panel", "panel_strict"))):
        ax = AXES[key]; col = CORE_C if key == "core" else PANEL_C
        a, s_ = R.loc[(c, m_rep)], R.loc[(c, m_str)]
        ax.plot([a.ci_lo, a.ci_hi], [yy, yy], color=col, lw=0.8, solid_capstyle="butt", zorder=2)
        ax.scatter([a.auc_neuron_markers], [yy], s=22, marker="|", color=NEURO_C, linewidth=1.1, zorder=3)
        ax.scatter([s_.auc], [yy], s=13, facecolor="white", edgecolor=col, linewidth=0.8, zorder=4)
        ax.scatter([a.auc], [yy], s=13, color=col, zorder=5, linewidth=0)
        M.text(NUMX[key][0], yy, f2(a.auc), ha="center", va="center", fontsize=6.0, color=INK)
        M.text(NUMX[key][1], yy, f2(s_.auc), ha="center", va="center", fontsize=6.0, color=MUTED)
    M.text(XNEU, yy, f2(r.auc_neuron_markers), ha="center", va="center", fontsize=6.0, color=MUTED)
for c in ORDER:
    cohort_row(c, ROWY[c])
rule(2.0, 181.0, ROWY["GSE7621"] + STEP / 2 + 0.25, lw=0.3)

# pooled rows
rule(2.0, 181.0, PY["unseen"] + STEP / 2 + 0.6, lw=0.5, color="#30343A")
for tag, P in (("unseen", POOL_U), ("all", POOL_A)):
    yy = PY[tag]
    lab = f"pooled, {len(P['cohorts'])} unseen cohorts" if tag == "unseen" else f"pooled, all {len(P['cohorts'])} cohorts"
    M.text(4.0, yy, lab, ha="left", va="center", fontsize=6.2, fontweight="bold" if tag == "unseen" else "normal")
    M.text(52.5, yy, f"{P['control']} / {P['PD']}", ha="center", va="center", fontsize=6.0,
           fontweight="bold" if tag == "unseen" else "normal")
    for key, (m_rep, m_str) in (("core", ("frozen", "strict")), ("panel", ("panel", "panel_strict"))):
        ax = AXES[key]; col = CORE_C if key == "core" else PANEL_C
        a, s_ = P["auc"][m_rep], P["auc"][m_str]
        h = 1.45
        ax.add_patch(Polygon([[a["ci_lo"], yy], [a["auc"], yy + h], [a["ci_hi"], yy], [a["auc"], yy - h]], closed=True,
                             facecolor=col, alpha=1.0 if tag == "unseen" else 0.45, edgecolor="none", zorder=4))
        ax.scatter([s_["auc"]], [yy], s=13, facecolor="white", edgecolor=col, linewidth=0.8, zorder=5)
        ax.scatter([P["auc"]["neuron"]["auc"]], [yy], s=22, marker="|", color=NEURO_C, linewidth=1.1, zorder=3)
        M.text(NUMX[key][0], yy, f2(a["auc"]), ha="center", va="center", fontsize=6.0, fontweight="bold" if tag == "unseen" else "normal")
        M.text(NUMX[key][1], yy, f2(s_["auc"]), ha="center", va="center", fontsize=6.0, color=MUTED)
    M.text(XNEU, yy, f2(P["auc"]["neuron"]["auc"]), ha="center", va="center", fontsize=6.0, color=MUTED)
for key, (x0, w) in AX.items():
    M.text(x0 + w / 2, YB - 5.4, "AUC in the external cohort", ha="center", va="top", fontsize=6.0, color=INK)

# key and footnote
KY = YB - 11.3
mark(4.8, KY, "#4A4F56", True, 11); M.text(6.8, KY, "model as reported (trained on all 63 discovery people), with 95% CI", va="center", fontsize=5.9)
mark(84.8, KY, "#4A4F56", False, 11)
M.text(86.8, KY, "strictly independent: same recipe, retrained without any discovery study from the same brain bank",
       va="center", fontsize=5.9)
M.scatter([4.8], [KY - 3.6], s=22, marker="|", color=NEURO_C, linewidth=1.1)
M.text(6.8, KY - 3.6, "8 dopamine-neuron marker genes alone (fewer neurons = more PD-like)", va="center", fontsize=5.9)
M.text(84.0, KY - 3.6, "$^{\\dagger}$examined in earlier versions of this project; the other seven were first opened after the model was locked",
       va="center", fontsize=5.6, color=MUTED)

# ======================= b: pooled table =======================
TOPB = KY - 8.8
rule(2.0, 181.0, TOPB + 2.2, lw=0.4)
U = POOL_U
letter(2.0, TOPB - 3.2, "b", f"Pooled over the {len(U['cohorts'])} unseen cohorts ({U['people']} people)")
T0 = TOPB - 7.6
COLS = [(62.0, "AUC (95% CI)"), (82.0, "neuron content\nremoved"), (95.5, "accuracy"), (107.5, "sensitivity"), (119.5, "specificity")]
rule(4.0, 125.0, T0, lw=0.6, color="#30343A")
for x, t in COLS:
    M.text(x, T0 - 1.2, t, ha="center", va="top", fontsize=5.9, color=MUTED, linespacing=1.0)
rule(4.0, 125.0, T0 - 6.0, lw=0.4)
TROWS = [("neuron", "dopamine-neuron markers alone", NEURO_C, None),
         ("frozen", "core classifier", CORE_C, True), ("strict", "   strictly independent", CORE_C, False),
         ("panel", "Boruta-panel forest", PANEL_C, True), ("panel_strict", "   strictly independent", PANEL_C, False)]
CALLS = U["calls@oob"]
for i, (key, lab, col, filled) in enumerate(TROWS):
    yy = T0 - 9.4 - i * 4.3
    if filled is None:
        M.scatter([6.0], [yy], s=20, marker="|", color=col, linewidth=1.1)
    else:
        mark(6.0, yy, col, filled, 10)
    bold = key in ("frozen", "panel")
    M.text(8.5, yy, lab.strip() if not lab.startswith(" ") else "strictly independent", ha="left", va="center", fontsize=6.1,
           color=INK if bold or key == "neuron" else MUTED, fontweight="bold" if key == "panel" else "normal")
    a = U["auc"][key]
    M.text(COLS[0][0], yy, f"{a['auc']:.2f} ({a['ci_lo']:.2f}–{a['ci_hi']:.2f})", ha="center", va="center", fontsize=6.1,
           fontweight="bold" if key == "panel" else "normal")
    if key == "neuron":
        for x, _ in COLS[1:]:
            M.text(x, yy, "–", ha="center", va="center", fontsize=6.0, color=MUTED)
        continue
    nr = U["auc"][key + "|neuron_removed"]
    M.text(COLS[1][0], yy, f2(nr["auc"]), ha="center", va="center", fontsize=6.1, fontweight="bold" if key == "panel" else "normal")
    for (x, _), k in zip(COLS[2:], ("accuracy", "sensitivity", "specificity")):
        M.text(x, yy, f2(CALLS[key][k]), ha="center", va="center", fontsize=6.1)
ybot = T0 - 9.4 - 4 * 4.3 - 2.6
rule(4.0, 125.0, ybot, lw=0.6, color="#30343A")
TH = SUMM["thresholds"]
M.text(4.0, ybot - 2.6, "Accuracy, sensitivity and specificity at each forest's out-of-bag cut-off, fixed on the discovery people before any "
       "external data were seen", ha="left", va="top", fontsize=5.5, color=MUTED)
M.text(4.0, ybot - 5.4, f"(core {TH['core']:.2f}, panel {TH['panel']:.2f}). Cohort AUCs are weighted by PD-control pairs; "
       "95% CI by resampling people within cohorts.", ha="left", va="top", fontsize=5.5, color=MUTED)
M.text(4.0, ybot - 8.2, "Neuron content removed: each score regressed on the 8-gene neuron score within its cohort.",
       ha="left", va="top", fontsize=5.5, color=MUTED)

# ======================= c: gene-level agreement =======================
letter(131.0, TOPB - 3.2, "c", "The panel genes, gene by gene")
G = GMF.copy()
axC = fax(140.0, 8.5, 39.0, TOPB - 17.5)
lim = 1.45
axC.set_xlim(-lim, lim); axC.set_ylim(-lim * 0.9, lim * 0.9)
axC.add_patch(Rectangle((0, 0), lim, lim, color="#F2EFEA", lw=0, zorder=0))
axC.add_patch(Rectangle((-lim, -lim), lim, lim, color="#F2EFEA", lw=0, zorder=0))
axC.axhline(0, color="#9AA1A9", lw=0.4, zorder=1); axC.axvline(0, color="#9AA1A9", lw=0.4, zorder=1)
for r in G.itertuples():
    axC.plot([r.g_lcm, r.g_lcm], [r.g_external, r.g_external_neuron_adj], color="#B9BEC4", lw=0.45, zorder=2)
axC.scatter(G.g_lcm, G.g_external, s=7, facecolor="white", edgecolor="#9AA1A9", linewidth=0.5, zorder=3)
axC.scatter(G.g_lcm, G.g_external_neuron_adj, s=12, color=[SETC[bool(d)] for d in G.deg], linewidth=0, zorder=4)
axC.set_xticks([-1, 0, 1]); axC.set_yticks([-1, 0, 1])
axC.set_xlabel("Hedges' g, laser-capture neurons", fontsize=6.0, labelpad=1.5)
axC.set_ylabel("Hedges' g, bulk nigra (pooled)", fontsize=6.0, labelpad=1.0)
AG = SUMM["gene_agreement"]; ga, gr = AG["g_external_neuron_adj"], AG["g_external"]
pfmt = lambda p: f"{p:.3f}" if p >= 0.001 else f"{p:.0e}"
axC.text(-lim + 0.07, lim * 0.9 - 0.07, f"{ga['same_sign']}/{ga['n']} same sign\nP = {pfmt(ga['binom_p'])}, ρ = {ga['spearman']:.2f}",
         ha="left", va="top", fontsize=5.6, color=INK, linespacing=1.15)
axC.text(lim - 0.07, -lim * 0.9 + 0.07, f"unadjusted\n{gr['same_sign']}/{gr['n']}, ρ = {gr['spearman']:.2f}",
         ha="right", va="bottom", fontsize=5.4, color=MUTED, linespacing=1.15)
# key for c, under the title
KX, KC = 131.5, TOPB - 8.0
axK = [(SETC[True], True, "Boruta & DEG"), (SETC[False], True, "Boruta only"), ("#9AA1A9", False, "unadjusted")]
xx = KX
for col, filled, t in axK:
    M.scatter([xx + 0.8], [KC], s=9 if filled else 7, facecolor=col if filled else "white", edgecolor=col, linewidth=0.6)
    M.text(xx + 2.3, KC, t, va="center", fontsize=5.6, color=MUTED)
    xx += 2.3 + len(t) * 1.02 + 2.6

for ext in ("pdf", "png"):
    fig.savefig(FIG_OUT / f"Figure07_external_multicohort.{ext}", dpi=600 if ext == "png" else None)
print("saved Figure07_external_multicohort")
