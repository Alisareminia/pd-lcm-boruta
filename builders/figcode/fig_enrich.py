import os, re, json
from pathlib import Path
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
FIG_IN = Path(os.environ.get("FIG_IN", str(OUT)))
PLN = pd.read_csv(FIG_IN / "enrich_panel_lineage.csv")
BGL = pd.read_csv(FIG_IN / "enrich_background_lineage.csv")
SBT = pd.read_csv(FIG_IN / "enrich_subtypes.csv")
ORT = pd.read_csv(FIG_IN / "enrich_ora.csv")
GTB = pd.read_csv(FIG_IN / "enrich_gene_table.csv").set_index("gene")
SMY = json.load(open(os.environ.get("FIG_SUMMARY", str(FIG_IN / "enrich_summary.json"))))
SU = SMY["subtypes"]
NICE = GTB.symbol.to_dict()
UPPER2NICE = {v.upper(): v for v in GTB.symbol}
UPC, DNC, ALLC = PANEL_C, CORE_C, "#4A4F56"

C = Canvas(183.0, 128.0); M = C.M

# ======================= a: the lineage axis =======================
C.letter(2.0, 124.0, "a", "Panel genes on the dopamine-neuron lineage axis")
X0, W = 24.0, 150.0
LT = 2.0                                                   # linear part of the symmetric-log scale
tr = lambda v: np.sign(v) * np.log10(1 + np.abs(v) / LT)  # symmetric log, used for placement and bins
lo, hi = tr(-22.0), tr(19.0)
axT = C.ax(X0, 88.0, W, 25.0); axT.set_xlim(lo, hi); axT.set_ylim(0, 2.3); axT.axis("off")
axH = C.ax(X0, 79.0, W, 8.0); axH.set_xlim(lo, hi)
for ax in (axT, axH):
    ax.axvspan(lo, tr(SU["background_p05"]), color="#EEF1F4", lw=0, zorder=0)
    ax.axvspan(tr(SU["background_p95"]), hi, color="#EEF1F4", lw=0, zorder=0)
    ax.axvline(0, color="#C9CED4", lw=0.5, zorder=0)
bins = np.linspace(lo, hi, 90)
hh, _ = np.histogram(tr(BGL.lineage.to_numpy()), bins=bins)
axH.fill_between(np.repeat(bins, 2)[1:-1], 0, np.repeat(np.sqrt(hh), 2), color="#B9BEC4", lw=0, zorder=1)
axH.set_ylim(0, np.sqrt(hh).max() * 1.05); axH.set_yticks([]); axH.spines["left"].set_visible(False)
TK = [-16, -8, -4, -2, 0, 2, 4, 8, 16]
axH.set_xticks([tr(v) for v in TK]); axH.set_xticklabels([f"{v:+d}" if v else "0" for v in TK]); axH.tick_params(axis="x", labelsize=5.8)
axH.patch.set_alpha(0)
C.text(X0 - 1.5, 83.0, f"All {len(BGL):,} genes", ha="right", fontsize=5.7, color=MUTED)
C.text(X0 + W / 2, 71.0, "Lineage score: mean marker z in CALB1 subtypes minus SOX6 subtypes (Kamath et al. 2022)", ha="center", fontsize=6.0)
C.text(X0, 71.0, "$\\leftarrow$ SOX6 lineage (vulnerable)", ha="left", fontsize=5.8, color=DNC)
C.text(X0 + W, 71.0, "CALB1 lineage (resilient) $\\rightarrow$", ha="right", fontsize=5.8, color=UPC)
TRACK = {"down": 1.55, "up": 0.55}
for d, col, lab in (("down", DNC, "Down in PD"), ("up", UPC, "Up in PD")):
    sub = PLN[PLN.direction == d]
    yv = TRACK[d]
    axT.plot([lo, hi], [yv, yv], color="#DADDE1", lw=0.5, zorder=1)
    axT.scatter(tr(sub.lineage.to_numpy()), np.full(len(sub), yv), s=16, color=col, edgecolor="white", linewidth=0.4, zorder=4)
    C.text(X0 - 1.5, 88.0 + 25.0 * yv / 2.3, f"{lab} ({len(sub)})", ha="right", fontsize=6.0, color=col, fontweight="bold")
    # labels for the genes beyond the central band, staggered so they never touch
    lab_df = sub[sub.lineage.abs() > 2.5].assign(x=lambda d_: tr(d_.lineage)).sort_values("x")
    mm_per = W / (hi - lo); last = {}
    for r in lab_df.itertuples():
        level = 0
        while level in last and (r.x - last[level]) * mm_per < 9.0:
            level += 1
        last[level] = r.x
        ty = yv + (0.22 + 0.2 * level) * (1 if d == "down" else -1)
        axT.plot([r.x, r.x], [yv + 0.06 * np.sign(ty - yv), ty - 0.04 * np.sign(ty - yv)], color="#9AA1A9", lw=0.4, zorder=2)
        axT.text(r.x, ty, NICE.get(r.gene, r.symbol), ha="center", va="bottom" if d == "down" else "top", fontsize=5.8,
                 fontstyle="italic", color=col)
pf = lambda p: f"P = {p:.4f}" if p >= 0.0001 else "P < 0.0001"
C.text(181.0, 124.0, f"Down vs up genes: {pf(SU['up_vs_down'])} (Mann-Whitney)", ha="right", fontsize=5.8)
C.text(181.0, 120.8, f"vs genes equally changed in PD: {pf(SU['p_vs_DE_matched'])}", ha="right", fontsize=5.8)
C.text(181.0, 117.6, "shaded: outer 5% of all genes", ha="right", fontsize=5.5, color=MUTED)
C.rule(2.0, 181.0, 66.5, lw=0.35, color="#8C9299")

# ======================= b: subtype markers =======================
C.letter(2.0, 61.5, "b", "Overlap with subtype markers")
ORDER = ["SOX6_AGTR1", "SOX6_PART1", "SOX6_DDT", "SOX6_GFRA2", "CALB1_CALCR", "CALB1_CRYM_CCDC68", "CALB1_GEM", "CALB1_PPP1R17",
         "CALB1_RBP4", "CALB1_TRHR"]
CX = {"down": 44.0, "up": 56.0}
RY0, RS = 49.0, 4.2
C.text(CX["down"], 55.0, "Down\n(12)", ha="center", fontsize=5.8, color=DNC, fontweight="bold", linespacing=1.0)
C.text(CX["up"], 55.0, "Up\n(18)", ha="center", fontsize=5.8, color=UPC, fontweight="bold", linespacing=1.0)
C.rule(4.0, 70.0, 52.4, lw=0.5, color=INK)
sbt = SBT.set_index(["subtype", "list"])
for i, st in enumerate(ORDER):
    yy = RY0 - i * RS - (1.6 if i >= 4 else 0)
    C.text(28.5, yy, st.replace("_", " "), ha="right", fontsize=5.7)
    for d in ("down", "up"):
        r = sbt.loc[(st, d)]; col = DNC if d == "down" else UPC
        if r.hits:
            sig = r.q_BH < 0.05
            M.scatter([CX[d]], [yy], s=6 + 9 * r.hits, color=col if sig else "white", edgecolor=col, linewidth=0.8, zorder=3)
            C.text(CX[d] + 3.2, yy, str(int(r.hits)), ha="left", fontsize=5.3, color=col if sig else MUTED)
        else:
            M.plot([CX[d] - 0.6, CX[d] + 0.6], [yy, yy], color="#C9CED4", lw=0.6)
    if st == "SOX6_AGTR1":
        C.text(63.0, yy, "lost in PD", ha="left", fontsize=5.3, color=MUTED, fontstyle="italic")
for grp, i0, i1 in (("SOX6", 0, 3), ("CALB1", 4, 9)):
    y0 = RY0 - i0 * RS - (1.6 if i0 >= 4 else 0) + 1.6; y1 = RY0 - i1 * RS - (1.6 if i1 >= 4 else 0) - 1.6
    M.plot([4.8, 4.8], [y1, y0], color=DNC if grp == "SOX6" else UPC, lw=1.4, solid_capstyle="butt")
    C.text(3.2, (y0 + y1) / 2, grp, ha="center", rotation=90, fontsize=5.6, color=DNC if grp == "SOX6" else UPC, fontweight="bold")
LY = RY0 - 9 * RS - 1.6 - 6.0
for k, (s_, lab) in enumerate(((1, "1"), (2, "2"), (4, "4 genes"))):
    M.scatter([8.0 + k * 7.5], [LY], s=6 + 9 * s_, color="white", edgecolor="#6B7178", linewidth=0.7)
    C.text(9.8 + k * 7.5, LY, lab, ha="left", fontsize=5.3, color=MUTED)
M.scatter([38.0], [LY], s=24, color="#6B7178", linewidth=0); C.text(40.0, LY, "filled: FDR < 0.05 (200 top markers)", ha="left", fontsize=5.3, color=MUTED)

# ======================= c: over-representation, against a calibrated threshold =======================
C.letter(78.0, 61.5, "c", "Pathway over-representation")
def nice_term(t):
    t = re.sub(r"\s*\((GO:\d+)\)$", "", t); t = re.sub(r"\s+R-HSA-\d+$", "", t); t = re.sub(r"\s+WP\d+$", "", t)
    t = t[0] + t[1:].lower()
    for a in ("dna", "rna", "gpcr", "hsf1", "cns"):
        t = re.sub(rf"\b{a}\b", a.upper(), t)
    return t if len(t) <= 50 else t[:48].rstrip() + "..."
AXX0, AXW = 126.0, 30.0
GX = AXX0 + AXW + 2.0
axC = C.ax(AXX0, 7.5, AXW, 44.5)
ROWS, yy, GROUPS = [], 0.0, []
for lst, col, lab in (("up", UPC, "Up in PD (18 genes)"), ("down", DNC, "Down in PD (12 genes)"), ("all", ALLC, "All 30 genes")):
    top = ORT[ORT.list == lst].nsmallest(4, "p")
    y_start = yy
    for r in top.itertuples():
        ROWS.append((yy, r, col)); yy -= 1.0
    GROUPS.append((lst, col, lab, y_start, yy + 1.0))
    yy -= 0.9
axC.set_ylim(yy + 0.5, 1.4); axC.set_xlim(0, 4.6)
axC.spines["left"].set_visible(False); axC.set_yticks([]); axC.patch.set_alpha(0)
axC.set_xticks([0, 1, 2, 3, 4]); axC.set_xlabel("$-\\log_{10}$ P", fontsize=6.0, labelpad=1.5)
thr = SMY["ora_calibration"]
for lst, col, lab, y0, y1 in GROUPS:
    t = -np.log10(thr[lst]["p_fwer05_random"])
    axC.plot([t, t], [y1 - 0.45, y0 + 0.45], color="#8C9299", lw=0.7, ls=(0, (2.2, 1.6)), zorder=1)
    axC.text(0.02, y0 + 0.62, lab, ha="left", va="bottom", fontsize=5.8, color=col, fontweight="bold",
             transform=axC.get_yaxis_transform())
for y_, r, col in ROWS:
    x = -np.log10(r.p)
    axC.plot([0, x], [y_, y_], color="#D5D9DE", lw=0.6, zorder=1)
    axC.scatter([x], [y_], s=6 + 7 * r.hits, color=col, zorder=3, linewidth=0)
    gl = [UPPER2NICE.get(g, g) for g in r.genes.split(", ")]
    gtxt = ", ".join(gl) if len(gl) <= 3 else ", ".join(gl[:3]) + f" +{len(gl) - 3}"
    ytxt = 7.5 + 44.5 * (y_ - (yy + 0.5)) / (1.4 - (yy + 0.5))
    C.text(AXX0 - 1.2, ytxt, nice_term(r.term), ha="right", fontsize=5.4)
    C.text(GX, ytxt, gtxt, ha="left", fontsize=5.1, fontstyle="italic", color=col)
C.text(82.2, 57.4, "Top four terms per list; dashed line: family-wise 5% threshold from random gene sets of the same size",
       ha="left", fontsize=5.2, color=MUTED)
C.text(GX, 54.0, "Genes", ha="left", fontsize=5.5, color=INK)
C.save("Figure08_panel_biology")
