from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Circle
import matplotlib.patheffects as pe
UPC, DNC, ALLC = PANEL_C, CORE_C, "#3E454D"
COLS_L = {"up": UPC, "down": DNC, "all": ALLC}
SRC_C = {"GO:BP": "#6B5B7A", "GO:MF": "#4C6583", "GO:CC": "#5F6B3A", "REAC": "#23324A", "KEGG": "#7A2533", "WP": "#8C6B3A"}
RENAME = {"cerebellar climbing fiber to purkinje cell synapse": "Climbing-fibre synapse (GO label)*"}
GPR["disp"] = [RENAME.get(n.lower(), cap(n)) for n in GPR.name]       # the GO label is renamed, not hidden
GPR["fold"] = (GPR.hits / GPR.query_size) / (GPR.term_size / GPR.domain)          # times more panel genes than expected
fmax = max(GPR[GPR.significant].fold.max() if GPR.significant.any() else 1, 1)
GEF = pd.read_csv(FIG_IN / "composition_gene_effects.csv").set_index("symbol")
UPG = set(", ".join(GPR[GPR.list == "up"].genes.dropna()).split(", "))
SIG = GPR[GPR.significant].drop_duplicates("name").sort_values("p_adj")
C = Canvas(183.0, 215.0); M = C.M
N_TOP = {"up": 3, "down": 3, "all": 4}
def chip(x, y_, src):
    M.add_patch(Rectangle((x, y_ - 1.35), 10.4, 2.7, facecolor=SRC_C.get(src, "#E9ECEF"), edgecolor="none", zorder=2))
    C.text(x + 5.2, y_, SRCLAB.get(src, src), ha="center", fontsize=4.6, color="white", zorder=3)

# ======================= a: the results table =======================
C.letter(2.0, 211.0, "a", "g:Profiler with a 5,622-gene background and Benjamini-Hochberg FDR")
SHOW = {k: GPR[GPR.list == k].nsmallest(N_TOP[k], "p_adj") for k in ("up", "down", "all")}
GENES_S = []
for k in ("up", "down", "all"):
    for r in SHOW[k].itertuples():
        for g in str(r.genes).split(", "):
            if g and g not in GENES_S:
                GENES_S.append(g)
GENES_S = sorted(GENES_S, key=lambda g: (g not in UPG, g))
CHX, NX, SZX, FEX, FEW, AX0, AXW, PX, KX, GX0, GX1 = 3.0, 15.0, 74.0, 82.0, 14.0, 100.0, 22.0, 125.0, 134.0, 142.0, 181.0
gcw = (GX1 - GX0) / len(GENES_S)
GXP = {g: GX0 + (i + 0.5) * gcw for i, g in enumerate(GENES_S)}
HY = 198.0
for x, t, ha in ((CHX, "Source", "left"), (NX, "Pathway", "left"), (SZX, "Genes in\nterm / panel", "center"),
                 (FEX + FEW / 2, "Fold\nenrichment", "center"), (AX0 + AXW / 2, "$-\\log_{10}$ FDR", "center"),
                 (PX, "FDR", "center"), (KX, "P vs random\nlists", "center")):
    C.text(x, HY, t, ha=ha, fontsize=5.7, linespacing=1.0)
for g in GENES_S:
    C.text(GXP[g], HY - 1.6, g, ha="center", va="bottom", rotation=90, fontsize=5.1, fontstyle="italic",
           color=UPC if g in UPG else DNC)
C.text((GX0 + GX1) / 2, 209.6, "Panel genes in the pathway", ha="center", fontsize=5.7)
C.rule(2.0, 181.0, HY - 2.6, lw=0.5, color=INK)
ROWS, y_ = [], HY - 6.0
for k in ("up", "down", "all"):
    C.text(NX, y_, LISTLAB[k], ha="left", fontsize=5.8, color=COLS_L[k], fontweight="bold")
    ns = int(GPR[GPR.list == k].significant.sum())
    C.text(CHX, y_, f"{ns} significant", ha="left", fontsize=5.2, color=COLS_L[k] if ns else MUTED)
    y_ -= 4.6
    for r in SHOW[k].itertuples():
        ROWS.append((y_, k, r)); y_ -= 6.2
    y_ -= 1.2
axA = C.ax(AX0, y_ + 3.0, AXW, HY - 3.4 - (y_ + 3.0)); axA.set_ylim(y_ + 3.0, HY - 3.4); axA.set_xlim(0, 3.2)
axA.spines["left"].set_visible(False); axA.set_yticks([]); axA.patch.set_alpha(0)
axA.set_xticks([0, 1, 2, 3]); axA.tick_params(axis="x", labelsize=5.6)
axA.axvline(-np.log10(0.05), color="#8C9299", lw=0.7, ls=(0, (2.2, 1.6)), zorder=1)
for i, (yy, k, r) in enumerate(ROWS):
    col = COLS_L[k]
    if r.significant:
        M.add_patch(Rectangle((2.0, yy - 3.1), 179.0, 6.2, facecolor="#F1F3F6", lw=0, zorder=0))
        M.add_patch(Rectangle((2.0, yy - 3.1), 0.9, 6.2, facecolor=col, lw=0, zorder=1))
    chip(CHX, yy, r.source)
    C.text(NX, yy, "\n".join(wrap(r.disp, 52)), ha="left", fontsize=5.3, linespacing=1.02,
           fontweight="bold" if r.significant else "normal")
    C.text(SZX, yy, f"{int(r.term_size)} / {int(r.hits)}", ha="center", fontsize=5.4)
    fw = FEW * min(r.fold / fmax, 1)
    M.add_patch(Rectangle((FEX, yy - 1.2), max(fw, 0.4), 2.4, facecolor=col if r.significant else "#C9CED4", lw=0))
    C.text(FEX + FEW + 1.0, yy, f"{r.fold:.0f}x", ha="left", fontsize=5.2, color=INK if r.significant else MUTED)
    x = -np.log10(r.p_adj)
    axA.plot([0, x], [yy, yy], color="#C9CED4", lw=0.6, zorder=2)
    axA.scatter([x], [yy], s=8 + 8 * r.hits, color=col if r.significant else "white", edgecolor=col, linewidth=0.7, zorder=3)
    C.text(PX, yy, f"{r.p_adj:.3f}", ha="center", fontsize=5.4, fontweight="bold" if r.significant else "normal")
    C.text(KX, yy, f"{r.calibrated_p:.2f}", ha="center", fontsize=5.4,
           fontweight="bold" if r.calibrated_p < 0.05 else "normal", color=INK if r.calibrated_p < 0.05 else MUTED)
    members = set(str(r.genes).split(", "))
    for g in GENES_S:
        if g in members:
            M.scatter([GXP[g]], [yy], s=13, color=col, lw=0, zorder=3)
        else:
            M.scatter([GXP[g]], [yy], s=1.5, color="#C9CED4", lw=0, zorder=2)
BOT = y_ - 1.0
C.text(3.0, BOT, "Shaded rows pass FDR 0.05. Fold enrichment: panel genes in the term against what chance gives. \"P vs random "
       "lists\": how often 100 random lists of the same size reached a smaller FDR.", ha="left", fontsize=5.3, color=MUTED)
C.text(3.0, BOT - 2.8, "*GO \"cerebellar climbing fibre to Purkinje cell synapse\": a label for a type of excitatory synapse, named "
       "after the tissue where these genes were first described - not a midbrain structure.", ha="left", fontsize=5.3, color=MUTED)
C.rule(2.0, 181.0, BOT - 6.4, lw=0.35, color="#8C9299")

# ======================= b: the graph of significant pathways and their genes =======================
TOPB = BOT - 11.0
C.letter(2.0, TOPB, "b", "Every significant pathway and the panel genes that carry it")
NET = SIG.reset_index(drop=True)
GENES_B = list(dict.fromkeys(sum([str(r.genes).split(", ") for r in NET.itertuples()], [])))
MEMB = {r.disp: str(r.genes).split(", ") for r in NET.itertuples()}
NH = 46.0
axB = C.ax(4.0, TOPB - NH - 2.0, 92.0, NH)
axB.set_xlim(0, 92); axB.set_ylim(0, NH); axB.axis("off"); axB.set_aspect("equal")
cx, cy, R = 44.0, NH / 2, 17.5
ang = {t: np.pi / 2 + 2 * np.pi * i / len(NET) for i, t in enumerate(NET.disp)}
tpos = {t: (cx + R * np.cos(a), cy + R * np.sin(a)) for t, a in ang.items()}
gpos, gdir, used = {}, {}, {}
for g in GENES_B:
    inn = [t for t in NET.disp if g in MEMB[t]]
    if len(inn) > 1:
        bx = np.mean([tpos[t][0] for t in inn]); by = np.mean([tpos[t][1] for t in inn])
        gpos[g] = (cx + (bx - cx) * 0.42, cy + (by - cy) * 0.42)
        v = np.array(gpos[g]) - np.array([cx, cy]); n = np.linalg.norm(v)
        gdir[g] = v / n if n > 1e-6 else np.array([0.0, -1.0])
    else:
        t = inn[0]; k = used.get(t, 0); used[t] = k + 1
        a = ang[t] + (0.32 if k == 0 else -0.32) * (1 if k < 2 else 2)
        gpos[g] = (cx + (R + 11.0) * np.cos(a), cy + (R + 11.0) * np.sin(a))
        gdir[g] = np.array([np.cos(a), np.sin(a)])
SHARED = {}
for g in GENES_B:
    SHARED.setdefault(tuple(sorted(t for t in NET.disp if g in MEMB[t])), []).append(g)
for key, gs in SHARED.items():
    if len(key) > 1 and len(gs) > 1:
        bx, by = gpos[gs[0]]
        for j, g in enumerate(gs):
            a = 2 * np.pi * j / len(gs) + np.pi / 4
            gpos[g] = (bx + 7.5 * np.cos(a), by + 7.5 * np.sin(a))
            gdir[g] = np.array([np.cos(a), np.sin(a)])
STRETCH = 1.9                                        # the panel is twice as wide as it is tall
for d_ in (tpos, gpos):
    for k_ in d_:
        d_[k_] = (cx + (d_[k_][0] - cx) * STRETCH, d_[k_][1])
for g in GENES_B:
    v_ = np.array([gdir[g][0] * STRETCH, gdir[g][1]]); gdir[g] = v_ / np.linalg.norm(v_)
PADX, PADB, PADT = 2.0, 8.0, 1.5                    # room for labels, the legend row, the panel letter
AVW, AVH = 92.0 - 2 * PADX, NH - PADB - PADT
PLBL = 5.9                                           # the "P1" label, in points
RMIN = 0.3528 * PLBL * 0.92                          # a circle this wide holds it with air around it
trad = {r.disp: float(np.sqrt(RMIN ** 2 + 0.085 * min(r.term_size, 60))) for r in NET.itertuples()}
GRAD = 0.95                                          # gene dots
def label_box(g, sc):
    """Where a gene's label lands, in layout units, for the given scale."""
    x, y = gpos[g]; dx, dy = gdir[g]
    ax_, ay_ = x + 3.1 * dx, y + 3.1 * dy
    w, h = 1.18 * len(g) / sc, 2.2 / sc                # the text itself does not shrink with the graph
    lo = ax_ - w / 2 if abs(dx) < 0.45 else (ax_ if dx > 0 else ax_ - w)
    bo = ay_ - h / 2 if abs(dy) < 0.45 else (ay_ if dy > 0 else ay_ - h)
    return lo, lo + w, bo, bo + h
sc = 1.0
for _ in range(8):                                   # label size feeds back into the scale; a few passes settle it
    xs_, ys_ = [], []
    for t_, pnt in tpos.items():
        m_ = (trad[t_] + 0.6) / sc
        xs_ += [pnt[0] - m_, pnt[0] + m_]; ys_ += [pnt[1] - m_, pnt[1] + m_]
    for pnt in gpos.values():
        m_ = (GRAD + 0.4) / sc
        xs_ += [pnt[0] - m_, pnt[0] + m_]; ys_ += [pnt[1] - m_, pnt[1] + m_]
    for g in GENES_B:
        a_, b_, c_, d_ = label_box(g, sc); xs_ += [a_, b_]; ys_ += [c_, d_]
    x_lo, x_hi, y_lo, y_hi = min(xs_), max(xs_), min(ys_), max(ys_)
    sc = min(AVW / (x_hi - x_lo), AVH / (y_hi - y_lo))
ox = PADX + (AVW - sc * (x_hi - x_lo)) / 2 - sc * x_lo
oy = PADB + (AVH - sc * (y_hi - y_lo)) / 2 - sc * y_lo
fit = lambda p_: (ox + sc * p_[0], oy + sc * p_[1])
tpos = {t: fit(p_) for t, p_ in tpos.items()}
gpos = {g: fit(p_) for g, p_ in gpos.items()}
for t in NET.disp:
    for g in MEMB[t]:
        col = UPC if g in UPG else DNC
        (x0, y0), (x1, y1) = tpos[t], gpos[g]
        mx, my = (x0 + x1) / 2 + (y1 - y0) * 0.10, (y0 + y1) / 2 - (x1 - x0) * 0.10
        axB.add_patch(PathPatch(MPath([(x0, y0), (mx, my), (x1, y1)], [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]),
                                facecolor="none", edgecolor=col, lw=1.1, alpha=0.5, zorder=2))
for i, r in enumerate(NET.itertuples()):
    x, y0 = tpos[r.disp]
    axB.add_patch(Circle((x, y0), trad[r.disp], facecolor="white", edgecolor=SRC_C.get(r.source, INK), linewidth=1.3, zorder=4))
    axB.text(x, y0, f"P{i + 1}", ha="center", va="center", fontsize=PLBL, fontweight="bold", color=SRC_C.get(r.source, INK), zorder=6)
HALO = [pe.withStroke(linewidth=1.8, foreground="white")]
for g in GENES_B:
    x, y0 = gpos[g]; col = UPC if g in UPG else DNC
    dx, dy = gdir[g]
    axB.add_patch(Circle((x, y0), GRAD, facecolor=col, edgecolor="white", linewidth=0.8, zorder=5))
    ha = "center" if abs(dx) < 0.45 else ("left" if dx > 0 else "right")
    va = "center" if abs(dy) < 0.45 else ("bottom" if dy > 0 else "top")
    axB.text(x + (GRAD + 1.3) * dx, y0 + (GRAD + 1.3) * dy, g, ha=ha, va=va, fontsize=5.5, fontstyle="italic", color=col, zorder=6,
             path_effects=HALO)
axB.add_patch(Circle((3.0, 3.4), GRAD, facecolor=UPC, edgecolor="white", linewidth=0.8))
axB.text(5.6, 3.4, "higher in PD", ha="left", va="center", fontsize=5.2, color=MUTED)
axB.add_patch(Circle((28.0, 3.4), GRAD, facecolor=DNC, edgecolor="white", linewidth=0.8))
axB.text(30.6, 3.4, "lower in PD", ha="left", va="center", fontsize=5.2, color=MUTED)
axB.add_patch(Circle((52.0, 3.4), max(trad.values()), facecolor="white", edgecolor="#8C9299", linewidth=1.1))
axB.text(55.6, 3.4, "pathway, ringed by source; area = genes in it", ha="left", va="center", fontsize=5.2, color=MUTED)
LX = 100.0
ly = TOPB - 5.0
for i, r in enumerate(NET.itertuples()):
    C.text(LX, ly, f"P{i + 1}", ha="left", fontsize=5.8, fontweight="bold", color=SRC_C.get(r.source, INK))
    chip(LX + 6.0, ly, r.source)
    C.text(LX + 18.0, ly, f"FDR {r.p_adj:.3f}    {r.fold:.0f}x    {int(r.term_size)} genes    P vs random {r.calibrated_p:.2f}",
           ha="left", fontsize=5.0, color=MUTED)
    C.text(LX, ly - 3.4, "\n".join(wrap(r.disp, 52)), ha="left", va="top", fontsize=5.3, linespacing=1.15)
    ly -= 4.0 + 2.9 * len(wrap(r.disp, 52)) + 1.4
BB = TOPB - NH - 6.0
C.text(2.0, BB, "Four pathways rest on five genes, three of them shared. Those genes are largely markers of which dopamine neurons "
       "survive,", ha="left", fontsize=5.3, color=MUTED)
C.text(2.0, BB - 2.8, "so part of this signal is the neuron mix rather than a pathway (Figure 10).", ha="left", fontsize=5.3, color=MUTED)
C.rule(2.0, 181.0, BB - 6.6, lw=0.35, color="#8C9299")

# ======================= c: what the same settings give on random gene lists =======================
DY = BB - 11.4
C.letter(2.0, DY, "c", "The same settings on 100 random gene lists of the same size")
BW, BH0, BY = 50.0, 17.0, 13.0
for i, k in enumerate(("up", "down", "all")):
    x0 = 16.0 + i * (BW + 6.0)
    ax = C.ax(x0, BY, BW, BH0)
    c = CALR[CALR.list == k].n_significant.to_numpy()
    obs = int(GPR[GPR.list == k].significant.sum())
    top = max(c.max(), obs) + 1
    ax.hist(c, bins=np.arange(-0.5, top + 1.5, 1), color="#C9CED4", lw=0)
    ax.axvline(obs, color=COLS_L[k], lw=1.3, zorder=3)
    ax.text(obs + 0.6, ax.get_ylim()[1] * 0.94, f"the panel: {obs}", fontsize=5.3, color=COLS_L[k], ha="left", va="top")
    ax.set_xlabel("\"Significant\" pathways returned", fontsize=5.6, labelpad=1.4)
    if i == 0:
        ax.set_ylabel("Random lists", fontsize=5.6, labelpad=2)
    ax.tick_params(labelsize=5.3)
    C.text(x0 + BW / 2, BY + BH0 + 1.8, LISTLAB[k], ha="center", fontsize=5.8, color=COLS_L[k], fontweight="bold")
    C.text(x0 + BW / 2, BY - 8.2, f"{100 * CAL[k]['random_with_any_significant']:.0f}% of random lists return at least one "
           f"(mean {CAL[k]['random_mean_significant']:.1f});\nstrongest term vs random lists: P = {CAL[k]['calibrated_p_best']:.2f}",
           ha="center", va="top", fontsize=5.3, color=MUTED, linespacing=1.25)
C.save("Figure09_gprofiler_pathways")
