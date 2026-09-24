# ============================== FIGURE 4 ==============================
# The Venn is drawn rather than called: both circle AREAS are the set sizes and
# the lens area is solved numerically, so the overlap is genuinely five genes'
# worth of area. It sits in the volcano's upper-left quadrant - a region the data
# leaves empty, since almost nothing is both strongly down-regulated and highly
# significant - so it costs no space and needs no frame.
#
# The three set counts used to appear three times over: in the strapline, in the
# Venn and again in the key. The Venn keeps them; the other two were cut. The
# gene dossiers are set on a label/value grid rather than as run-on sentences,
# which is what made them read as filler.
from matplotlib.patches import Circle

de   = T("03_de_results_full").copy()
memb = T("06_gene_set_membership")
ovl  = T("06_overlap_analysis").set_index("set")["n"]
ann4 = T("13_gene_annotation_master").set_index("gene")
boruta_set  = set(memb.loc[memb["in_boruta"] == True, "gene"])
overlap_set = set(memb.loc[memb["in_overlap"] == True, "gene"])

# the pipeline records which DE rule fed the sets; the nominal rule is judged on
# the raw p, so that is the scale drawn
PCOL = "padj" if DE_RULE == "strict" else "pvalue"
P_THR = float(DE_SET.get("de_fdr", 0.05)) if DE_RULE == "strict" else float(DE_SET.get("nominal_p", 0.01))
de["neglog10"] = -np.log10(de[PCOL].clip(lower=1e-300))
de["is_deg"]    = de["is_deg"].astype(bool)
de["is_boruta"] = de["gene"].isin(boruta_set)
de["is_ovl"]    = de["gene"].isin(overlap_set)

def pfmt(p):
    if not np.isfinite(p) or p <= 0:
        return "–"
    e = int(np.floor(np.log10(p)))
    return f"${p / 10 ** e:.1f}\\times10^{{{e}}}$"

def venn_geometry(n1, n2, n12):
    """Radii and centre distance giving areas n1, n2 and a lens of area n12."""
    r1, r2 = np.sqrt(n1 / np.pi), np.sqrt(n2 / np.pi)
    def lens(d):
        if d >= r1 + r2:
            return 0.0
        if d <= abs(r1 - r2):
            return np.pi * min(r1, r2) ** 2
        a1 = r1 ** 2 * np.arccos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
        a2 = r2 ** 2 * np.arccos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
        a3 = 0.5 * np.sqrt(max((-d + r1 + r2) * (d + r1 - r2) *
                               (d - r1 + r2) * (d + r1 + r2), 0.0))
        return a1 + a2 - a3
    lo, hi = abs(r1 - r2) + 1e-9, r1 + r2
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if lens(mid) > n12:
            lo = mid
        else:
            hi = mid
    return r1, r2, 0.5 * (lo + hi)

fig = plt.figure(figsize=(13.8, 9.4))
suptitle(fig, "Dual-track discovery: predictive selection versus statistical significance")

L, R = 0.050, 0.986
COL_L, COL_R = 0.742, 0.986
ax = fig.add_axes([L, 0.092, 0.648, 0.792])

cat_bg  = de[~de["is_deg"] & ~de["is_boruta"]]
cat_deg = de[de["is_deg"] & ~de["is_boruta"]]
cat_bor = de[de["is_boruta"] & ~de["is_ovl"]]
cat_ovl = de[de["is_ovl"]]

ax.scatter(cat_bg["log2FC"], cat_bg["neglog10"], s=7, c=VOL_BG, alpha=0.75,
           edgecolors="none", rasterized=True, zorder=1)
ax.scatter(cat_deg["log2FC"], cat_deg["neglog10"], s=40, c=VOL_DEG, alpha=0.92,
           edgecolors="white", linewidths=0.6, zorder=3)
ax.scatter(cat_bor["log2FC"], cat_bor["neglog10"], s=92, c=VOL_BOR, alpha=0.98,
           edgecolors="white", linewidths=1.3, zorder=5)
ax.scatter(cat_ovl["log2FC"], cat_ovl["neglog10"], s=215, c=VOL_BOTH,
           edgecolors="white", linewidths=1.9, zorder=6)

ax.axhline(-np.log10(P_THR), ls=(0, (4, 4)), lw=1.0, color="#A7ADB3", zorder=2)
if DE_RULE == "strict":
    for xv in (-1, 1):
        ax.axvline(xv, ls=(0, (4, 4)), lw=1.0, color="#A7ADB3", zorder=2)

xmax = max(1.0 if DE_RULE != "strict" else 3.0, de["log2FC"].abs().quantile(0.9995),
           de.loc[de["is_deg"] | de["is_boruta"], "log2FC"].abs().max()) * 1.15
ax.set_xlim(-xmax, xmax)
ax.set_ylim(-0.35, de["neglog10"].max() * 1.10)
ax.set_xlabel("log$_2$ fold change   (PD vs Control)")
ax.set_ylabel("$-$log$_{10}$ adjusted $p$" if DE_RULE == "strict" else "$-$log$_{10}$ $p$")
dashgrid(ax)
ax.text(-xmax * 0.985, -np.log10(P_THR) + 0.07,
        "adj. $p$ = 0.05" if DE_RULE == "strict" else f"$p$ = {P_THR:g}  (not FDR-corrected)",
        ha="left", va="bottom", fontsize=9.0, color=MUTED)
if DE_RULE == "strict":
    ax.text(1.06, -0.18, "|log$_2$FC| = 1", ha="left", va="bottom", fontsize=9.0,
            color=MUTED)

ax.figure.canvas.draw()
# The Venn goes wherever the volcano is emptiest, and that space is then marked
# occupied so no gene label can be parked underneath it.
VW, VH = 0.212, 0.252
_inv = ax.transData.inverted()
_pts = np.c_[de["log2FC"], de["neglog10"]]
_wt = np.where(de["is_deg"] | de["is_boruta"], 25.0, 1.0)
def _box_data(x0f, y0f):
    p0 = _inv.transform(fig.transFigure.transform((x0f, y0f)))
    p1 = _inv.transform(fig.transFigure.transform((x0f + VW, y0f + VH)))
    return p0, p1
def _crowd(xy):
    p0, p1 = _box_data(*xy)
    inside = ((_pts[:, 0] >= p0[0]) & (_pts[:, 0] <= p1[0]) &
              (_pts[:, 1] >= p0[1]) & (_pts[:, 1] <= p1[1]))
    return float((inside * _wt).sum())
_cands = [(0.068, 0.596), (0.300, 0.610), (0.470, 0.596), (0.068, 0.330), (0.470, 0.330)]
VX, VY = min(_cands, key=_crowd)
_p0, _p1 = _box_data(VX, VY)
venn_block = [(x_, y_, 24.0) for x_ in np.linspace(_p0[0], _p1[0], 6)
              for y_ in np.linspace(_p0[1], _p1[1], 6)]
lab_items = [(r["log2FC"], r["neglog10"], sym(r["gene"]), VOL_BOR_INK, True)
             for _, r in cat_bor.iterrows()]
lab_items += [(r["log2FC"], r["neglog10"], sym(r["gene"]), VOL_DEG_INK, False)
              for _, r in cat_deg[cat_deg["gene"].map(named)]
              .nlargest(6, "neglog10").iterrows()]
obstacles = [(r["log2FC"], r["neglog10"], 13.5) for _, r in cat_ovl.iterrows()]
obstacles += [(r["log2FC"], r["neglog10"], 7.5) for _, r in cat_bor.iterrows()]
obstacles += venn_block
smart_labels(ax, lab_items, fontsize=9.2, avoid=obstacles,
             gap_pt=6.5, step_pt=2.4, max_steps=190, lw=0.6)

# ---- the Venn, area-true, in the quadrant the data leaves empty ----------
n_bor, n_ovlp, n_deg = int(ovl["Boruta only"]), int(ovl["Overlap"]), int(ovl["DEG only"])
n_bor_t, n_deg_t = int(ovl["Boruta total"]), int(ovl["DEG total"])
r1, r2, dd = venn_geometry(n_bor + n_ovlp, n_deg + n_ovlp, n_ovlp)
pad = max(r1, r2) * 0.40
axV = fig.add_axes([VX, VY, VW, VH])
axV.set_aspect("equal"); axV.axis("off")
axV.set_xlim(-r1 - pad * 0.7, dd + r2 + pad * 0.7)
axV.set_ylim(-max(r1, r2) - pad * 1.35, max(r1, r2) + pad * 1.95)
axV.add_patch(Circle((0.0, 0.0), r1, facecolor=VOL_BOR, alpha=0.60,
                     edgecolor=VOL_BOR_INK, lw=1.5, zorder=3))
axV.add_patch(Circle((dd, 0.0), r2, facecolor=VOL_DEG, alpha=0.52,
                     edgecolor=VOL_DEG_INK, lw=1.5, zorder=2))

# the lens is 0.27 x 0.70 in at this scale, so the count goes inside it
x_lens = ((dd - r2) + r1) / 2.0
axV.text(-r1 * 0.46, 0, f"{n_bor}", ha="center", va="center", fontsize=15,
         color=VOL_BOR_INK, fontweight="bold", zorder=5)
axV.text(dd + r2 * 0.34, 0, f"{n_deg}", ha="center", va="center", fontsize=15,
         color=VOL_DEG_INK, fontweight="bold", zorder=5)
axV.text(x_lens, 0, f"{n_ovlp}", ha="center", va="center", fontsize=13,
         color=VOL_BOTH, fontweight="bold", zorder=7)

for xx, lbl, n, col in ((-r1 * 0.46, "Boruta", n_bor_t, VOL_BOR_INK),
                        (dd + r2 * 0.34, DE_SHORT, n_deg_t, VOL_DEG_INK)):
    top_y = (r1 if lbl == "Boruta" else r2)
    axV.text(xx, top_y + pad * 0.24, lbl, ha="center", va="bottom",
             fontsize=10.4, color=col, fontweight="bold")
    axV.text(xx, top_y + pad * 0.98, f"n = {n}", ha="center", va="bottom",
             fontsize=9.2, color=MUTED)
axV.text(x_lens, -max(r1, r2) - pad * 0.52, "circle area $\\propto$ set size",
         ha="center", va="center", fontsize=8.6, color=FAINT)

# ---- category key: names only, the Venn already carries the counts -------
key = [(VOL_BG, 5.5, "all genes"),
       (VOL_DEG, 7.0, "differentially expressed" if DE_RULE == "strict"
                      else f"nominal $p$ < {P_THR:g}"),
       (VOL_BOR, 8.5, "Boruta-selected"), (VOL_BOTH, 10.5, "both tracks")]
kx = L
for col, ms, txt in key:
    fig.add_artist(Line2D([kx], [0.033], transform=fig.transFigure, marker="o",
                          ms=ms, color=col, mec="white", mew=0.8, ls="none"))
    fig.text(kx + 0.011, 0.033, txt, ha="left", va="center", fontsize=9.6,
             color=INK if col == VOL_BOTH else MUTED,
             fontweight="bold" if col == VOL_BOTH else "regular")
    kx += 0.012 + len(txt) * 0.0056 + 0.022

# ---- the intersection, set on a label/value grid -------------------------
fig.text(L, 0.958, "Two discovery tracks, one intersection", ha="left",
         va="center", fontsize=12.6, color=INK)
fig.text(L, 0.930, f"{len(de):,} genes tested across {N_PEOPLE} people; "
         f"the {n_bor_t}-gene Boruta panel selected independently of them"
         + ("" if DE_RULE == "strict" else "   ·   no gene reaches FDR < 0.05"),
         ha="left", va="center", fontsize=10.2, color=MUTED)
rule(fig, L, R, 0.912)

fig.text(COL_L, 0.870, "GENES BOTH TRACKS AGREE ON", ha="left", va="bottom",
         fontsize=9.2, color=VOL_BOTH, fontweight="bold")
rule(fig, COL_L, COL_R, 0.862, color=VOL_BOTH, lw=0.9)

LAB_X, VAL_X = COL_L + 0.072, COL_L + 0.080
cards_all = cat_ovl.sort_values("neglog10", ascending=False).reset_index(drop=True)
cards = cards_all.head(5)
EH, ETOP = 0.150, 0.842
if len(cards_all) > 5:
    fig.text(COL_L, 0.055, "also in both: " + ", ".join(sym(g) for g in cards_all["gene"][5:]),
             ha="left", va="center", fontsize=9.2, color=MUTED)
if not len(cards_all):
    fig.text(COL_L, 0.80, "no gene is in both sets", ha="left", va="center",
             fontsize=10.5, color=MUTED)
for k, r in cards.iterrows():
    g  = r["gene"]
    a  = ann4.loc[g] if g in ann4.index else None
    up = r["log2FC"] > 0
    dc = VOL_BOTH if up else VOL_BOR_INK
    y1 = ETOP - k * EH
    fig.text(COL_L, y1 - 0.020, sym(g), ha="left", va="center", fontsize=13.5,
             color=INK, fontweight="bold")
    fig.text(COL_R, y1 - 0.020, "up in PD" if up else "down in PD", ha="right",
             va="center", fontsize=9.6, color=dc, fontweight="bold")
    rows4 = [("log$_2$FC", f"{r['log2FC']:+.2f}"),
             ("adj. $p$" if DE_RULE == "strict" else "$p$", pfmt(r[PCOL]))]
    if a is not None:
        rows4 += [("SHAP rank", f"{int(a['shap_rank'])} of {n_bor_t}"),
                  ("alone, AUC", f"{a['single_gene_auc']:.3f}")]
    for j, (lab, val) in enumerate(rows4):
        yy = y1 - 0.049 - j * 0.0225
        fig.text(LAB_X, yy, lab, ha="right", va="center", fontsize=9.4,
                 color=MUTED)
        fig.text(VAL_X, yy, val, ha="left", va="center", fontsize=9.8, color=INK)
    if k < len(cards) - 1:
        rule(fig, COL_L, COL_R, y1 - EH + 0.026, lw=0.7)
    ax.annotate("", xy=(r["log2FC"], r["neglog10"]), xycoords="data",
                xytext=(COL_L - 0.007, y1 - 0.020), textcoords="figure fraction",
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color="#C6CDD4", lw=0.8,
                                alpha=0.9, shrinkA=2, shrinkB=8,
                                connectionstyle="arc3,rad=0.035"))

finish(fig, "Figure04_venn_volcano",
       "The differential-expression landscape with Boruta selections overlaid, "
       "an area-true Venn of the two tracks, and a dossier for each gene "
       "they agree on")