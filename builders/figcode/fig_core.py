# ============================== CORE MODEL ==============================
# The high-AUC Random Forest in one page: how well it separates people it never saw
# (a), which of its 30 components carry the decision (b), and which genes those
# components are built from (c). A gene's share is its components' SHAP importance
# split by squared loading, so panel c describes the model rather than testing genes.
from matplotlib.patches import Rectangle

hp = PERF_ALL.set_index("model").loc[HEAD_NAME]
roc = CORE_ROC
comp = CORE_COMP.sort_values("mean_abs_shap", ascending=False).head(10).reset_index(drop=True)
TOT = float(CORE_COMP["mean_abs_shap"].sum())
genes = CORE_GENES.sort_values("rank").head(20).reset_index(drop=True)
PANEL_SET = set(T("04_boruta_selected_genes")["gene"])

fig = plt.figure(figsize=(15.2, 7.8))
suptitle(fig, "The core Random Forest: performance on unseen people, the components it uses, and the genes behind them")

# ---------------- a: ROC of the out-of-fold scores ------------------------------------------
axA = fig.add_axes([0.045, 0.10, 0.255, 0.60])
axA.plot([0, 1], [0, 1], ls=(0, (3, 3)), lw=0.9, color="#9AA0A6")
axA.fill_between(roc.fpr, 0, roc.tpr, step="post", color=LIGHTBL, alpha=0.35, lw=0)
axA.step(roc.fpr, roc.tpr, where="post", color=DEEP_RED, lw=2.2)
axA.set_xlim(-0.01, 1.01); axA.set_ylim(-0.01, 1.01); axA.set_aspect("equal")
axA.set_xlabel("False-positive rate"); axA.set_ylabel("True-positive rate")
dashgrid(axA)
pm = lambda v, d: f" ± {v:.{d}f}" if np.isfinite(v) else ""
lines = [(f"AUC {hp.cv_auc:.3f}{pm(hp.cv_auc_sd, 3)}", "5 x 5-fold, by person", INK),
         (f"accuracy {hp.cv_accuracy:.2f}{pm(hp.cv_accuracy_sd, 2)}", f"calling everyone PD: {MAJ:.2f}", INK),
         (f"unseen study: AUC {hp.lodo_auc:.2f}", "train on 3 studies, test the 4th", INK)]
if CONF:
    lines += [(f"choice made in-fold: AUC {CONF['nested']['auc']:.2f}", "top 6 variants, inner CV", MUTED),
              (f"random labels: p = {CONF['permutation']['p_auc']:.3f}", f"{CONF['permutation']['n_perm']} shuffles", MUTED)]
for i, (big, small, col) in enumerate(lines):
    yv = 0.43 - i * 0.085
    axA.text(0.97, yv, big, transform=axA.transAxes, ha="right", va="center", fontsize=9.8, color=col,
             fontweight="bold" if i == 0 else "regular")
    axA.text(0.97, yv - 0.036, small, transform=axA.transAxes, ha="right", va="center", fontsize=8.4, color=MUTED)

# ---------------- b: components ---------------------------------------------------------------
axB = fig.add_axes([0.385, 0.10, 0.215, 0.60]); axB.axis("off")
nB = len(comp); axB.set_xlim(0, 1.0); axB.set_ylim(nB - 0.4, -0.6)
vmax = comp.mean_abs_shap.max()
for i, r in comp.iterrows():
    w = 0.62 * r.mean_abs_shap / vmax
    axB.add_patch(Rectangle((0.30, i - 0.30), w, 0.60, facecolor=NAVY if i == 0 else STEEL, alpha=0.9 if i == 0 else 0.6,
                            edgecolor="none"))
    axB.text(0.0, i, r.component, ha="left", va="center", fontsize=10, color=INK, fontweight="bold" if i == 0 else "regular")
    axB.text(0.135, i, f"{100 * r.variance_explained:.1f}% var.", ha="left", va="center", fontsize=8.4, color=MUTED)
    axB.text(0.30 + w + 0.02, i, f"{100 * r.mean_abs_shap / TOT:.0f}%", ha="left", va="center", fontsize=9.2, color=MUTED)
axB.text(0.30, nB - 0.05, "share of the model's SHAP importance", ha="left", va="top", fontsize=8.8, color=MUTED)

# ---------------- c: genes --------------------------------------------------------------------
axC = fig.add_axes([0.665, 0.10, 0.325, 0.60]); axC.axis("off")
nC = len(genes); axC.set_xlim(0, 1.0); axC.set_ylim(nC - 0.4, -0.6)
share = 100 * genes.importance / TOT
smax = share.max()
for i, r in genes.iterrows():
    col = DEEP_RED if r.direction_rho > 0 else STEEL
    inpanel = r.gene in PANEL_SET
    x1 = 0.24 + 0.46 * share[i] / smax
    axC.plot([0.24, x1], [i, i], color=col, lw=1.6, alpha=0.75, solid_capstyle="round")
    axC.plot([x1], [i], marker="o", ms=6.5, color=col, mec=INK if inpanel else "white", mew=1.4 if inpanel else 0.8)
    axC.text(0.0, i, r.symbol, ha="left", va="center", fontsize=9.6, color=INK, fontweight="bold" if inpanel else "regular")
    axC.text(0.985, i, f"{100 * r.top50_fold_frequency:.0f}%", ha="right", va="center", fontsize=8.8, color=MUTED)
axC.text(0.985, -1.05, "in top 50\nof folds", ha="right", va="bottom", fontsize=8.4, color=MUTED, linespacing=1.1)
axC.text(0.24, nC - 0.05, f"share of importance (top gene {smax:.2f}%)", ha="left", va="top", fontsize=8.8, color=MUTED)

# ---------------- headers -----------------------------------------------------------------------
for x0, x1, letter, ttl, sub in (
        (0.045, 0.300, "a", "People it never saw", "averaged out-of-fold scores, all 63 people"),
        (0.385, 0.600, "b", "Components that decide", "TreeSHAP of the forest, top 10 of 30"),
        (0.665, 0.990, "c", "Genes behind them", "red: higher within-person rank looks more PD-like")):
    rule(fig, x0, x1, 0.775)
    fig.text(x0, 0.835, f"$\\bf{{{letter}}}$   {ttl}", ha="left", va="center", fontsize=11.4, color=INK)
    fig.text(x0, 0.805, sub, ha="left", va="center", fontsize=9.2, color=MUTED)
n_ag = AGREE["overlap_with_core_topN"] if AGREE else None
note = ("Within-person gene ranks are compressed into 30 principal components on the training people only; the forest "
        "splits on the components. A gene's share is each component's SHAP importance split in proportion to the squared "
        "loadings, so the signal is spread thinly over many genes.")
if AGREE:
    note += (f" Bold genes with a dark ring are also in the {AGREE['panel_size']}-gene Boruta panel "
             f"({n_ag} of its genes fall in this model's top {AGREE['panel_size']}; chance {AGREE['expected_by_chance']:.2f}).")
fig.text(0.045, 0.018, note, ha="left", va="center", fontsize=8.9, color=MUTED, wrap=True)

finish(fig, "Figure_core_random_forest",
       "Performance of the core Random Forest on unseen people, its most important components, and the genes that build them")
