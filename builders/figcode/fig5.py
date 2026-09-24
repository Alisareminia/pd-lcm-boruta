# ============================== FIGURE 5 ==============================
# One panel, twenty curves. The trellis of small frames read cleanly but broke
# the figure into twenty pictures; this is a single plot again, and it stays
# legible because only the seven genes whose bootstrap interval clears chance
# are given a colour and a weight. The other thirteen are drawn as one grey
# thicket, which is honestly what they are. The ranked key is set by hand inside
# the same axes - a matplotlib legend of twenty entries was the reason this
# figure looked machine-made the first time.
sg   = T("07_single_gene_auc").copy()
N_TEST = int(T("02_baseline_rf_performance").set_index("metric").loc["n_test", "value"])
order = sg.sort_values("auc", ascending=False).reset_index(drop=True)
singles = T("15_roc_curves").query("curve_type == 'single_gene'") \
          if HAS_FIGDATA else None

STRONG = [DEEP_RED, NAVY, FOREST, AMBER, PLUM, CORAL, "#2E8B8B"]
sig_mask = order["ci_lo"] > 0.5
colour_of, k = {}, 0
for _, r in order.iterrows():
    if r["ci_lo"] > 0.5 and k < len(STRONG):
        colour_of[r["gene"]] = STRONG[k]; k += 1
    else:
        colour_of[r["gene"]] = "#B6C3CF"
n_sig = int(sig_mask.sum())

fig = plt.figure(figsize=(12.6, 8.4))
suptitle(fig, "Individual discriminative performance of each Boruta gene")
ax = fig.add_axes([0.055, 0.098, 0.925, 0.775])

ax.plot([0, 1], [0, 1], ls=(0, (3, 3)), lw=0.9, color="#C2C7CC", zorder=1)
if singles is not None:
    for _, r in order.iloc[::-1].iterrows():          # weakest drawn first
        cur = singles[singles["curve"] == r["gene"]]
        if not len(cur):
            continue
        strong = r["ci_lo"] > 0.5
        ax.plot(cur["fpr"], cur["tpr"], lw=2.5 if strong else 1.1,
                color=colour_of[r["gene"]], alpha=1.0 if strong else 0.75,
                zorder=5 if strong else 2, solid_joinstyle="round")
    top = singles[singles["curve"] == order.iloc[0]["gene"]]
    if len(top):
        ax.fill_between(top["fpr"], 0, top["tpr"], color=DEEP_RED, alpha=0.07,
                        zorder=1)

ax.set_xlabel("1 $-$ Specificity (FPR)")
ax.set_ylabel("Sensitivity (TPR)")
ax.set_xlim(-0.02, 1.52); ax.set_ylim(-0.02, 1.03)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.spines["bottom"].set_bounds(0, 1)
dashgrid(ax, "y")
for gx in (0.2, 0.4, 0.6, 0.8, 1.0):
    ax.plot([gx, gx], [0, 1], color="#E4E7EA", lw=0.5, zorder=0)

# ---- ranked key, set by hand in the space the curves leave --------------
KX_R, KX_S, KX_N, KX_A = 1.075, 1.090, 1.150, 1.510
ytop, ybot = 0.995, 0.045
ys = np.linspace(ytop, ybot, len(order))
ax.text(KX_N, ytop + 0.055, "gene", ha="left", va="center", fontsize=9.4,
        color=INK, fontweight="bold")
ax.text(KX_A, ytop + 0.055, "AUC", ha="right", va="center", fontsize=9.4,
        color=INK, fontweight="bold")
ax.plot([KX_R - 0.012, KX_A], [ytop + 0.028, ytop + 0.028], color=BORDER, lw=0.8)
for i, r in order.iterrows():
    y = ys[i]
    strong = r["ci_lo"] > 0.5
    c = colour_of[r["gene"]]
    ax.text(KX_R, y, f"{i + 1}", ha="right", va="center", fontsize=8.6,
            color=FAINT)
    ax.plot([KX_S, KX_S + 0.042], [y, y], color=c, lw=2.5 if strong else 1.4,
            solid_capstyle="round")
    ax.text(KX_N, y, sym(r["gene"]), ha="left", va="center", fontsize=9.5,
            color=INK if strong else MUTED,
            fontweight="bold" if strong else "regular")
    ax.text(KX_A, y, f"{r['auc']:.3f}", ha="right", va="center", fontsize=9.5,
            color=c if strong else MUTED, fontweight="bold" if strong else "regular")

rule(fig, 0.055, 0.980, 0.888)
fig.text(0.055, 0.940, "Each gene alone, on the held-out test set",
         ha="left", va="center", fontsize=11.4, color=INK)
fig.text(0.055, 0.913, f"direction-corrected   ·   ranked by AUC   ·   the "
         f"{len(order) - n_sig} genes whose bootstrap interval includes 0.5 are drawn as one "
         f"grey thicket   ·   test n = {N_TEST}", ha="left", va="center", fontsize=9.5, color=MUTED)
fig.text(0.980, 0.940, f"{n_sig} of {len(order)} clear chance on their own",
         ha="right", va="center", fontsize=10.6, color=DEEP_RED,
         fontweight="bold")
fig.text(0.980, 0.913, "colour = bootstrap 95% CI excludes AUC 0.5",
         ha="right", va="center", fontsize=9.5, color=MUTED)

finish(fig, "Figure05_boruta_gene_roc",
       "Individual discriminative performance of every Boruta gene in a single "
       "panel, ranked by AUC")