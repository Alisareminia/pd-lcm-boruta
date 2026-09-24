# ============================== ACCURACY CEILING ==============================
# Two questions on one page: how accurate can these 63 people be classified when
# nothing is tuned on the test fold, and how far away is 80%. The right panel
# answers the second by measuring accuracy at smaller training sizes and fitting
# the curve those points fall on.
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit

A = pd.read_csv(OUTS / "S4_tuned_accuracy.csv")
LC = pd.read_csv(OUTS / "S4_learning_curve.csv")
S4 = json.loads((OUTS / "S4_summary.json").read_text())
BULK = pd.read_csv(OUTS / "S4_bulk_tissue_cv.csv")
MAJ = S4["majority_baseline"]

fig = plt.figure(figsize=(15.0, 8.4))
suptitle(fig, "How high can accuracy honestly go on 63 people")

# ---------------- a: every tuned family --------------------------------
nR = len(A)
ax = fig.add_axes([0.030, 0.085, 0.430, 0.700]); ax.axis("off")
ax.set_xlim(0.30, 1.16); ax.set_ylim(nR - 0.4, -1.5)
for i, r in A.iterrows():
    best = i == 0
    if best:
        ax.add_patch(Rectangle((0.30, i - 0.46), 0.86, 0.92, facecolor="#FBEEEA", edgecolor="none", zorder=0))
    elif i % 2 == 0:
        ax.add_patch(Rectangle((0.30, i - 0.46), 0.86, 0.92, facecolor="#F5F7F9", edgecolor="none", zorder=0))
    a, sd = float(r["accuracy"]), float(r["accuracy_sd"])
    ax.text(0.325, i, r["family"], ha="left", va="center", fontsize=10.0, color=INK,
            fontweight="bold" if best else "regular")
    ax.plot([a - sd, a + sd], [i, i], color=STEEL, lw=2.0, solid_capstyle="round", zorder=3)
    ax.plot([a], [i], marker="D", ms=7.5, color=DEEP_RED if best else NAVY, mec="white", mew=1.1, ls="none", zorder=4)
    ax.text(1.14, i, f"{a:.3f}", ha="right", va="center", fontsize=9.6,
            color=INK if best else MUTED, fontweight="bold" if best else "regular", zorder=4)
ax.plot([MAJ, MAJ], [-0.55, nR - 0.5], ls=(0, (3, 3)), lw=1.0, color="#8E959B", zorder=2)
ax.text(MAJ, -0.75, f"call everyone PD: {MAJ:.2f}", ha="center", va="bottom", fontsize=9.0, color=MUTED)
ax.plot([0.80, 0.80], [-0.55, nR - 0.5], lw=1.2, color=DEEP_RED, alpha=0.7, zorder=2)
ax.text(0.80, -0.75, "80%", ha="center", va="bottom", fontsize=9.6, color=DEEP_RED, fontweight="bold")
for tick in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    ax.plot([tick, tick], [nR - 0.55, nR - 0.42], color="#8E959B", lw=0.8)
    ax.text(tick, nR - 0.30, f"{tick:g}", ha="center", va="top", fontsize=8.4, color=MUTED)
ax.text(0.73, nR + 0.25, "accuracy, threshold chosen inside the training folds", ha="center", va="top",
        fontsize=9.3, color=MUTED)

# ---------------- b: learning curve ------------------------------------
axB = fig.add_axes([0.545, 0.335, 0.435, 0.450])
n = LC["n_train"].to_numpy(float); acc = LC["acc"].to_numpy(float); sd = LC["acc_sd"].to_numpy(float)
axB.fill_between(n, acc - sd, acc + sd, color=LIGHTBL, alpha=0.45, linewidth=0)
axB.plot(n, acc, color=STEEL, lw=2.2, zorder=4)
axB.plot(n, acc, marker="o", ms=5.5, color=STEEL, mec="white", mew=1.0, ls="none", zorder=5)
NEED = S4.get("people_needed_for_80pct")
def f(x, a_, b_, c_): return a_ - b_ * np.power(x, -c_)
try:
    popt, _ = curve_fit(f, n, acc, p0=[0.8, 1.0, 0.5], maxfev=20000, bounds=([0.5, 0, 0.05], [1.0, 20, 3]))
    xf = np.linspace(n.min(), max(300, (NEED or 300) * 1.15), 400)
    axB.plot(xf, f(xf, *popt), ls=(0, (4, 3)), lw=1.4, color=NAVY, alpha=0.85, zorder=3)
    axB.axhline(popt[0], lw=0.9, color=FOREST, ls=(0, (2, 3)))
    axB.text(xf[-1], popt[0] + 0.004, f"fitted ceiling {popt[0]:.2f}", ha="right", va="bottom",
             fontsize=9.2, color=FOREST)
except Exception:
    popt = None
axB.axhline(0.80, lw=1.2, color=DEEP_RED, alpha=0.8)
axB.text(n.min(), 0.806, "80% accuracy", ha="left", va="bottom", fontsize=9.6, color=DEEP_RED, fontweight="bold")
axB.axhline(MAJ, lw=0.9, color="#8E959B", ls=(0, (3, 3)))
axB.text(n.min(), MAJ - 0.012, "call everyone PD", ha="left", va="top", fontsize=9.0, color=MUTED)
if NEED and np.isfinite(NEED):
    axB.plot([NEED], [0.80], marker="v", ms=9, color=DEEP_RED, mec="white", mew=1.0, ls="none", zorder=6)
    axB.annotate(f"{NEED:.0f} people", xy=(NEED, 0.80), xytext=(NEED, 0.80 - 0.07), ha="center", va="top",
                 fontsize=9.6, color=DEEP_RED, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=DEEP_RED, lw=0.9))
axB.set_xlabel("People used for training")
axB.set_ylabel("Accuracy on the people left out")
axB.set_ylim(0.40, 0.92)
dashgrid(axB)

# ---------------- c: where 80% is reachable ----------------------------
axC = fig.add_axes([0.545, 0.085, 0.435, 0.135]); axC.axis("off")
axC.set_xlim(0, 1); axC.set_ylim(0, 1)
bars = [("laser-captured neurons, 63 people", float(A.loc[0, "accuracy"]), STEEL),
        ("bulk substantia nigra, 25 samples", float(BULK["accuracy"].mean()), DEEP_RED)]
for j, (lab, v, col) in enumerate(bars):
    yb = 0.66 - j * 0.42
    axC.text(0.0, yb, lab, ha="left", va="center", fontsize=9.8, color=INK)
    axC.add_patch(Rectangle((0.44, yb - 0.11), 0.42 * (v - 0.3) / 0.7, 0.22, facecolor=col, alpha=0.85,
                            edgecolor="none", transform=axC.transAxes))
    axC.text(0.88, yb, f"{v:.2f}", ha="left", va="center", fontsize=10.0, color=col, fontweight="bold")
axC.plot([0.44 + 0.42 * (0.80 - 0.3) / 0.7] * 2, [0.06, 0.90], color=DEEP_RED, lw=1.2, alpha=0.7)

# ---------------- headers ----------------------------------------------
for x0, x1, letter, ttl, sub in (
        (0.030, 0.470, "a", "Sixteen tuned model families",
         "hyperparameters and cut-off chosen by inner cross-validation, then scored once on the held-out fold"),
        (0.545, 0.980, "b", "What would reach 80%",
         f"accuracy against training size, and the curve those points fall on")):
    rule(fig, x0, x1, 0.838)
    fig.text(x0, 0.878, f"$\\bf{{{letter}}}$   {ttl}", ha="left", va="center", fontsize=11.4, color=INK)
    fig.text(x0, 0.853, sub, ha="left", va="center", fontsize=9.4, color=MUTED)
fig.text(0.545, 0.255, "$\\bf{c}$   Where 80% is already reachable", ha="left", va="center", fontsize=10.6, color=INK)
fig.text(0.545, 0.232, "same kind of model, cross-validated; bulk tissue separates the groups mostly by neuron loss",
         ha="left", va="center", fontsize=9.2, color=MUTED)

finish(fig, "Accuracy_ceiling",
       "Honest accuracy of every tuned model family, the training size 80% would need, and the bulk-tissue comparison")
print("accuracy figure written")
