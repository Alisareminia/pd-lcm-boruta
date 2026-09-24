# ============================== PERFORMANCE TABLE (RANDOM FORESTS) ==============================
# One line per Random Forest, all scored on the same folds. PCA and Boruta, where
# a row uses them, are refitted inside every training fold, so no row has seen
# its own test people. Two bars carry the two evaluations that matter; the rest is
# printed, because it is read one row at a time.
from matplotlib.patches import Rectangle

CHOSEN = HEAD_NAME
SPLIT = {HEAD_NAME: ("within-person ranks, 30 components", "Random Forest"),
         "Boruta (perc 100) + Random Forest": ("Boruta genes, standard (perc 100)", "Random Forest"),
         "Boruta (perc 99) + Random Forest": ("Boruta genes, broadened (perc 99)", "Random Forest"),
         "Random Forest, all genes": ("all 5,622 genes", "Random Forest")}

L = PERF_ALL.sort_values("cv_auc", ascending=False).reset_index(drop=True)
nR = len(L)
FH = 2.9 + 0.34 * nR
fig = plt.figure(figsize=(15.2, FH))
suptitle(fig, "The core Random Forest against the Boruta panels and a plain forest, on the same folds")
ax = fig.add_axes([0.030, 0.075, 0.952, (FH - 1.75) / FH]); ax.axis("off")

X_R, X_S, X_M = 1.5, 3.2, 25.0
CVB, CVW = 42.0, 12.0
LOB, LOW = 64.0, 12.0
COLS = [("cv_accuracy", "accuracy", 87.0), ("cv_bal_accuracy", "bal. acc", 93.0), ("cv_sensitivity", "sens.", 98.5),
        ("cv_specificity", "spec.", 104.0), ("cv_mcc", "MCC", 109.5), ("cv_brier", "Brier", 115.0)]
ax.set_xlim(0, 120); ax.set_ylim(nR - 0.4, -3.0)
scale = lambda v, x0, w: x0 + w * (np.clip(v, 0.3, 1.0) - 0.3) / 0.7

for i, r in L.iterrows():
    chosen = r["model"] == CHOSEN
    if chosen:
        ax.add_patch(Rectangle((0, i - 0.46), 120, 0.92, facecolor="#FBEEEA", edgecolor="none", zorder=0))
    elif i % 2 == 0:
        ax.add_patch(Rectangle((0, i - 0.46), 120, 0.92, facecolor="#F5F7F9", edgecolor="none", zorder=0))
    sel, clf = SPLIT.get(r["model"], (r["model"], ""))
    ax.text(X_R, i, f"{int(r['rank'])}", ha="right", va="center", fontsize=9.2, color=FAINT)
    ax.text(X_S, i, sel, ha="left", va="center", fontsize=9.8, color=NAVY if chosen else MUTED,
            fontweight="bold" if chosen else "regular")
    ax.text(X_M, i, clf, ha="left", va="center", fontsize=10.0, color=INK, fontweight="bold" if chosen else "regular")
    for key, x0, w, col in (("cv_auc", CVB, CVW, DEEP_RED), ("lodo_auc", LOB, LOW, STEEL)):
        v = float(r[key])
        ax.plot([scale(0.5, x0, w)] * 2, [i - 0.42, i + 0.42], color="#C9CED3", lw=0.8, zorder=1)
        ax.add_patch(Rectangle((scale(0.5, x0, w), i - 0.26), scale(v, x0, w) - scale(0.5, x0, w), 0.52,
                               facecolor=col, alpha=0.9 if chosen else 0.55, edgecolor="none", zorder=3))
        if key == "cv_auc" and np.isfinite(float(r["cv_auc_sd"])):
            sd = float(r["cv_auc_sd"])
            ax.plot([scale(v - sd, x0, w), scale(v + sd, x0, w)], [i, i], color="#5F6469", lw=0.9, zorder=4)
        ax.text(x0 + w + 1.0, i, f"{v:.3f}", ha="left", va="center", fontsize=9.4, color=INK if chosen else MUTED,
                fontweight="bold" if chosen else "regular", zorder=4)
    for key, _, x in COLS:
        ax.text(x, i, f"{float(r[key]):.2f}", ha="right", va="center", fontsize=9.2,
                color=INK if (chosen and key == "cv_accuracy") else MUTED,
                fontweight="bold" if (chosen and key == "cv_accuracy") else "regular", zorder=4)

# ---- column heads --------------------------------------------------------
ax.text(X_R, -1.35, "rank", ha="right", va="center", fontsize=9.2, color=MUTED)
ax.text(X_S, -1.35, "features", ha="left", va="center", fontsize=9.6, color=INK, fontweight="bold")
ax.text(X_S, -1.95, "refitted inside every training fold", ha="left", va="center", fontsize=8.6, color=MUTED)
ax.text(X_M, -1.35, "classifier", ha="left", va="center", fontsize=9.6, color=INK, fontweight="bold")
for x0, w, lab, sub, col in ((CVB, CVW, "cross-validated AUC", "5 x 5-fold by person; rule = 1 SD", DEEP_RED),
                             (LOB, LOW, "leave one study out", "train on 3 studies, test on the 4th", STEEL)):
    ax.text(x0, -2.55, lab, ha="left", va="center", fontsize=9.8, color=col, fontweight="bold")
    ax.text(x0, -1.95, sub, ha="left", va="center", fontsize=8.6, color=MUTED)
    for tick in (0.5, 0.75, 1.0):
        ax.plot([scale(tick, x0, w)] * 2, [-1.05, -0.9], color="#8E959B", lw=0.8)
        ax.text(scale(tick, x0, w), -1.35, f"{tick:g}", ha="center", va="center", fontsize=8.2, color=MUTED)
for key, lab, x in COLS:
    ax.text(x, -1.35, lab, ha="right", va="center", fontsize=9.2, color=INK, fontweight="bold")
ax.plot([0.5, 119.5], [-0.62, -0.62], color=BORDER, lw=0.9)

rC = L.loc[L.model == CHOSEN].iloc[0]
rule(fig, 0.030, 0.982, 1 - 1.30 / FH)
fig.text(0.030, 1 - 0.62 / FH, f"{nR} Random Forests on {N_PEOPLE} people, {N_BG:,} genes", ha="left", va="center",
         fontsize=11.6, color=INK)
pm = lambda v: f" +/- {v:.2f}" if np.isfinite(v) else ""
sub = (f"chosen (tinted): {SPLIT[CHOSEN][0]} + {SPLIT[CHOSEN][1]}   ·   accuracy {rC['cv_accuracy']:.2f}"
       f"{pm(rC['cv_accuracy_sd'])}, AUC {rC['cv_auc']:.3f}   ·   calling everyone PD scores {MAJ:.2f}")
if CONF:
    sub += (f"   ·   choice made in-fold: AUC {CONF['nested']['auc']:.2f}   ·   random labels p = "
            f"{CONF['permutation']['p_auc']:.3f}")
fig.text(0.030, 1 - 0.90 / FH, sub, ha="left", va="center", fontsize=9.6, color=MUTED)
fig.text(0.030, 0.028,
         "Every row is scored on the same 25 cross-validation folds and four leave-one-study-out folds. PCA and Boruta (Random Forest "
         "importance) are refitted from scratch on each training fold. Metrics other than AUC use a 0.5 probability cut.",
         ha="left", va="center", fontsize=9.3, color=MUTED)

finish(fig, "Performance_table_random_forest",
       "Cross-validated and cross-study performance of the core Random Forest, the Boruta panels and a plain forest")

# ---- the same table as CSV and LaTeX ------------------------------------
keep = ["rank", "model", "cv_auc", "cv_auc_sd", "cv_accuracy", "cv_accuracy_sd", "cv_bal_accuracy", "cv_sensitivity",
        "cv_specificity", "cv_f1", "cv_mcc", "cv_brier", "lodo_auc"] + [c for c in L.columns if c.startswith("lodo_auc_")]
tab = L[keep].copy()
tab.to_csv(OUT_DIR / "performance_table.csv", index=False)
lines = [r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{Random Forests on the merged 63-person cohort, ordered by cross-validated AUC "
         r"(5 repeats of stratified 5-fold by person). PCA and Boruta are refitted inside every training fold. "
         r"LODO = leave one study out. Metrics other than AUC use a 0.5 probability cut; the core model is in bold.}",
         r"\label{tab:classifiers}", r"\begin{tabular}{@{}rllccccccccc@{}}", r"\toprule",
         r"\# & Features & Classifier & CV AUC & Accuracy & Bal.\ acc & Sens. & Spec. & F1 & MCC & Brier & LODO AUC \\",
         r"\midrule"]
for _, r in tab.iterrows():
    sel, clf = SPLIT.get(r["model"], (r["model"], ""))
    if r["model"] == CHOSEN:
        sel, clf = f"\\textbf{{{sel}}}", f"\\textbf{{{clf}}}"
    lines.append(f"{int(r['rank'])} & {sel} & {clf} & {r['cv_auc']:.3f} $\\pm$ {r['cv_auc_sd']:.3f} & "
                 f"{r['cv_accuracy']:.2f} & {r['cv_bal_accuracy']:.2f} & {r['cv_sensitivity']:.2f} & {r['cv_specificity']:.2f} & "
                 f"{r['cv_f1']:.2f} & {r['cv_mcc']:.2f} & {r['cv_brier']:.2f} & {r['lodo_auc']:.3f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(OUT_DIR / "performance_table.tex").write_text("\n".join(lines) + "\n")
print("wrote performance_table.csv / .tex")
