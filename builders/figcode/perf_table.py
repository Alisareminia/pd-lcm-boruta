# ============================== PERFORMANCE TABLE ==============================
# One line per candidate classifier, ordered by cross-validated AUC. Two bars
# carry the two evaluations that matter: within-cohort cross-validation by person
# and leave-one-dataset-out. The rest of the columns are printed, not plotted,
# because they are read one row at a time rather than compared across rows.
from matplotlib.patches import Rectangle

S1 = pd.read_csv(OUTS / "S1_candidate_performance.csv")
CHOSEN = "genes | Logistic L2 (C=1)"
nested = json.loads((OUTS / "S1_winner_summary.json").read_text())
permsel = json.loads((OUTS / "S2_chosen_permutation.json").read_text())

L = S1.sort_values("cv_auc", ascending=False).reset_index(drop=True)
nR = len(L)
FH = 2.5 + 0.30 * nR
fig = plt.figure(figsize=(15.2, FH))
suptitle(fig, "Every candidate classifier under the same two tests")
ax = fig.add_axes([0.030, 0.055, 0.952, (FH - 2.05) / FH]); ax.axis("off")

X_R, X_F, X_M = 1.5, 4.0, 15.0
CVB, CVW = 44.0, 12.0
LOB, LOW = 66.0, 12.0
COLS = [("cv_bal_acc", "bal. acc", 88.0), ("cv_sensitivity", "sens.", 94.0), ("cv_specificity", "spec.", 100.0),
        ("cv_f1", "F1", 105.5), ("cv_mcc", "MCC", 111.0), ("cv_brier", "Brier", 117.0)]
ax.set_xlim(0, 120); ax.set_ylim(nR - 0.4, -2.9)
scale = lambda v, x0, w: x0 + w * (np.clip(v, 0.3, 1.0) - 0.3) / 0.7

FEAT_C = {"genes": NAVY, "pathways": FOREST, "cell types": AMBER}
for i, r in L.iterrows():
    chosen = r["candidate"] == CHOSEN
    if chosen:
        ax.add_patch(Rectangle((0, i - 0.46), 120, 0.92, facecolor="#FBEEEA", edgecolor="none", zorder=0))
    elif i % 2 == 0:
        ax.add_patch(Rectangle((0, i - 0.46), 120, 0.92, facecolor="#F5F7F9", edgecolor="none", zorder=0))
    ax.text(X_R, i, f"{int(r['rank'])}", ha="right", va="center", fontsize=9.2, color=FAINT)
    ax.text(X_F, i, r["features"], ha="left", va="center", fontsize=9.8, color=FEAT_C[r["features"]])
    ax.text(X_M, i, r["model"], ha="left", va="center", fontsize=10.2, color=INK,
            fontweight="bold" if chosen else "regular")
    for key, x0, w, col in (("cv_auc", CVB, CVW, DEEP_RED), ("lodo_auc", LOB, LOW, STEEL)):
        v = float(r[key])
        ax.plot([scale(0.5, x0, w), scale(0.5, x0, w)], [i - 0.42, i + 0.42], color="#C9CED3", lw=0.8, zorder=1)
        ax.add_patch(Rectangle((scale(0.5, x0, w), i - 0.26), scale(v, x0, w) - scale(0.5, x0, w), 0.52,
                               facecolor=col, alpha=0.9 if chosen else 0.55, edgecolor="none", zorder=3))
        if key == "cv_auc":
            sd = float(r["cv_auc_sd"])
            ax.plot([scale(v - sd, x0, w), scale(v + sd, x0, w)], [i, i], color="#5F6469", lw=0.9, zorder=4)
        ax.text(x0 + w + 1.0, i, f"{v:.3f}", ha="left", va="center", fontsize=9.4, color=INK if chosen else MUTED,
                fontweight="bold" if chosen else "regular", zorder=4)
    for key, _, x in COLS:
        ax.text(x, i, f"{float(r[key]):.2f}", ha="right", va="center", fontsize=9.2, color=MUTED, zorder=4)

# ---- column heads --------------------------------------------------------
ax.text(X_R, -1.35, "rank", ha="right", va="center", fontsize=9.2, color=MUTED)
ax.text(X_F, -1.35, "features", ha="left", va="center", fontsize=9.6, color=INK, fontweight="bold")
ax.text(X_M, -1.35, "model", ha="left", va="center", fontsize=9.6, color=INK, fontweight="bold")
for x0, w, lab, sub, col in ((CVB, CVW, "cross-validated AUC", "10 x 5-fold by person; rule = 1 SD", DEEP_RED),
                             (LOB, LOW, "leave one dataset out", "train on 3 studies, test on the 4th", STEEL)):
    ax.text(x0, -2.05, lab, ha="left", va="center", fontsize=9.8, color=col, fontweight="bold")
    ax.text(x0, -1.70, sub, ha="left", va="center", fontsize=8.6, color=MUTED)
    for tick in (0.5, 0.75, 1.0):
        ax.plot([scale(tick, x0, w)] * 2, [-1.05, -0.9], color="#8E959B", lw=0.8)
        ax.text(scale(tick, x0, w), -1.35, f"{tick:g}", ha="center", va="center", fontsize=8.2, color=MUTED)
for key, lab, x in COLS:
    ax.text(x, -1.35, lab, ha="right", va="center", fontsize=9.2, color=INK, fontweight="bold")
ax.plot([0.5, 119.5], [-0.62, -0.62], color=BORDER, lw=0.9)

rule(fig, 0.030, 0.982, 1 - 1.30 / FH)
fig.text(0.030, 1 - 0.62 / FH, f"{nR} classifiers on {N_PEOPLE} people, {N_BG:,} genes", ha="left", va="center",
         fontsize=11.6, color=INK)
fig.text(0.030, 1 - 0.90 / FH,
         f"chosen classifier (tinted): {CHOSEN.split(' | ')[1]} on all genes, picked in 9 of 15 nested folds   ·   "
         f"cross-validated AUC {float(L.loc[L.candidate == CHOSEN, 'cv_auc'].iloc[0]):.3f}, "
         f"label-permutation null {permsel['null_mean']:.3f} (95th pct {permsel['null_95th']:.3f}), p = {permsel['perm_p']:.3f}",
         ha="left", va="center", fontsize=9.6, color=MUTED)
fig.text(0.030, 0.020,
         f"Choosing the best of these {nR} candidates is itself a fitted step: nested cross-validation of that choice gives "
         f"AUC {nested['nested_selection_auc_mean']:.3f} +/- {nested['nested_selection_auc_sd']:.3f}. "
         "Pathway scores are label-free mean z-scores of GO, Reactome and WikiPathways gene sets; cell-type scores are "
         "mean z-scores of marker genes. Metrics other than AUC use a 0.5 probability cut.",
         ha="left", va="center", fontsize=9.3, color=MUTED, linespacing=1.5)

finish(fig, "Performance_table_classifiers",
       "Cross-validated and cross-dataset performance of every candidate classifier")

# ---- the same table as CSV and LaTeX ------------------------------------
keep = ["rank", "features", "model", "cv_auc", "cv_auc_sd", "cv_bal_acc", "cv_sensitivity", "cv_specificity",
        "cv_f1", "cv_mcc", "cv_brier", "lodo_auc", "lodo_auc_GSE182622", "lodo_auc_GSE20141",
        "lodo_auc_GSE24378", "lodo_auc_GSE169755"]
tab = L[keep].copy()
tab.to_csv(OUTS / "S3_performance_table.csv", index=False)
lines = [r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{Candidate classifiers on the merged 63-person cohort, ordered by cross-validated AUC "
         r"(10 repeats of stratified 5-fold by person). LODO = leave one dataset out. Metrics other than AUC use a "
         r"0.5 probability cut. The chosen classifier is in bold; nested cross-validation of the choice itself gives "
         f"AUC {nested['nested_selection_auc_mean']:.3f} $\\pm$ {nested['nested_selection_auc_sd']:.3f}.}}",
         r"\label{tab:classifiers}", r"\begin{tabular}{@{}rllcccccccc@{}}", r"\toprule",
         r"\# & Features & Model & CV AUC & Bal.\ acc & Sens. & Spec. & F1 & MCC & Brier & LODO AUC \\", r"\midrule"]
for _, r in tab.iterrows():
    nm = f"\\textbf{{{r['model']}}}" if f"{r['features']} | {r['model']}" == CHOSEN else r["model"]
    lines.append(f"{int(r['rank'])} & {r['features']} & {nm} & {r['cv_auc']:.3f} $\\pm$ {r['cv_auc_sd']:.3f} & "
                 f"{r['cv_bal_acc']:.2f} & {r['cv_sensitivity']:.2f} & {r['cv_specificity']:.2f} & {r['cv_f1']:.2f} & "
                 f"{r['cv_mcc']:.2f} & {r['cv_brier']:.2f} & {r['lodo_auc']:.3f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(OUTS / "S3_performance_table.tex").write_text("\n".join(lines) + "\n")
print("wrote S3_performance_table.csv / .tex")
