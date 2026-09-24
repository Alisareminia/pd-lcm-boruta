UPC, DNC = PANEL_C, CORE_C
CTC, PDC = CT_C, PD_C
C = Canvas(183.0, 215.0); M = C.M
CO = pd.read_csv(FIG_IN / "composition_scores.csv")
CVT = pd.read_csv(FIG_IN / "composition_cv.csv")
AUCT = pd.read_csv(FIG_IN / "composition_oof_auc.csv")
GEF = pd.read_csv(FIG_IN / "composition_gene_effects.csv")
CSU = json.load(open(FIG_IN / "pathway_composition_summary.json"))
SUMC, LRC = CSU["composition"], CSU["logistic"]
STUDIES = sorted(CO.dataset.unique())

# ======================= a: the neuron mix of every donor =======================
C.letter(2.0, 211.0, "a", "Estimated dopamine-neuron mix of every donor")
axA = C.ax(16.0, 162.0, 74.0, 40.0)
xs = {d: i for i, d in enumerate(STUDIES)}
rngj = np.random.default_rng(3)
for d in STUDIES:
    for lab, col, off in ((0, CTC, -0.17), (1, PDC, 0.17)):
        v = CO[(CO.dataset == d) & (CO.y == lab)].composition_score.to_numpy()
        x = xs[d] + off + rngj.uniform(-0.07, 0.07, len(v))
        axA.scatter(x, v, s=11, color=col, edgecolor="white", linewidth=0.3, zorder=3)
        axA.plot([xs[d] + off - 0.13, xs[d] + off + 0.13], [v.mean()] * 2, color=col, lw=1.1, zorder=4)
axA.axhline(0, color="#C9CED4", lw=0.5, zorder=1)
axA.set_xticks(range(len(STUDIES))); axA.set_xticklabels(STUDIES, fontsize=5.5)
axA.tick_params(axis="x", length=2.0, pad=1.6)
axA.set_xlim(-0.5, len(STUDIES) - 0.5)
axA.set_ylabel("Composition score\n(CALB1-like $-$ SOX6-like)", fontsize=6.0, labelpad=2, linespacing=1.05)
for x, col, t in ((16.0, CTC, "control"), (30.0, PDC, "PD")):
    M.scatter([x], [206.5], s=12, color=col, edgecolor="white", linewidth=0.3); C.text(x + 1.6, 206.5, t, ha="left", fontsize=5.8)
C.text(16.0, 157.0, f"PD donors are more CALB1-like: AUC {SUMC['auc_composition_pd_more_CALB1_like']:.2f}, Mann-Whitney "
       f"P = {SUMC['composition_mwu_p']:.1e}", ha="left", fontsize=5.6, color=INK)
C.text(16.0, 153.6, f"Markers: the {len(CSU['markers']['calb1'])} strongest CALB1-lineage and {len(CSU['markers']['sox6'])} strongest "
       "SOX6-lineage genes of Kamath et al. (2022), panel genes excluded", ha="left", fontsize=5.3, color=MUTED)

# ======================= b: how much the classifier follows the mix =======================
C.letter(100.0, 211.0, "b", "Classifier score against the mix")
axB = C.ax(114.0, 162.0, 62.0, 40.0)
for lab, col, t in ((0, CTC, "control"), (1, PDC, "PD")):
    s_ = CO[CO.y == lab]
    axB.scatter(s_.composition_score, s_.core_oof_score, s=12, color=col, edgecolor="white", linewidth=0.3, zorder=3)
xx = np.linspace(CO.composition_score.min(), CO.composition_score.max(), 50)
axB.plot(xx, np.polyval(np.polyfit(CO.composition_score, CO.core_oof_score, 1), xx), color="#8C9299", lw=0.7, ls=(0, (3, 1.6)), zorder=2)
axB.set_xlabel("Composition score", fontsize=6.0, labelpad=2)
axB.set_ylabel("Core classifier, out-of-fold score", fontsize=6.0, labelpad=2)
axB.text(0.03, 0.97, f"Spearman ρ = {SUMC['rho_oof_composition']:.2f}\ncontrols only ρ = {SUMC['rho_oof_composition_controls']:.2f}",
         transform=axB.transAxes, ha="left", va="top", fontsize=5.6, color=MUTED, linespacing=1.15)
C.rule(2.0, 181.0, 149.0, lw=0.35, color="#8C9299")

# ======================= c: does the classifier survive the mix being removed? =======================
C.letter(2.0, 144.5, "c", "The classifier with the neuron mix regressed out of every gene")
ORDER = ["original", "composition (50 markers)", "composition (25 markers)", "composition (100 markers)", "composition (200 markers)",
         "SOX6_AGTR1 markers only", "composition + dopamine purity"]
LBL = {"original": "No adjustment", "composition (50 markers)": "Composition removed (50 markers)",
       "composition (25 markers)": "25 markers", "composition (100 markers)": "100 markers", "composition (200 markers)": "200 markers",
       "SOX6_AGTR1 markers only": "SOX6/AGTR1 markers only", "composition + dopamine purity": "Composition + dopamine purity"}
core_auc = AUCT[AUCT.model == "core"].set_index("variant")
AXC0, AXCW = 68.0, 52.0
y0, step = 137.0, 5.6
axC = C.ax(AXC0, y0 - (len(ORDER) - 1) * step - 3.0, AXCW, (len(ORDER) - 1) * step + 6.0)
axC.set_ylim(y0 - (len(ORDER) - 1) * step - 3.0, y0 + 3.0); axC.set_xlim(0.4, 1.0)
axC.spines["left"].set_visible(False); axC.set_yticks([]); axC.patch.set_alpha(0)
axC.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); axC.tick_params(axis="x", labelsize=5.6)
axC.axvline(0.5, color="#A9AFB6", lw=0.5, ls=(0, (2.2, 1.8)), zorder=0)
for i, v in enumerate(ORDER):
    yy = y0 - i * step
    r = core_auc.loc[v]
    prim = v in ("original", "composition (50 markers)")
    col = INK if v == "original" else (CORE_C if prim else "#6B7178")
    C.text(AXC0 - 2.0, yy, LBL[v], ha="right", fontsize=5.8, fontweight="bold" if prim else "normal",
           color=INK if prim else MUTED)
    axC.plot([r.ci_lo, r.ci_hi], [yy, yy], color=col, lw=0.8, zorder=2)
    axC.scatter([r.oof_auc], [yy], s=22 if prim else 14, color=col, zorder=3, linewidth=0)
    C.text(AXC0 + AXCW + 2.0, yy, f"{r.oof_auc:.2f} [{r.ci_lo:.2f}–{r.ci_hi:.2f}]", ha="left", fontsize=5.6,
           fontweight="bold" if prim else "normal")
    C.text(AXC0 + AXCW + 28.0, yy, ("P < 0.0001" if r.shuffle_p < 0.0001 else f"P = {r.shuffle_p:.4f}"), ha="left", fontsize=5.6,
           color=INK if r.shuffle_p < 0.05 else MUTED)
C.text(AXC0 + AXCW / 2, y0 + 5.6, "Out-of-fold AUC of the core classifier [95% CI]", ha="center", fontsize=5.8)
C.text(AXC0 + AXCW + 2.0, y0 + 5.6, "AUC [95% CI]", ha="left", fontsize=5.5, color=MUTED)
C.text(AXC0 + AXCW + 28.0, y0 + 5.6, "Label shuffles", ha="left", fontsize=5.5, color=MUTED)
ybot = y0 - (len(ORDER) - 1) * step - 9.0        # clear of the axis tick labels
C.text(2.0, ybot, f"Same 5 x 5 folds and the same model as the paper's classifier. Adding the classifier's score to the composition "
       f"raises the leave-one-out AUC from {LRC['loo_auc_composition']:.2f} to {LRC['loo_auc_composition_plus_score']:.2f}"
       + (f" (likelihood-ratio P = {LRC['lr_p']:.4f})." if "lr_p" in LRC else "."), ha="left", fontsize=5.4, color=MUTED)
C.rule(2.0, 181.0, ybot - 4.0, lw=0.35, color="#8C9299")

# ======================= d: gene by gene =======================
C.letter(2.0, ybot - 8.5, "d", "PD effect of each panel gene before and after the mix is removed")
G = GEF.sort_values("g_original").reset_index(drop=True)
DX0, DW, DY0 = 44.0, 100.0, 12.0
DH = ybot - 12.0 - DY0
axD = C.ax(DX0, DY0, DW, DH)
lim = 1.45
ylo_, yhi_ = -0.8, len(G) - 0.2
axD.set_xlim(-lim, lim); axD.set_ylim(ylo_, yhi_)
row_mm = lambda i: DY0 + DH * (i - ylo_) / (yhi_ - ylo_)
axD.spines["left"].set_visible(False); axD.set_yticks([]); axD.patch.set_alpha(0)
axD.axvline(0, color="#A9AFB6", lw=0.5, zorder=1)
axD.set_xlabel("PD effect in the discovery neurons (Hedges' g)", fontsize=6.0, labelpad=2)
axD.tick_params(axis="x", labelsize=5.6)
for i, r in enumerate(G.itertuples()):
    col = UPC if r.direction == "up" else DNC
    axD.plot([r.g_original, r.g_composition_removed], [i, i], color="#C9CED4", lw=0.8, zorder=2)
    axD.scatter([r.g_original], [i], s=13, facecolor="white", edgecolor=col, linewidth=0.8, zorder=3)
    axD.scatter([r.g_composition_removed], [i], s=13, color=col, zorder=4, linewidth=0)
    yy = row_mm(i)
    C.text(DX0 - 2.0, yy, r.symbol, ha="right", fontsize=5.3, fontstyle="italic", color=col)
    C.text(DX0 + DW + 2.0, yy, f"{100 * r.retained:.0f}%", ha="left", fontsize=5.3,
           color=INK if r.retained >= 0.5 else MUTED)
    C.text(DX0 + DW + 12.0, yy, f"{r.lineage:+.1f}" if abs(r.lineage) >= 0.05 else "0", ha="left", fontsize=5.3, color=MUTED)
C.text(DX0 + DW + 2.0, DY0 + DH + 2.2, "Effect\nkept", ha="left", fontsize=5.4, linespacing=1.0)
C.text(DX0 + DW + 12.0, DY0 + DH + 2.2, "Lineage\nscore", ha="left", fontsize=5.4, color=MUTED, linespacing=1.0)
kx = 4.0
M.scatter([kx], [DY0 + 6.0], s=13, facecolor="white", edgecolor="#6B7178", linewidth=0.8)
C.text(kx + 2.0, DY0 + 6.0, "before", ha="left", fontsize=5.4)
M.scatter([kx], [DY0 + 2.6], s=13, color="#6B7178", linewidth=0)
C.text(kx + 2.0, DY0 + 2.6, "after the mix is removed", ha="left", fontsize=5.4)
C.text(2.0, 4.0, f"{int((GEF.retained >= 0.5).sum())} of 30 genes keep at least half of their PD effect; median kept "
       f"{100 * GEF.retained.median():.0f}%.", ha="left", fontsize=5.4, color=MUTED)
C.save("Figure10_neuron_mix")
