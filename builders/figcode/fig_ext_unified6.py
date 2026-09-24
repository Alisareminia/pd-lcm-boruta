C = Canvas(183.0, 155.0); M = C.M
n_c, n_p = POOL["control"], POOL["PD"]

# ---------------- a: the frozen models, averaged over the eight cohorts ----------------
C.letter(2.0, 151.0, "a", "Frozen models in eight independent bulk cohorts")
axA = C.ax(12.0, 99.0, 46.0, 46.0)
axA.plot([0, 1], [0, 1], color="#B7BDC4", lw=0.5, ls=(0, (2, 2)))
for key, col, ls, lw in (("neuron", NEURO_C, (0, (1.2, 1.2)), 1.0), ("panel", PANEL_C, (0, (3, 1.6)), 0.9), ("core", CORE_C, "-", 1.3)):
    axA.plot(GRID, ROC[key], color=col, ls=ls, lw=lw, solid_joinstyle="miter")
axA.set_xlim(-0.01, 1.01); axA.set_ylim(-0.01, 1.01); axA.set_aspect("equal")
axA.set_xticks([0, 0.5, 1]); axA.set_yticks([0, 0.5, 1]); axA.set_xticklabels(["0", "0.5", "1"]); axA.set_yticklabels(["0", "0.5", "1"])
axA.set_xlabel("False-positive rate", fontsize=6.5, labelpad=2); axA.set_ylabel("True-positive rate", fontsize=6.5, labelpad=2)
KEY = [("core classifier", A["frozen"]["auc"], CORE_C, "-", 1.3, True), ("Boruta-panel forest", A["panel"]["auc"], PANEL_C, (0, (3, 1.6)), 0.9, False),
       ("neuron markers alone", A["neuron"]["auc"], NEURO_C, (0, (1.2, 1.2)), 1.0, False)]
for i, (lab, a, col, ls, lw, b) in enumerate(KEY):
    yy = 0.30 - i * 0.095
    axA.plot([0.22, 0.31], [yy, yy], color=col, ls=ls, lw=lw, transform=axA.transAxes)
    axA.text(0.34, yy, lab, transform=axA.transAxes, ha="left", va="center", fontsize=5.9, color=INK, fontweight="bold" if b else "normal")
    axA.text(1.0, yy, f2(a), transform=axA.transAxes, ha="right", va="center", fontsize=5.9, color=INK, fontweight="bold" if b else "normal")
axA.text(1.0, 0.30 + 0.095, "pooled AUC", transform=axA.transAxes, ha="right", va="center", fontsize=5.5, color=MUTED)
cA = A["frozen"]
C.text(12.0, 87.6, f"{len(COHORTS)} cohorts, {POOL['people']} people ({n_c} control, {n_p} PD); trained only on the 63 laser-capture people",
       fontsize=5.7, color=MUTED)
C.text(12.0, 84.6, f"core classifier AUC {cA['auc']:.2f} (95% CI {cA['ci_lo']:.2f}–{cA['ci_hi']:.2f}), label shuffles within cohorts "
       f"P {pfmt(PERM['core'])}", fontsize=5.7, color=MUTED)
C.text(12.0, 81.6, "curves: each cohort's ROC, averaged with weights equal to its PD–control pairs", fontsize=5.7, color=MUTED)

# ---------------- b: how much of it is neuron loss ----------------
C.letter(72.0, 151.0, "b", "How much of it is neuron loss")
axB = C.ax(82.0, 99.0, 46.0, 46.0)
for lab, col in ((0, CT_C), (1, PD_C)):
    m = P.y.to_numpy() == lab
    axB.scatter(P.neuron_score[m], P.core_centred[m], s=8, color=col, edgecolor="white", linewidth=0.3, zorder=3)
xx = np.linspace(P.neuron_score.min(), P.neuron_score.max(), 50)
axB.plot(xx, np.polyval(np.polyfit(P.neuron_score, P.core_centred, 1), xx), color="#8C9299", lw=0.7, ls=(0, (3, 1.6)), zorder=2)
axB.axhline(0, color="#D5D9DE", lw=0.4, zorder=1)
axB.set_xlabel("dopamine-neuron content (8 marker genes, z)", fontsize=6.3, labelpad=2)
axB.set_ylabel("core classifier score, centred within cohort", fontsize=6.3, labelpad=2)
axB.text(0.98, 0.97, f"Spearman ρ = {RHO['core']:.2f}", transform=axB.transAxes, ha="right", va="top", fontsize=5.8, color=MUTED)
for x, col, t in ((84.0, CT_C, "control"), (97.0, PD_C, "PD")):
    M.scatter([x], [146.2], s=13, color=col, edgecolor="white", linewidth=0.4)
    C.text(x + 1.6, 146.2, t, fontsize=5.8)
TX0, TX1, TY = 139.0, 181.0, 141.5
C.rule(TX0, TX1, TY + 2.2, lw=0.6, color="#3A3F45")
C.text(TX1, TY, "pooled AUC", ha="right", fontsize=5.9, color=MUTED)
C.rule(TX0, TX1, TY - 1.8, lw=0.35, color="#8C9299")
for i, (lab, v, bold) in enumerate(NEURO_TABLE):
    yy = TY - 4.5 - i * 4.0
    C.text(TX0, yy, lab, ha="left", fontsize=5.9, color=INK, fontweight="bold" if bold else "normal")
    C.text(TX1, yy, f2(v), ha="right", fontsize=5.9, color=INK, fontweight="bold" if bold else "normal")
C.rule(TX0, TX1, TY - 4.5 - (len(NEURO_TABLE) - 1) * 4.0 - 2.4, lw=0.6, color="#3A3F45")
C.text(TX0, TY - 4.5 - (len(NEURO_TABLE) - 1) * 4.0 - 5.4, "neuron content removed: score regressed on the", fontsize=5.4, color=MUTED, ha="left")
C.text(TX0, TY - 4.5 - (len(NEURO_TABLE) - 1) * 4.0 - 8.0, "neuron score within each cohort", fontsize=5.4, color=MUTED, ha="left")

# ---------------- c: every Boruta gene, cohort by cohort ----------------
C.rule(2.0, 181.0, 79.0)
C.letter(2.0, 74.5, "c", "Every Boruta gene: effect in the laser-capture cohort and in each bulk cohort")
G = GMT.copy(); G["grp"] = np.where(G.deg, 0, 1)
G = G.sort_values(["grp", "g_lcm"], ascending=[True, False]).reset_index(drop=True)
nG = len(G); X0, X1 = 44.0, 164.0; cw = (X1 - X0) / nG
ch, st = 3.4, 3.8
ROWS = [("laser-capture, pooled", 64.0, dict(zip(G.gene, G.g_lcm)))]
y_ = 64.0 - st - 1.4
EFFC = GBC.set_index(["cohort", "gene"])
for c in COHORTS:
    ROWS.append((c, y_, {g: EFFC.loc[(c, g), "g_adj"] for g in G.gene if (c, g) in EFFC.index})); y_ -= st
y_ -= 1.4
ROWS.append((f"{len(COHORTS)} cohorts pooled", y_, dict(zip(G.gene, G.g_external)))); y_ -= st
ROWS.append(("pooled, neuron content removed", y_, dict(zip(G.gene, G.g_external_neuron_adj))))
GMX = 1.5; norm = Normalize(-GMX, GMX)
lcm = dict(zip(G.gene, G.g_lcm))
for r_i, (lab, yy, vals) in enumerate(ROWS):
    bold = r_i == len(ROWS) - 1
    C.text(X0 - 1.5, yy, lab, ha="right", fontsize=5.9, color=INK, fontweight="bold" if bold else "normal")
    k = n = 0
    for j, g in enumerate(G.gene):
        v = vals.get(g, np.nan)
        face = "#E6E8EB" if not np.isfinite(v) else EFFECT(norm(np.clip(v, -GMX, GMX)))
        M.add_patch(Rectangle((X0 + j * cw, yy - ch / 2), cw, ch, facecolor=face, edgecolor="white", lw=0.6))
        if r_i and np.isfinite(v):
            n += 1; k += int(np.sign(v) == np.sign(lcm[g]))
            if np.sign(v) != np.sign(lcm[g]):
                M.plot([X0 + j * cw + 0.8, X0 + (j + 1) * cw - 0.8], [yy - ch / 2 + 0.6, yy + ch / 2 - 0.6], color="#8C9299", lw=0.4)
    if r_i:
        C.text(181.0, yy, f"{k}/{n}", ha="right", fontsize=6.0, color=INK, fontweight="bold" if bold else "normal")
C.text(181.0, ROWS[0][1], "same sign", ha="right", fontsize=5.7, color=MUTED)
cy0, cy1 = ROWS[1][1] + ch / 2, ROWS[len(COHORTS)][1] - ch / 2
M.plot([30.0, 30.0], [cy1, cy0], color="#9AA1A9", lw=0.5)
for yv in (cy0, cy1):
    M.plot([30.0, 31.0], [yv, yv], color="#9AA1A9", lw=0.5)
C.text(28.2, (cy0 + cy1) / 2, "neuron content removed", ha="center", rotation=90, fontsize=5.6, color=MUTED)
ylab = ROWS[-1][1] - ch / 2 - 0.8
for j, r in G.iterrows():
    C.text(X0 + (j + 0.5) * cw, ylab, r.symbol, ha="center", va="top", rotation=90, fontsize=5.7, fontstyle="italic",
           color=SETS["both" if r.deg else "boruta"][1])
for grp, key in ((0, "both"), (1, "boruta")):
    js = np.where(G.grp.to_numpy() == grp)[0]
    if len(js):
        M.add_patch(Rectangle((X0 + js.min() * cw + 0.2, ROWS[0][1] + ch / 2 + 1.0), (js.max() - js.min() + 1) * cw - 0.4, 0.8,
                              facecolor=SETS[key][0], edgecolor="none"))
        C.text(X0 + (js.min() + js.max() + 1) / 2 * cw, ROWS[0][1] + ch / 2 + 2.4, f"{SETS[key][2]}  ({len(js)})", ha="center",
               va="bottom", color=SETS[key][1])
cax = C.ax(8.0, 2.2, 22.0, 1.6)
cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=EFFECT); cax.set_xticks([]); cax.set_yticks([])
for s_ in cax.spines.values():
    s_.set_linewidth(0.3); s_.set_color("#B7BDC4")
C.text(7.0, 3.0, f"−{GMX:g}", ha="right", fontsize=5.7, color=MUTED)
C.text(31.0, 3.0, f"+{GMX:g}   Hedges' g, PD minus control", ha="left", fontsize=5.7, color=MUTED)
ga = SUM["gene_agreement"]["g_external_neuron_adj"]
C.text(181.0, 3.0, f"struck through: opposite sign to discovery  ·  grey: not measured  ·  bottom row: binomial P {pfmt(ga['binom_p'])}, "
       f"Spearman ρ = {ga['spearman']:.2f}", ha="right", fontsize=5.6, color=MUTED)
C.save("Figure06_external_validation_unified")
