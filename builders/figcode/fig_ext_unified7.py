C = Canvas(183.0, 150.0); M = C.M
PLAT = {"GSE7621": "array (U133 Plus 2)", "GSE8397": "array (U133A)", "GSE20163": "array (U133A)", "GSE20164": "array (U133A)",
        "GSE20292": "array (U133A)", "GSE49036": "array (U133 Plus 2)", "GSE114517": "RNA-seq", "GSE168496": "RNA-seq"}
def mark(x, y_, color, filled=True, size=11):
    M.scatter([x], [y_], s=size, facecolor=color if filled else "white", edgecolor=color, linewidth=0.8, zorder=5, clip_on=False)

# ======================= a: forest plot, every cohort alike =======================
C.letter(2.0, 145.0, "a", "The frozen models, cohort by cohort")
Y0, STEP = 131.0, 4.9
ROWY = {c: Y0 - i * STEP for i, c in enumerate(COHORTS)}
PYY = ROWY[COHORTS[-1]] - 7.0
YB, YT = PYY - 3.2, Y0 + 3.2
AX = {"core": (76.0, 34.0), "panel": (127.0, 33.0)}
NUMX = {"core": (114.0, 120.0), "panel": (164.3, 170.2)}
XNEU = 177.5
HY = YT + 2.4
for x, t, ha in ((4.0, "cohort", "left"), (21.0, "platform", "left"), (52.5, "control / PD", "center"), (65.5, "shared donors\nremoved", "center")):
    C.text(x, HY, t, ha=ha, va="bottom", fontsize=5.9, color=MUTED, linespacing=1.05)
for key, (x0, w) in AX.items():
    col = CORE_C if key == "core" else PANEL_C
    C.text(x0 + w / 2, HY + 3.4, "core classifier" if key == "core" else "Boruta-panel forest", ha="center", va="bottom",
           fontsize=6.6, color=col, fontweight="bold")
    C.text(x0 + w / 2, HY, "AUC", ha="center", va="bottom", fontsize=5.9, color=MUTED)
    mark(NUMX[key][0], HY + 1.0, col, True, 9); mark(NUMX[key][1], HY + 1.0, col, False, 9)
M.scatter([XNEU], [HY + 1.0], s=16, marker="|", color=NEURO_C, linewidth=1.1, clip_on=False)
C.text(XNEU, HY + 3.0, "neuron\nmarkers", ha="center", va="bottom", fontsize=5.6, color=MUTED, linespacing=1.0)
C.rule(2.0, 181.0, YT + 0.9, lw=0.6, color="#30343A")
AXES = {}
for key, (x0, w) in AX.items():
    ax = C.ax(x0, YB, w, YT - YB); ax.set_xlim(0.12, 1.02); ax.set_ylim(YB, YT)
    ax.spines["left"].set_visible(False); ax.set_yticks([])
    ax.set_xticks([0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0.25", "0.5", "0.75", "1"])
    ax.axvline(0.5, color="#B7BDC4", lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax.patch.set_alpha(0); AXES[key] = ax
    C.text(x0 + w / 2, YB - 5.4, "AUC in the external cohort", ha="center", va="top")
for i, c in enumerate(COHORTS):
    if i % 2 == 0:
        M.add_patch(Rectangle((2.0, ROWY[c] - STEP / 2), 179.0, STEP, color="#F5F4F1", lw=0, zorder=0))
for c in COHORTS:
    yy = ROWY[c]; r = RES.loc[(c, "frozen")]
    C.text(4.0, yy, c, ha="left", fontsize=6.2)
    C.text(21.0, yy, PLAT[c], ha="left", fontsize=5.9, color=MUTED)
    C.text(52.5, yy, f"{int(r.control)} / {int(r.PD)}", ha="center")
    k = int(OVL.loc[c, "shared_with_discovery"])
    C.text(65.5, yy, str(k) if k else "–", ha="center", color=INK if k else MUTED, fontweight="bold" if k else "normal")
    for key, (m_rep, m_str) in (("core", ("frozen", "strict")), ("panel", ("panel", "panel_strict"))):
        ax = AXES[key]; col = CORE_C if key == "core" else PANEL_C
        a, s_ = RES.loc[(c, m_rep)], RES.loc[(c, m_str)]
        ax.plot([a.ci_lo, a.ci_hi], [yy, yy], color=col, lw=0.8, solid_capstyle="butt", zorder=2)
        ax.scatter([a.auc_neuron_markers], [yy], s=22, marker="|", color=NEURO_C, linewidth=1.1, zorder=3)
        ax.scatter([s_.auc], [yy], s=13, facecolor="white", edgecolor=col, linewidth=0.8, zorder=4)
        ax.scatter([a.auc], [yy], s=13, color=col, zorder=5, linewidth=0)
        C.text(NUMX[key][0], yy, f2(a.auc), ha="center")
        C.text(NUMX[key][1], yy, f2(s_.auc), ha="center", color=MUTED)
    C.text(XNEU, yy, f2(r.auc_neuron_markers), ha="center", color=MUTED)
C.rule(2.0, 181.0, PYY + STEP / 2 + 0.6, lw=0.5, color="#30343A")
C.text(4.0, PYY, f"pooled, {len(COHORTS)} cohorts", ha="left", fontsize=6.2, fontweight="bold")
C.text(52.5, PYY, f"{POOL['control']} / {POOL['PD']}", ha="center", fontweight="bold")
C.text(65.5, PYY, str(int(OVL.shared_with_discovery.sum())), ha="center", fontweight="bold")
for key, (m_rep, m_str) in (("core", ("frozen", "strict")), ("panel", ("panel", "panel_strict"))):
    ax = AXES[key]; col = CORE_C if key == "core" else PANEL_C
    a, s_ = A[m_rep], A[m_str]; h = 1.45
    ax.add_patch(Polygon([[a["ci_lo"], PYY], [a["auc"], PYY + h], [a["ci_hi"], PYY], [a["auc"], PYY - h]], closed=True,
                         facecolor=col, edgecolor="none", zorder=4))
    ax.scatter([s_["auc"]], [PYY], s=13, facecolor="white", edgecolor=col, linewidth=0.8, zorder=5)
    ax.scatter([A["neuron"]["auc"]], [PYY], s=22, marker="|", color=NEURO_C, linewidth=1.1, zorder=3)
    C.text(NUMX[key][0], PYY, f2(a["auc"]), ha="center", fontweight="bold")
    C.text(NUMX[key][1], PYY, f2(s_["auc"]), ha="center", color=MUTED)
C.text(XNEU, PYY, f2(A["neuron"]["auc"]), ha="center", color=MUTED)
KY = YB - 11.0
mark(4.8, KY, "#4A4F56", True, 11); C.text(6.8, KY, "model as reported (trained on all 63 discovery people), with 95% CI", ha="left", fontsize=5.9)
mark(84.8, KY, "#4A4F56", False, 11)
C.text(86.8, KY, "strictly independent: same recipe, retrained without the discovery studies of the same source (c)", ha="left", fontsize=5.9)
M.scatter([4.8], [KY - 3.6], s=22, marker="|", color=NEURO_C, linewidth=1.1)
C.text(6.8, KY - 3.6, "8 dopamine-neuron marker genes alone (fewer neurons = more PD-like)", ha="left", fontsize=5.9)
C.text(84.0, KY - 3.6, "pooled: cohort AUCs weighted by PD–control pairs, each donor once; diamond width = 95% CI", ha="left",
       fontsize=5.6, color=MUTED)
TOP = KY - 7.0
C.rule(2.0, 181.0, TOP, lw=0.4)

# ======================= b: calls at cut-offs fixed before testing =======================
C.letter(2.0, TOP - 4.6, "b", "Calls at cut-offs fixed before testing")
T0 = TOP - 8.4
C.rule(4.0, 123.0, T0, lw=0.6, color="#30343A")
G1, G2 = (61.0, 71.5, 82.0), (96.0, 106.5, 117.0)
for xs, lab in ((G1, "cut-off 0.5"), (G2, "discovery out-of-bag cut-off")):
    C.text(np.mean(xs), T0 - 2.3, lab, ha="center", fontsize=5.8, color=INK)
    C.rule(xs[0] - 5.0, xs[-1] + 5.0, T0 - 4.1, lw=0.35, color="#8C9299")
    for x, t in zip(xs, ("accuracy", "sensitivity", "specificity")):
        C.text(x, T0 - 6.0, t, ha="center", fontsize=5.5, color=MUTED)
C.text(44.0, T0 - 6.0, "AUC (95% CI)", ha="center", fontsize=5.5, color=MUTED)
C.rule(4.0, 123.0, T0 - 8.0, lw=0.35, color="#8C9299")
TROWS = [("frozen", "core classifier", CORE_C, True), ("strict", "strictly independent", CORE_C, False),
         ("panel", "Boruta-panel forest", PANEL_C, True), ("panel_strict", "strictly independent", PANEL_C, False)]
for i, (key, lab, col, filled) in enumerate(TROWS):
    yy = T0 - 11.0 - i * 4.2
    mark(6.0, yy, col, filled, 10)
    C.text(8.5, yy, lab, ha="left", fontsize=6.0, color=INK if filled else MUTED, fontweight="bold" if key == "panel" else "normal")
    a = A[key]
    C.text(44.0, yy, f"{a['auc']:.2f} ({a['ci_lo']:.2f}–{a['ci_hi']:.2f})", ha="center", fontweight="bold" if key == "panel" else "normal")
    for xs, calls in ((G1, POOL["calls@0.5"][key]), (G2, POOL["calls@oob"][key])):
        for x, m in zip(xs, ("accuracy", "sensitivity", "specificity")):
            C.text(x, yy, f2(calls[m]), ha="center")
TB = T0 - 11.0 - 3 * 4.2 - 2.6
C.rule(4.0, 123.0, TB, lw=0.6, color="#30343A")
TH = SUM["thresholds"]
C.text(4.0, TB - 2.8, f"Pooled over {len(COHORTS)} cohorts ({POOL['people']} people). Out-of-bag cut-offs set on the 63 discovery people: "
       f"core {TH['core']:.2f}, panel {TH['panel']:.2f}.", ha="left", fontsize=5.4, color=MUTED)
# balanced accuracy, cohort by cohort
axS = C.ax(16.0, 6.6, 106.0, TB - 9.6 - 6.6)
xs = np.arange(len(COHORTS))
for key, col, dx in (("frozen", CORE_C, -0.12), ("panel", PANEL_C, 0.12)):
    v = [RES.loc[(c, key), "balanced_accuracy@oob"] for c in COHORTS]
    axS.scatter(xs + dx, v, s=11, color=col, zorder=3, linewidth=0)
axS.axhline(0.5, color="#B7BDC4", lw=0.5, ls=(0, (2, 2)), zorder=0)
axS.set_xlim(-0.5, len(COHORTS) - 0.5); axS.set_ylim(0.2, 1.02)
axS.set_yticks([0.25, 0.5, 0.75, 1]); axS.set_yticklabels(["0.25", "0.5", "0.75", "1"])
axS.set_xticks(xs); axS.set_xticklabels(COHORTS, fontsize=5.5)
axS.tick_params(axis="x", length=0, pad=2.0)
axS.set_ylabel("balanced\naccuracy", fontsize=5.7, labelpad=2, linespacing=1.0)
axS.patch.set_alpha(0)
for c_i in range(len(COHORTS)):
    if c_i % 2 == 0:
        axS.axvspan(c_i - 0.5, c_i + 0.5, color="#F5F4F1", lw=0, zorder=-1)
C.text(16.0, TB - 6.4, "each cohort, balanced accuracy at the out-of-bag cut-off:", ha="left", fontsize=5.4, color=MUTED)
for x, col, t in ((72.0, CORE_C, "core classifier"), (89.0, PANEL_C, "Boruta-panel forest")):
    M.scatter([x], [TB - 6.4], s=9, color=col); C.text(x + 1.3, TB - 6.4, t, ha="left", fontsize=5.4, color=MUTED)

# ======================= c: which brains overlap =======================
XC = 128.0
C.letter(XC, TOP - 4.6, "c", "Shared brains, found and removed")
LX0, LX1, RX0, RX1 = 129.5, 146.5, 157.0, 179.5
BH = 3.4
C.text((LX0 + LX1) / 2, TOP - 9.2, "discovery (laser capture)", ha="center", fontsize=5.6, color=MUTED)
C.text((RX0 + RX1) / 2, TOP - 9.2, "external (bulk)", ha="center", fontsize=5.6, color=MUTED)
FAM = [("one laboratory series", ["GSE20141", "GSE24378"], ["GSE20292", "GSE20163", "GSE20164"]),
       ("Netherlands Brain Bank", ["GSE182622"], ["GSE168496", "GSE49036"]),
       ("other brain banks", ["GSE169755"], ["GSE7621", "GSE8397", "GSE114517"])]
EXT_N = {c: int(OVL.loc[c, "people"]) for c in COHORTS}
KEPT = {c: int(RES.loc[(c, "frozen"), "n"]) for c in COHORTS}
yy = TOP - 12.2
POS = {}
for f_i, (fam, left, right) in enumerate(FAM):
    top = yy
    yy -= 2.8
    rys = []
    for c in right:
        POS[c] = (yy - BH / 2); rys.append(yy - BH / 2); yy -= BH + 0.9
    bottom = yy + 0.9 - 0.8
    if f_i < 2:
        M.add_patch(Rectangle((XC + 0.5, bottom), 181.0 - XC - 0.5, top - bottom, color="#F3F1EC", lw=0, zorder=0))
    C.text(XC + 1.5, top - 1.3, fam, ha="left", fontsize=5.5, color=MUTED, fontstyle="italic")
    if len(left) == 1:
        POS[left[0]] = rys[0] if fam != "other brain banks" else np.mean(rys)
    else:
        POS[left[0]] = np.mean(rys[:2]); POS[left[1]] = rys[2]
    yy -= 1.2
def box(x0, x1, yc, name, right_txt, strong=False):
    M.add_patch(FancyBboxPatch((x0, yc - BH / 2), x1 - x0, BH, boxstyle="round,pad=0,rounding_size=0.6",
                               facecolor="white", edgecolor="#30343A" if strong else "#9AA1A9", lw=0.5, zorder=3))
    C.text(x0 + 1.0, yc, name, ha="left", fontsize=5.5, zorder=4)
    C.text(x1 - 1.0, yc, right_txt, ha="right", fontsize=5.4, color=MUTED, zorder=4)
for c, n in DISC_N.items():
    box(LX0, LX1, POS[c], c, str(n))
for c in COHORTS:
    box(RX0, RX1, POS[c], c, f"{EXT_N[c]}$\\rightarrow${KEPT[c]}" if KEPT[c] != EXT_N[c] else str(EXT_N[c]), strong=KEPT[c] != EXT_N[c])
def link(y0, y1, color, ls, label, lw=0.9):
    xa, xb = LX1, RX0; xm = (xa + xb) / 2
    path = MPath([(xa, y0), (xm, y0), (xm, y1), (xb, y1)], [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
    M.add_patch(PathPatch(path, facecolor="none", edgecolor=color, lw=lw, ls=ls, zorder=2))
    C.text(xm, (y0 + y1) / 2, label, ha="center", fontsize=5.8, color=color, fontweight="bold",
           bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none"), zorder=4)
m = MATCH.groupby("cohort").size().to_dict()
link(POS["GSE20141"], POS["GSE20292"], PD_C, "-", str(m.get("GSE20292", 0)), lw=1.4)
link(POS["GSE20141"], POS["GSE20163"], PD_C, "-", str(m.get("GSE20163", 0)), lw=0.9)
link(POS["GSE182622"], POS["GSE168496"], "#8C9299", (0, (2, 1.5)), "0", lw=0.7)
dup = int(OVL.loc["GSE20163", "also_in_another_external_cohort"])
if dup:
    xb = RX1 + 1.0
    M.plot([RX1, xb, xb, RX1], [POS["GSE20292"], POS["GSE20292"], POS["GSE20163"], POS["GSE20163"]], color=PD_C, lw=0.6)
    C.text(xb + 0.6, (POS["GSE20292"] + POS["GSE20163"]) / 2, str(dup), ha="left", fontsize=5.6, color=PD_C, fontweight="bold")
LG = 3.2
M.plot([XC + 1.0, XC + 5.0], [LG + 2.8, LG + 2.8], color=PD_C, lw=1.1)
C.text(XC + 6.0, LG + 2.8, "same donor IDs: removed before testing", ha="left", fontsize=5.3, color=MUTED)
M.plot([XC + 1.0, XC + 5.0], [LG, LG], color="#8C9299", lw=0.7, ls=(0, (2, 1.5)))
C.text(XC + 6.0, LG, "IDs compared, none shared   ·   people before$\\rightarrow$after", ha="left", fontsize=5.3, color=MUTED)
C.save("Figure07_external_multicohort_unified")
