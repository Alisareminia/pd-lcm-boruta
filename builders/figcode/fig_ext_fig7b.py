C = Canvas(183.0, 150.0); M = C.M
BOLD = dict(fontweight="bold")
PLAT = {"GSE7621": "U133 Plus 2", "GSE8397": "U133A", "GSE20163": "U133A", "GSE20164": "U133A", "GSE20292": "U133A",
        "GSE49036": "U133 Plus 2", "GSE114517": "RNA-seq", "GSE168496": "RNA-seq"}
MODELS = {"core": ("frozen", "strict", CORE_C, "Core classifier"), "panel": ("panel", "panel_strict", PANEL_C, "Boruta-panel forest")}
def square(x, y_, col, s=12):
    M.scatter([x], [y_], s=s, marker="s", color=col, linewidth=0, zorder=5, clip_on=False)
def ring(x, y_, col, s=12):
    M.scatter([x], [y_], s=s, marker="o", facecolor="white", edgecolor=col, linewidth=0.8, zorder=5, clip_on=False)
def diamond(x, y_, col, s=16):
    M.scatter([x], [y_], s=s, marker="D", color=col, linewidth=0, zorder=5, clip_on=False)
def tick(x, y_, s=18):
    M.scatter([x], [y_], s=s, marker="|", color=NEURO_C, linewidth=1.1, zorder=5, clip_on=False)

# ======================= a: forest plot =======================
C.letter(2.0, 146.0, "a", "Discrimination in each external cohort")
COLX = {"cohort": 3.0, "plat": 17.0, "cp": 36.0, "shared": 46.5}
FX = {"core": (53.0, 30.0), "panel": (112.0, 30.0)}           # forest axes: x0, width (mm)
NX = {"core": (94.0, 106.3), "panel": (153.0, 165.3)}         # AUC [95% CI], strictly independent
XNEU = 175.8
# group headers with coloured bars
for key, (m_rep, m_str, col, name) in MODELS.items():
    x0 = FX[key][0]; x1 = NX[key][1] + 3.5
    C.text((x0 + x1) / 2, 140.6, name, ha="center", fontsize=6.5, color=col, **BOLD)
    M.add_patch(Rectangle((x0, 138.3), x1 - x0, 0.75, facecolor=col, edgecolor="none"))
M.add_patch(Rectangle((XNEU - 4.0, 138.3), 8.0, 0.75, facecolor=NEURO_C, edgecolor="none"))
HY = 134.6
for x, t, ha in ((COLX["cohort"], "Cohort", "left"), (COLX["plat"], "Platform", "left"), (COLX["cp"], "Control / PD", "center"),
                 (COLX["shared"], "Shared\ndonors", "center"), (NX["core"][0], "AUC [95% CI]", "center"), (NX["core"][1], "Strict", "center"),
                 (NX["panel"][0], "AUC [95% CI]", "center"), (NX["panel"][1], "Strict", "center"), (XNEU, "Neuron\nmarkers", "center")):
    C.text(x, HY, t, ha=ha, fontsize=5.9, color=INK, linespacing=1.0)
for key, (x0, w) in FX.items():
    C.text(x0 + w / 2, HY, "AUC", ha="center", fontsize=5.9, color=INK)
C.rule(2.0, 181.0, 131.9, lw=0.5, color=INK)
Y0, STEP = 127.8, 4.6
ROWY = {c: Y0 - i * STEP for i, c in enumerate(COHORTS)}
PYY = ROWY[COHORTS[-1]] - 6.4
YB, YT = PYY - 2.8, Y0 + 2.9
WT = {c: RES.loc[(c, "frozen"), "control"] * RES.loc[(c, "frozen"), "PD"] for c in COHORTS}
wmax = max(WT.values())
AXES = {}
for key, (x0, w) in FX.items():
    ax = C.ax(x0, YB, w, YT - YB); ax.set_xlim(0.12, 1.02); ax.set_ylim(YB, YT)
    ax.spines["left"].set_visible(False); ax.set_yticks([]); ax.patch.set_alpha(0)
    ax.set_xticks([0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0.25", "0.5", "0.75", "1"])
    ax.axvline(0.5, color="#A9AFB6", lw=0.5, ls=(0, (2.2, 1.8)), zorder=0)
    AXES[key] = ax
fmt_ci = lambda a, lo, hi: f"{a:.2f} [{lo:.2f}–{hi:.2f}]"
for c in COHORTS:
    yy = ROWY[c]; r = RES.loc[(c, "frozen")]
    C.text(COLX["cohort"], yy, c, ha="left", fontsize=6.1, color=INK)
    C.text(COLX["plat"], yy, PLAT[c], ha="left", fontsize=5.8, color=MUTED)
    C.text(COLX["cp"], yy, f"{int(r.control)} / {int(r.PD)}", ha="center", fontsize=5.9)
    k = int(OVL.loc[c, "shared_with_discovery"])
    C.text(COLX["shared"], yy, str(k) if k else "0", ha="center", fontsize=5.9, color=PD_C if k else MUTED, **(BOLD if k else {}))
    for key, (m_rep, m_str, col, _) in MODELS.items():
        ax = AXES[key]; a, s_ = RES.loc[(c, m_rep)], RES.loc[(c, m_str)]
        ax.plot([a.ci_lo, a.ci_hi], [yy, yy], color=col, lw=0.75, solid_capstyle="butt", zorder=2)
        ax.scatter([a.auc_neuron_markers], [yy], s=20, marker="|", color=NEURO_C, linewidth=1.1, zorder=3)
        ax.scatter([s_.auc], [yy], s=12, facecolor="white", edgecolor=col, linewidth=0.8, zorder=4)
        ax.scatter([a.auc], [yy], s=7 + 30 * WT[c] / wmax, marker="s", color=col, linewidth=0, zorder=5)
        C.text(NX[key][0], yy, fmt_ci(a.auc, a.ci_lo, a.ci_hi), ha="center", fontsize=5.8)
        C.text(NX[key][1], yy, f2(s_.auc), ha="center", fontsize=5.8, color=MUTED)
    C.text(XNEU, yy, f2(r.auc_neuron_markers), ha="center", fontsize=5.8, color=MUTED)
C.rule(2.0, 181.0, PYY + 3.1, lw=0.35, color="#8C9299")
C.text(COLX["cohort"], PYY, "Pooled", ha="left", fontsize=6.1, **BOLD)
C.text(COLX["plat"], PYY, f"{len(COHORTS)} cohorts", ha="left", fontsize=5.8, color=MUTED)
C.text(COLX["cp"], PYY, f"{POOL['control']} / {POOL['PD']}", ha="center", fontsize=5.9, **BOLD)
C.text(COLX["shared"], PYY, str(int(OVL.shared_with_discovery.sum())), ha="center", fontsize=5.9, color=PD_C, **BOLD)
for key, (m_rep, m_str, col, _) in MODELS.items():
    ax = AXES[key]; a, s_ = A[m_rep], A[m_str]; h = 1.35
    ax.add_patch(Polygon([[a["ci_lo"], PYY], [a["auc"], PYY + h], [a["ci_hi"], PYY], [a["auc"], PYY - h]], closed=True,
                         facecolor=col, edgecolor="none", zorder=4))
    ax.scatter([s_["auc"]], [PYY], s=12, facecolor="white", edgecolor=col, linewidth=0.8, zorder=5)
    ax.scatter([A["neuron"]["auc"]], [PYY], s=20, marker="|", color=NEURO_C, linewidth=1.1, zorder=3)
    C.text(NX[key][0], PYY, fmt_ci(a["auc"], a["ci_lo"], a["ci_hi"]), ha="center", fontsize=5.8, **BOLD)
    C.text(NX[key][1], PYY, f2(s_["auc"]), ha="center", fontsize=5.8, color=MUTED)
C.text(XNEU, PYY, f2(A["neuron"]["auc"]), ha="center", fontsize=5.8, color=MUTED)
# symbol key: every mark in both colours
KY = YB - 10.2
square(4.0, KY, CORE_C, 14); square(6.4, KY, PANEL_C, 14)
C.text(8.4, KY, "Model as reported (trained on all 63 discovery donors); square area $\\propto$ cohort weight", ha="left", fontsize=5.7)
ring(104.0, KY, CORE_C); ring(106.4, KY, PANEL_C)
C.text(108.4, KY, "Strictly independent (retrained without same-source studies)", ha="left", fontsize=5.7)
KY2 = KY - 3.5
diamond(4.0, KY2, CORE_C); diamond(6.4, KY2, PANEL_C)
C.text(8.4, KY2, "Pooled estimate; width = 95% CI", ha="left", fontsize=5.7)
tick(52.4, KY2)
C.text(54.4, KY2, "Neuron-marker score alone", ha="left", fontsize=5.7)
M.plot([85.0, 88.6], [KY2, KY2], color="#A9AFB6", lw=0.5, ls=(0, (2.2, 1.8)))
C.text(90.2, KY2, "Chance (AUC 0.5)", ha="left", fontsize=5.7)
TOP = KY2 - 4.2
C.rule(2.0, 181.0, TOP, lw=0.35, color="#8C9299")

# ======================= b: classification at pre-specified thresholds =======================
C.letter(2.0, TOP - 4.6, "b", "Classification at pre-specified thresholds")
T0 = TOP - 8.4
C.rule(4.0, 123.0, T0, lw=0.6, color=INK)
G1, G2 = (61.0, 71.5, 82.0), (96.0, 106.5, 117.0)
for xs, lab in ((G1, "Threshold 0.5"), (G2, "Out-of-bag threshold")):
    C.text(np.mean(xs), T0 - 2.3, lab, ha="center", fontsize=5.8, color=INK)
    C.rule(xs[0] - 5.0, xs[-1] + 5.0, T0 - 4.1, lw=0.45, color=INK)
    for x, t in zip(xs, ("Accuracy", "Sensitivity", "Specificity")):
        C.text(x, T0 - 6.0, t, ha="center", fontsize=5.5, color=INK)
C.text(44.0, T0 - 6.0, "AUC [95% CI]", ha="center", fontsize=5.5, color=INK)
C.rule(4.0, 123.0, T0 - 8.0, lw=0.35, color="#8C9299")
TROWS = [("frozen", "Core classifier", CORE_C, True), ("strict", "strictly independent", CORE_C, False),
         ("panel", "Boruta-panel forest", PANEL_C, True), ("panel_strict", "strictly independent", PANEL_C, False)]
for i, (key, lab, col, filled) in enumerate(TROWS):
    yy = T0 - 11.0 - i * 4.2
    (square if filled else ring)(6.0, yy, col, 11)
    C.text(8.5, yy, lab, ha="left", fontsize=6.0, color=INK if filled else MUTED, fontweight="bold" if key == "panel" else "normal")
    a = A[key]
    C.text(44.0, yy, f"{a['auc']:.2f} [{a['ci_lo']:.2f}–{a['ci_hi']:.2f}]", ha="center", fontweight="bold" if key == "panel" else "normal")
    for xs, calls in ((G1, POOL["calls@0.5"][key]), (G2, POOL["calls@oob"][key])):
        for x, m in zip(xs, ("accuracy", "sensitivity", "specificity")):
            C.text(x, yy, f2(calls[m]), ha="center")
TB = T0 - 11.0 - 3 * 4.2 - 2.6
C.rule(4.0, 123.0, TB, lw=0.6, color=INK)
TH = SUM["thresholds"]
C.text(4.0, TB - 2.8, f"Pooled over {len(COHORTS)} cohorts ({POOL['people']} donors). Out-of-bag thresholds set on the 63 discovery donors: "
       f"core {TH['core']:.2f}, panel {TH['panel']:.2f}.", ha="left", fontsize=5.4, color=MUTED)
# balanced accuracy, cohort by cohort
axS = C.ax(16.0, 6.6, 106.0, TB - 9.6 - 6.6)
xs = np.arange(len(COHORTS))
for key, col, dx in (("frozen", CORE_C, -0.12), ("panel", PANEL_C, 0.12)):
    v = [RES.loc[(c, key), "balanced_accuracy@oob"] for c in COHORTS]
    axS.scatter(xs + dx, v, s=10, marker="s", color=col, zorder=3, linewidth=0)
axS.axhline(0.5, color="#A9AFB6", lw=0.5, ls=(0, (2.2, 1.8)), zorder=0)
axS.set_xlim(-0.5, len(COHORTS) - 0.5); axS.set_ylim(0.2, 1.02)
axS.set_yticks([0.25, 0.5, 0.75, 1]); axS.set_yticklabels(["0.25", "0.5", "0.75", "1"])
axS.set_xticks(xs); axS.set_xticklabels(COHORTS, fontsize=5.5)
axS.tick_params(axis="x", length=0, pad=2.0)
axS.set_ylabel("Balanced\naccuracy", fontsize=5.7, labelpad=2, linespacing=1.0)
axS.patch.set_alpha(0)
for c_i in range(len(COHORTS)):
    if c_i % 2 == 0:
        axS.axvspan(c_i - 0.5, c_i + 0.5, color="#F0F2F4", lw=0, zorder=-1)
C.text(16.0, TB - 6.4, "Balanced accuracy at the out-of-bag threshold, per cohort", ha="left", fontsize=5.5, color=INK)
for x, col, t in ((83.0, CORE_C, "Core classifier"), (101.0, PANEL_C, "Boruta-panel forest")):
    square(x, TB - 6.4, col, 9); C.text(x + 1.4, TB - 6.4, t, ha="left", fontsize=5.4, color=INK)

# ======================= c: which brains overlap =======================
XC = 128.0
C.letter(XC, TOP - 4.6, "c", "Donor overlap with discovery")
LX0, LX1, RX0, RX1 = 129.5, 146.5, 157.0, 179.5
BH = 3.4
C.text((LX0 + LX1) / 2, TOP - 9.2, "Discovery (laser capture)", ha="center", fontsize=5.6, color=INK)
C.text((RX0 + RX1) / 2, TOP - 9.2, "External (bulk)", ha="center", fontsize=5.6, color=INK)
FAM = [("Same laboratory series", ["GSE20141", "GSE24378"], ["GSE20292", "GSE20163", "GSE20164"]),
       ("Netherlands Brain Bank", ["GSE182622"], ["GSE168496", "GSE49036"]),
       ("Other brain banks", ["GSE169755"], ["GSE7621", "GSE8397", "GSE114517"])]
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
        M.add_patch(Rectangle((XC + 0.5, bottom), 181.0 - XC - 0.5, top - bottom, color="#EEF1F4", lw=0, zorder=0))
        M.add_patch(Rectangle((XC + 0.5, bottom), 0.6, top - bottom, color="#8C9299", lw=0, zorder=1))
    C.text(XC + 2.0, top - 1.3, fam, ha="left", fontsize=5.5, color="#3E454D", fontstyle="italic")
    if len(left) == 1:
        POS[left[0]] = rys[0] if fam != "Other brain banks" else np.mean(rys)
    else:
        POS[left[0]] = np.mean(rys[:2]); POS[left[1]] = rys[2]
    yy -= 1.2
def box(x0, x1, yc, name, right_txt, strong=False):
    M.add_patch(FancyBboxPatch((x0, yc - BH / 2), x1 - x0, BH, boxstyle="square,pad=0",
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
           bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none"), zorder=4)
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
C.text(XC + 6.0, LG + 2.8, "Shared donor IDs (removed before testing)", ha="left", fontsize=5.3, color=MUTED)
M.plot([XC + 1.0, XC + 5.0], [LG, LG], color="#8C9299", lw=0.7, ls=(0, (2, 1.5)))
C.text(XC + 6.0, LG, "IDs compared, none shared  ·  n deposited$\\rightarrow$analysed", ha="left", fontsize=5.3, color=MUTED)
C.save("Figure07_external_multicohort_v2")
