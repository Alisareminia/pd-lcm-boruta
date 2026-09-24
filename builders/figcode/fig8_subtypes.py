from matplotlib.colors import LinearSegmentedColormap, Normalize
UPC, DNC = PANEL_C, CORE_C
C = Canvas(183.0, 180.0); M = C.M
LT = 2.0
tr = lambda v: np.sign(v) * np.log10(1 + np.abs(v) / LT)       # symmetric log for the lineage score
pfmt2 = lambda p: f"P = {p:.4f}" if p >= 0.0001 else "P < 0.0001"

# ======================= a: PD effect against lineage, every gene =======================
C.letter(2.0, 176.0, "a", "PD effect and dopamine-neuron lineage of every measured gene")
AX0, AY0, AW, AH = 17.0, 108.0, 90.0, 51.0
ax = C.ax(AX0, AY0, AW, AH)
xlo, xhi = tr(-22.0), tr(19.0); ylo, yhi = -1.45, 1.45
ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
p05, p95 = tr(SU["background_p05"]), tr(SU["background_p95"])
ax.add_patch(Rectangle((xlo, ylo), p05 - xlo, -ylo, facecolor=DNC, alpha=0.06, lw=0, zorder=0))
ax.add_patch(Rectangle((p95, 0), xhi - p95, yhi, facecolor=UPC, alpha=0.06, lw=0, zorder=0))
ax.axhline(0, color="#AEB4BB", lw=0.5, zorder=1); ax.axvline(0, color="#AEB4BB", lw=0.5, zorder=1)
for v in (p05, p95):
    ax.axvline(v, color="#C9CED4", lw=0.5, ls=(0, (2, 2)), zorder=1)
ax.scatter(tr(BGL.lineage), BGL.g, s=1.1, color="#9AA1A9", alpha=0.35, lw=0, zorder=2, rasterized=True)
size = lambda r: 16 + (31 - r) * 1.25
for d, col in (("down", DNC), ("up", UPC)):
    s_ = PLN[PLN.direction == d]
    ax.scatter(tr(s_.lineage), s_.g, s=size(s_.shap_rank), color=col, edgecolor="white", linewidth=0.5, zorder=4)
TK = [-16, -8, -4, -2, 0, 2, 4, 8, 16]
ax.set_xticks([tr(v) for v in TK]); ax.set_xticklabels([f"{v:+d}" if v else "0" for v in TK])
ax.set_yticks([-1, -0.5, 0, 0.5, 1]); ax.set_yticklabels(["-1", "-0.5", "0", "0.5", "1"])
ax.set_xlabel("Dopamine-neuron lineage score (CALB1 minus SOX6 subtype markers, symmetric log)", fontsize=6.0, labelpad=2)
ax.set_ylabel("PD effect in discovery neurons (Hedges' g)", fontsize=6.0, labelpad=2)
texts = []
for r in PLN.itertuples():
    texts.append(ax.text(tr(r.lineage), r.g, r.name, fontsize=5.6, fontstyle="italic", color=DNC if r.direction == "down" else UPC,
                         ha="center", va="center", zorder=6))
adjust_text(texts, x=list(tr(PLN.lineage)), y=list(PLN.g), ax=ax, expand=(1.25, 1.5), force_text=(0.4, 0.6),
            force_static=(0.4, 0.6), arrowprops=dict(arrowstyle="-", color="#8C9299", lw=0.35, shrinkA=0, shrinkB=2), max_move=None)
ax.text(xlo + 0.02, ylo + 0.05, "Lower in PD\nSOX6 lineage", fontsize=5.6, color=DNC, ha="left", va="bottom", fontweight="bold",
        linespacing=1.0, zorder=5)
ax.text(xhi - 0.02, 0.06, "Higher in PD\nCALB1 lineage", fontsize=5.6, color=UPC, ha="right", va="bottom", fontweight="bold",
        linespacing=1.0, zorder=5)
# marginal distributions: all genes in grey, panel genes as ticks
axT = C.ax(AX0, AY0 + AH + 0.8, AW, 7.0); axT.set_xlim(xlo, xhi); axT.axis("off")
hb, eb = np.histogram(tr(BGL.lineage), bins=np.linspace(xlo, xhi, 70))
axT.fill_between(np.repeat(eb, 2)[1:-1], 0, np.repeat(np.sqrt(hb), 2), color="#C9CED4", lw=0)
axT.set_ylim(-0.6 * np.sqrt(hb).max(), np.sqrt(hb).max() * 1.05)
for r in PLN.itertuples():
    axT.plot([tr(r.lineage)] * 2, [-0.55 * np.sqrt(hb).max(), -0.08 * np.sqrt(hb).max()], color=DNC if r.direction == "down" else UPC, lw=0.7)
axR = C.ax(AX0 + AW + 0.8, AY0, 7.0, AH); axR.set_ylim(ylo, yhi); axR.axis("off")
hg, eg = np.histogram(BGL.g, bins=np.linspace(ylo, yhi, 60))
axR.fill_betweenx(np.repeat(eg, 2)[1:-1], 0, np.repeat(np.sqrt(hg), 2), color="#C9CED4", lw=0)
axR.set_xlim(-0.6 * np.sqrt(hg).max(), np.sqrt(hg).max() * 1.05)
for r in PLN.itertuples():
    axR.plot([-0.55 * np.sqrt(hg).max(), -0.08 * np.sqrt(hg).max()], [r.g] * 2, color=DNC if r.direction == "down" else UPC, lw=0.7)
C.text(AX0, 171.0, f"Grey: all {len(BGL):,} measured genes.  Coloured: the 30 Boruta genes; dot area: SHAP rank",
       ha="left", fontsize=5.6, color=MUTED)
for k, rk in enumerate((1, 10, 20, 30)):
    M.scatter([AX0 + 81.0 + k * 4.2], [171.0], s=size(rk), color="#6B7178", edgecolor="white", linewidth=0.4)
    C.text(AX0 + 81.0 + k * 4.2, 168.4, str(rk), ha="center", fontsize=4.8, color=MUTED)
KY = 97.0
C.text(AX0, KY, f"Down vs up genes on the lineage axis: {pfmt2(SU['up_vs_down'])} (Mann-Whitney); against genes equally changed in PD: "
       f"{pfmt2(SU['p_vs_DE_matched'])}; dashed lines: 5th and 95th percentiles of all genes", ha="left", fontsize=5.5, color=INK)

# ======================= b: subtype marker overlap, down (left) and up (right) =======================
BX = 124.0
C.letter(BX, 176.0, "b", "Marker overlap by subtype")
L0, L1, R0, R1, LABX = 128.0, 144.0, 165.0, 181.0, 154.5
XMAX = 2.2
sc = lambda v, left: (L1 - (L1 - L0) * v / XMAX) if left else (R0 + (R1 - R0) * v / XMAX)
C.text((L0 + L1) / 2, 167.8, "Lower in PD (12)", ha="center", fontsize=5.8, color=DNC, fontweight="bold")
C.text((R0 + R1) / 2, 167.8, "Higher in PD (18)", ha="center", fontsize=5.8, color=UPC, fontweight="bold")
sbt = SBT.set_index(["subtype", "list"])
ROWY = {}
yy = 162.5
for i, st in enumerate(SUBTYPES):
    if i == 4:
        yy -= 3.2
    if i in (0, 4):
        grp = "SOX6 lineage (vulnerable)" if i == 0 else "CALB1 lineage (resilient)"
        C.text(LABX, yy + 0.6, grp, ha="center", fontsize=5.5, color=DNC if i == 0 else UPC, fontstyle="italic")
        yy -= 3.6
    ROWY[st] = yy; yy -= 4.3
top_b, bot_b = ROWY[SUBTYPES[0]] + 2.4, ROWY[SUBTYPES[-1]] - 2.4
thr = -np.log10(0.05)
for sts in (SUBTYPES[:4], SUBTYPES[4:]):
    ya, yb = ROWY[sts[0]] + 2.3, ROWY[sts[-1]] - 2.3
    if sts[-1] == SUBTYPES[-1]:
        yb = bot_b
    for x in (L1, R0):
        M.plot([x, x], [yb, ya], color="#8C9299", lw=0.5)
    for left in (True, False):
        xv = sc(thr, left)
        M.plot([xv, xv], [yb, ya], color="#8C9299", lw=0.5, ls=(0, (2, 1.6)))
for st in SUBTYPES:
    yy = ROWY[st]
    C.text(LABX, yy, st.replace("_", " "), ha="center", fontsize=5.2, color=INK, fontweight="bold" if st == "SOX6_AGTR1" else "normal")
    for d, left, col in (("down", True, DNC), ("up", False, UPC)):
        r = sbt.loc[(st, d)]
        v = min(-np.log10(r.q_BH), XMAX)
        if r.hits:
            x0, x1 = sorted((sc(0, left), sc(v, left)))
            M.add_patch(Rectangle((x0, yy - 1.3), max(x1 - x0, 0.25), 2.6, facecolor=col if r.q_BH < 0.05 else "white",
                                  edgecolor=col, lw=0.6, zorder=3))
            xt = sc(v, left) + (-0.8 if left else 0.8)
            C.text(xt, yy, str(int(r.hits)), ha="right" if left else "left", fontsize=5.2, color=col, zorder=4)
for left in (True, False):
    for v in (0, 1, 2):
        xv = sc(v, left)
        M.plot([xv, xv], [bot_b - 0.2, bot_b - 1.1], color=INK, lw=0.5)
        C.text(xv, bot_b - 2.6, str(v), ha="center", fontsize=5.4)
    M.plot([sc(0, left), sc(XMAX, left)], [bot_b - 0.2, bot_b - 0.2], color=INK, lw=0.5)
C.text(LABX, bot_b - 5.6, "$-\\log_{10}$ FDR  (dashed: FDR 0.05)", ha="center", fontsize=5.6)
C.text(LABX, bot_b - 8.6, "numbers: panel genes among the subtype's 200 top markers", ha="center", fontsize=5.2, color=MUTED)

# ======================= c: marker strength, gene by subtype =======================
C.rule(2.0, 181.0, 93.2, lw=0.35, color="#8C9299")
C.letter(2.0, 88.8, "c", "Marker strength of each panel gene in each dopamine-neuron subtype (Kamath et al. 2022)")
genes = DOWN + UP
HX0, HX1 = 40.0, 178.0
cw = (HX1 - HX0) / len(genes)
gx = {g: HX0 + (i + 0.5) * cw + (1.2 if g in UP else 0) for i, g in enumerate(genes)}
HX1 += 1.2
# PD effect bars above the heatmap
EB0, EH = 70.0, 9.0
gmap = dict(zip(PLN.name, PLN.g)); gmax = 1.3
M.plot([HX0 - 0.5, HX1 + 0.5], [EB0 + EH / 2] * 2, color="#8C9299", lw=0.4)
for g in genes:
    v = gmap[g]; h = EH / 2 * v / gmax
    M.add_patch(Rectangle((gx[g] - cw * 0.32, EB0 + EH / 2 + min(0, h)), cw * 0.64, abs(h), facecolor=DNC if v < 0 else UPC, lw=0))
C.text(HX0 - 2.0, EB0 + EH / 2, "PD effect (g)", ha="right", fontsize=5.6)
for v, lab in ((gmax, "+1.3"), (-gmax, "-1.3")):
    C.text(HX0 - 2.0, EB0 + EH / 2 + EH / 2 * v / gmax, lab, ha="right", fontsize=4.9, color=MUTED)
for grp, lab, col in ((DOWN, f"Lower in PD ({len(DOWN)})", DNC), (UP, f"Higher in PD ({len(UP)})", UPC)):
    xa, xb = gx[grp[0]] - cw / 2, gx[grp[-1]] + cw / 2
    M.add_patch(Rectangle((xa + 0.2, EB0 + EH + 1.4), xb - xa - 0.4, 0.7, facecolor=col, lw=0))
    C.text((xa + xb) / 2, EB0 + EH + 3.2, lab, ha="center", va="bottom", fontsize=5.8, color=col, fontweight="bold")
# heatmap
RH = 3.7
SOXMAP = LinearSegmentedColormap.from_list("sox", ["#F3F5F7", "#9AAABB", "#4C6583", "#1E3350"])
CALMAP = LinearSegmentedColormap.from_list("cal", ["#F7F3F3", "#C3A09E", "#8C4B52", "#551C28"])
ZMAX = 30.0
hy = {}
y_ = EB0 - 2.2
for i, st in enumerate(SUBTYPES):
    if i == 4:
        y_ -= 1.4
    hy[st] = y_ - RH / 2; y_ -= RH
for st in SUBTYPES:
    cmap = SOXMAP if st.startswith("SOX6") else CALMAP
    yc = hy[st]
    C.text(HX0 - 2.0, yc, st.replace("_", " "), ha="right", fontsize=5.6, fontweight="bold" if st == "SOX6_AGTR1" else "normal")
    for g in genes:
        z = ZP.loc[g, st]
        M.add_patch(Rectangle((gx[g] - cw / 2, yc - RH / 2), cw, RH, facecolor=cmap(min(z, ZMAX) / ZMAX) if z > 0 else "#FFFFFF",
                              edgecolor="white", lw=0.5))
        if z >= 5:
            C.text(gx[g], yc, f"{z:.0f}", ha="center", fontsize=4.4, color="white" if z >= 13 else INK)
C.text(8.5, hy["SOX6_AGTR1"], "lost in PD", ha="left", fontsize=5.0, color=MUTED, fontstyle="italic")
for grp, sts, col in (("SOX6", SUBTYPES[:4], DNC), ("CALB1", SUBTYPES[4:], UPC)):
    ya, yb = hy[sts[0]] + RH / 2, hy[sts[-1]] - RH / 2
    M.plot([5.0, 5.0], [yb + 0.2, ya - 0.2], color=col, lw=1.4, solid_capstyle="butt")
    C.text(3.3, (ya + yb) / 2, grp, ha="center", rotation=90, fontsize=5.6, color=col, fontweight="bold")
# lineage score strip and gene names below
LYS = hy[SUBTYPES[-1]] - RH / 2 - 2.6
lmap = dict(zip(PLN.name, PLN.lineage))
LINMAP = LinearSegmentedColormap.from_list("lin", ["#1E3350", "#4C6583", "#9AAABB", "#EEECE7", "#C3A09E", "#8C4B52", "#551C28"])
for g in genes:
    v = lmap[g]
    M.add_patch(Rectangle((gx[g] - cw / 2, LYS - 1.3), cw, 2.6, facecolor=LINMAP(0.5 + 0.5 * np.clip(tr(v) / tr(16), -1, 1)),
                          edgecolor="white", lw=0.5))
C.text(HX0 - 2.0, LYS, "Lineage score", ha="right", fontsize=5.6)
for g in genes:
    C.text(gx[g], LYS - 2.2, g, ha="center", va="top", rotation=90, fontsize=5.6, fontstyle="italic", color=DNC if g in DOWN else UPC)
# colour keys
KY0 = 3.2
for k, (cmap, lab) in enumerate(((SOXMAP, "SOX6 subtypes"), (CALMAP, "CALB1 subtypes"))):
    cax = C.ax(16.0 + k * 40.0, KY0 - 0.8, 16.0, 1.6)
    cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cmap); cax.set_xticks([]); cax.set_yticks([])
    for s_ in cax.spines.values():
        s_.set_linewidth(0.3); s_.set_color("#B7BDC4")
    C.text(15.2 + k * 40.0, KY0, "0", ha="right", fontsize=5.2, color=MUTED)
    C.text(32.8 + k * 40.0, KY0, f"{ZMAX:.0f}+  {lab}", ha="left", fontsize=5.2, color=MUTED)
C.text(4.0, KY0, "Marker z", ha="left", fontsize=5.4)
cax = C.ax(110.0, KY0 - 0.8, 16.0, 1.6)
cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=LINMAP); cax.set_xticks([]); cax.set_yticks([])
for s_ in cax.spines.values():
    s_.set_linewidth(0.3); s_.set_color("#B7BDC4")
C.text(109.2, KY0, "SOX6", ha="right", fontsize=5.2, color=DNC)
C.text(126.8, KY0, "CALB1   lineage score", ha="left", fontsize=5.2, color=MUTED)
C.text(181.0, KY0, "numbers: marker z >= 5", ha="right", fontsize=5.2, color=MUTED)
C.save("Figure08_panel_subtypes")
