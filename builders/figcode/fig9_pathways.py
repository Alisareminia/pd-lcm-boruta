UPC, DNC, ALLC = PANEL_C, CORE_C, "#3E454D"
C = Canvas(183.0, 195.0); M = C.M
def chip(x, y_, lib):
    M.add_patch(Rectangle((x, y_ - 1.35), 10.2, 2.7, facecolor="#E9ECEF", edgecolor="none", zorder=2))
    C.text(x + 5.1, y_, LIBTAG.get(lib, lib), ha="center", fontsize=4.6, color="#3E454D", zorder=3)

# ======================= a: over-representation with the genes behind each pathway =======================
C.letter(2.0, 191.0, "a", "Pathway over-representation of the lower-, higher- and all panel genes")
LISTS = [("up", UPC, f"Higher in PD ({len(UP)} genes)"), ("down", DNC, f"Lower in PD ({len(DOWN)} genes)"), ("all", ALLC, "All 30 genes")]
ROWS = []
for lst, col, lab in LISTS:
    ROWS.append(("header", lst, col, lab))
    for r in ORT[ORT.list == lst].nsmallest(5, "p").itertuples():
        ROWS.append(("term", lst, col, r))
GENES_A = []
for kind, lst, col, r in ROWS:
    if kind == "term":
        for g in r.genes.split(", "):
            if nice(g) not in GENES_A:
                GENES_A.append(nice(g))
GENES_A = [g for g in DOWN + UP if g in GENES_A]
TX, AX0, AX1, FDX, FWX, GX0, GX1 = 15.0, 78.0, 104.0, 110.0, 119.5, 126.0, 181.0
gcw = (GX1 - GX0) / len(GENES_A)
GXP = {g: GX0 + (i + 0.5) * gcw for i, g in enumerate(GENES_A)}
HY = 175.5
for x, t, ha in ((3.0, "Library", "left"), (TX, "Pathway", "left"), ((AX0 + AX1) / 2, "$-\\log_{10}$ P", "center"), (FDX, "FDR", "center"),
                 (FWX, "Family-\nwise P", "center")):
    C.text(x, HY, t, ha=ha, fontsize=5.7, linespacing=1.0)
for g in GENES_A:
    C.text(GXP[g], HY - 1.6, g, ha="center", va="bottom", rotation=90, fontsize=5.4, fontstyle="italic",
           color=DNC if DIRN[g] == "down" else UPC)
C.text((GX0 + GX1) / 2, 187.6, "Panel genes in the pathway", ha="center", fontsize=5.7)
C.rule(2.0, 181.0, HY - 2.6 - 0.0, lw=0.5, color=INK)
y_ = HY - 5.4
YPOS, GROUP_SPAN = [], {}
for kind, lst, col, r in ROWS:
    if kind == "header":
        C.text(TX, y_, r, ha="left", fontsize=5.8, color=col, fontweight="bold")
        GROUP_SPAN[lst] = [y_ - 2.6, None]; y_ -= 4.4
    else:
        YPOS.append((y_, lst, col, r)); GROUP_SPAN[lst][1] = y_ - 2.6; y_ -= 5.3
axA = C.ax(AX0, y_ + 2.0, AX1 - AX0, HY - 3.2 - (y_ + 2.0)); axA.set_ylim(y_ + 2.0, HY - 3.2); axA.set_xlim(0, 4.6)
axA.spines["left"].set_visible(False); axA.set_yticks([]); axA.patch.set_alpha(0)
axA.set_xticks([0, 1, 2, 3, 4]); axA.tick_params(axis="x", labelsize=5.6)
for lst, (ya, yb) in GROUP_SPAN.items():
    t = -np.log10(SMY["ora_calibration"][lst]["p_fwer05_random"])
    axA.plot([t, t], [yb + 0.6, ya], color="#8C9299", lw=0.7, ls=(0, (2.2, 1.6)), zorder=1)
for i, (yy, lst, col, r) in enumerate(YPOS):
    if i % 2 == 0:
        M.add_patch(Rectangle((2.0, yy - 2.6), 179.0, 5.2, facecolor="#F3F5F7", lw=0, zorder=0))
    chip(3.0, yy, r.library)
    lines = wrap(term_name(r.term), 58)
    C.text(TX, yy, "\n".join(lines), ha="left", fontsize=5.3, linespacing=1.02)
    x = -np.log10(r.p)
    axA.plot([0, x], [yy, yy], color="#C9CED4", lw=0.6, zorder=2)
    axA.scatter([x], [yy], s=8 + 8 * r.hits, color=col, zorder=3, linewidth=0)
    C.text(FDX, yy, f"{r.q_BH:.2f}", ha="center", fontsize=5.4)
    C.text(FWX, yy, f"{r.fwer_random:.3f}" if r.fwer_random < 0.1 else f"{r.fwer_random:.2f}", ha="center", fontsize=5.4)
    members = {nice(g) for g in r.genes.split(", ")}
    for g in GENES_A:
        if g in members:
            M.scatter([GXP[g]], [yy], s=15, color=DNC if DIRN[g] == "down" else UPC, lw=0, zorder=3)
        else:
            M.scatter([GXP[g]], [yy], s=1.6, color="#C9CED4", lw=0, zorder=2)
C.text(3.0, y_ - 3.2, f"Dashed lines: the P value that random gene sets of the same size reach, anywhere among the "
       f"{SMY['ora_calibration']['all']['terms']:,} pathways tested, in 5% of draws (family-wise 5% threshold). Dot area: number of panel genes.",
       ha="left", fontsize=5.3, color=MUTED)
BOT_A = y_ - 5.6
C.rule(2.0, 181.0, BOT_A, lw=0.35, color="#8C9299")

# ======================= b: do the pathway partners move with the panel gene? =======================
C.letter(2.0, BOT_A - 4.6, "b", "Pathway partners of the panel genes (label-shuffle test)")
NB = NBT.head(8)
TX2, AGX, BX0, BX1, PX, QX, NX = 15.0, 70.0, 86.0, 110.0, 116.0, 125.0, 131.0
HY2 = BOT_A - 10.6
for x, t, ha in ((3.0, "Library", "left"), (TX2, "Pathway", "left"), (AGX, "Panel gene", "left"), ((BX0 + BX1) / 2, "Partner shift (g)", "center"),
                 (PX, "P", "center"), (QX, "FDR", "center"), (NX, "Partners moving most with it", "left")):
    C.text(x, HY2, t, ha=ha, fontsize=5.7)
C.rule(2.0, 181.0, HY2 - 2.2, lw=0.5, color=INK)
y2 = HY2 - 5.4
YB = []
for i, r in enumerate(NB.itertuples()):
    if i % 2 == 0:
        M.add_patch(Rectangle((2.0, y2 - 2.6), 179.0, 5.2, facecolor="#F3F5F7", lw=0, zorder=0))
    col = UPC if r.direction == "up" else DNC
    chip(3.0, y2, r.library)
    C.text(TX2, y2, "\n".join(wrap(term_name(r.term), 50)), ha="left", fontsize=5.3, linespacing=1.02)
    C.text(AGX, y2, ", ".join(nice(a) for a in r.anchors.split(", ")), ha="left", fontsize=5.4, fontstyle="italic", color=col)
    C.text(PX, y2, f"{r.p:.3f}", ha="center", fontsize=5.4)
    C.text(QX, y2, f"{r.fdr:.2f}", ha="center", fontsize=5.4)
    C.text(NX, y2, ", ".join(r.top_neighbours.split(", ")[:6]), ha="left", fontsize=5.1, fontstyle="italic", color=MUTED)
    YB.append((y2, r, col)); y2 -= 5.3
axB = C.ax(BX0, y2 + 2.1, BX1 - BX0, HY2 - 2.6 - (y2 + 2.1)); axB.set_ylim(y2 + 2.1, HY2 - 2.6); axB.set_xlim(0, 0.3)
axB.spines["left"].set_visible(False); axB.set_yticks([]); axB.patch.set_alpha(0)
axB.set_xticks([0, 0.1, 0.2, 0.3]); axB.set_xticklabels(["0", "0.1", "0.2", "0.3"]); axB.tick_params(axis="x", labelsize=5.6)
for yy, r, col in YB:
    axB.plot([0, r.shift], [yy, yy], color="#C9CED4", lw=0.6, zorder=2)
    axB.scatter([r.shift], [yy], s=5 + 0.9 * r.neighbours, color=col, zorder=3, linewidth=0)
C.text(3.0, y2 - 4.6, f"The eight strongest of {len(NBT)} pathways holding a panel gene. Partner shift: mean PD effect of the pathway's other genes, "
       "in the panel gene's direction;", ha="left", fontsize=5.3, color=MUTED)
C.text(3.0, y2 - 7.2, "P from 5,000 label shuffles within study; none passes FDR or family-wise correction. Dot area: number of partner genes.",
       ha="left", fontsize=5.3, color=MUTED)
C.save("Figure09_panel_pathways")
