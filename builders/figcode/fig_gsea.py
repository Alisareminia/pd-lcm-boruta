UPC, DNC = PD_C, CT_C
SRC_C = {"Hallmark": "#7A2533", "Reactome": "#23324A", "KEGG": "#5F6B3A", "GO BP": "#6B5B7A"}
C = Canvas(183.0, 152.0); M = C.M
def chip(x, y_, src, w=11.0):
    M.add_patch(Rectangle((x, y_ - 1.35), w, 2.7, facecolor=SRC_C.get(src, "#E9ECEF"), edgecolor="none", zorder=2))
    C.text(x + w / 2, y_, src, ha="center", fontsize=4.6, color="white", zorder=3)

# ======================= a: the significant gene sets =======================
C.letter(2.0, 148.0, "a", f"Gene sets enriched in the PD ranking of all {G:,} measured genes")
CHX, NX, SZX, AX0, AXW, QX, QSX, LEX = 3.0, 16.0, 80.0, 86.0, 30.0, 120.0, 130.0, 138.0
HY = 140.0
for x, t, ha in ((CHX, "Source", "left"), (NX, "Gene set", "left"), (SZX, "Genes", "center"),
                 (AX0 + AXW / 2, "Normalised enrichment score", "center"), (QX, "FDR", "center"),
                 (QSX, "Strict\nFDR", "center"), (LEX, "Panel genes in the leading edge", "left")):
    C.text(x, HY, t, ha=ha, fontsize=5.7, linespacing=1.0)
C.rule(2.0, 181.0, HY - 2.8, lw=0.5, color=INK)
y_, ROWS = HY - 6.0, []
for r in TOP.itertuples():
    ROWS.append((y_, r)); y_ -= 4.5
axA = C.ax(AX0, y_ + 2.6, AXW, HY - 3.6 - (y_ + 2.6)); axA.set_ylim(y_ + 2.6, HY - 3.6)
axA.set_xlim(0, max(2.4, TOP.NES_gene_shuffle.max() * 1.08))
axA.spines["left"].set_visible(False); axA.set_yticks([]); axA.patch.set_alpha(0)
axA.set_xticks([0, 1, 2]); axA.tick_params(axis="x", labelsize=5.6)
for i, (yy, r) in enumerate(ROWS):
    if i % 2 == 0:
        M.add_patch(Rectangle((2.0, yy - 2.25), 179.0, 4.5, facecolor="#F3F5F7", lw=0, zorder=0))
    chip(CHX, yy, r.source)
    C.text(NX, yy, "\n".join(wrap(r.set, 56)), ha="left", fontsize=5.3, linespacing=1.02)
    C.text(SZX, yy, str(int(r.size)), ha="center", fontsize=5.4)
    axA.plot([0, r.NES_gene_shuffle], [yy, yy], color="#C9CED4", lw=0.6, zorder=2)
    axA.scatter([r.NES_gene_shuffle], [yy], s=9 + 0.09 * r.size, color=SRC_C.get(r.source, INK), zorder=3, linewidth=0)
    C.text(QX, yy, f"{r.q_gsea_gene_shuffle:.3f}", ha="center", fontsize=5.4,
           fontweight="bold" if r.q_gsea_gene_shuffle < 0.05 else "normal")
    C.text(QSX, yy, f"{r.q_gsea_label_shuffle:.2f}", ha="center", fontsize=5.4, color=MUTED)
    C.text(LEX, yy, str(r.panel_in_leading_edge) if isinstance(r.panel_in_leading_edge, str) else "-", ha="left", fontsize=5.3,
           fontstyle="italic", color=UPC if isinstance(r.panel_in_leading_edge, str) else MUTED)
BOT = y_ - 5.2
C.text(3.0, BOT, f"All {len(TOP)} gene sets below FDR {Q_THR} are shown; every one is higher in PD. FDR: the GSEA procedure with its "
       "usual gene-permutation null.", ha="left", fontsize=5.3, color=MUTED)
C.text(3.0, BOT - 2.8, "Strict FDR: the same procedure with the donors' diagnoses shuffled within study; no set reaches 0.25 "
       f"that way, so these are exploratory. {int(SUM['gsea_significant_with_panel_gene'])} carry a panel gene in the leading edge.",
       ha="left", fontsize=5.3, color=MUTED)
C.rule(2.0, 181.0, BOT - 6.6, lw=0.35, color="#8C9299")

# ======================= b: the running score of the strongest sets =======================
C.letter(2.0, BOT - 11.0, "b", "Where those genes sit in the ranking")
BY, BH0, BW = 3.0, 16.0, 50.0
shown = [r for r in TOP.head(3).itertuples()]
for i, r in enumerate(shown):
    x0 = 16.0 + i * (BW + 6.0)
    es, pos = running(r.set)
    ax = C.ax(x0, BY + 5.0, BW, BH0)
    col = SRC_C.get(r.source, INK)
    if es is not None:
        ax.plot(np.arange(G), es, color=col, lw=0.9)
        j = int(np.argmax(np.abs(es)))
        ax.scatter([j], [es[j]], s=10, color=col, zorder=4, linewidth=0)
        ax.axhline(0, color="#C9CED4", lw=0.5)
        axt = C.ax(x0, BY + 1.6, BW, 3.0); axt.set_xlim(0, G); axt.axis("off")
        for p_ in pos:
            axt.plot([p_, p_], [0, 1], color=col, lw=0.35, alpha=0.8)
    ax.set_xlim(0, G); ax.set_xticks([0, G // 2, G])
    ax.set_xticklabels(["most up\nin PD", "0", "most down\nin PD"], fontsize=5.0, linespacing=1.0)
    ax.tick_params(axis="x", length=2.0, pad=1.4); ax.tick_params(axis="y", labelsize=5.4)
    if i == 0:
        ax.set_ylabel("Running enrichment", fontsize=5.7, labelpad=2)
    C.text(x0 + BW / 2, BY + BH0 + 6.2, "\n".join(wrap(r.set, 40)), ha="center", va="bottom", fontsize=5.6, color=col,
           fontweight="bold", linespacing=1.05)
    C.text(x0 + BW / 2, BY - 1.4, f"NES {r.NES_gene_shuffle:.2f}, FDR {r.q_gsea_gene_shuffle:.3f}"
           + (f"; {r.panel_in_leading_edge}" if isinstance(r.panel_in_leading_edge, str) else ""), ha="center", fontsize=5.3, color=MUTED)
C.save("Figure11_gsea_pathways")

# ======================= a second, shorter figure: the same pathways in every donor =======================
C = Canvas(183.0, 78.0); M = C.M

# ======================= c: the same pathways scored in every donor =======================
C.text(2.0, 74.0, "Pathway scores of every donor (single-sample GSEA)", ha="left", va="baseline", fontsize=7.2)
SEL = (SSR[SSR.set.isin(TOP.set)].nsmallest(12, "p").set.tolist())
SS = SSC.copy()
ORDER = SS.sort_values(["dataset", "y"]).index.to_numpy()
HX0, HX1 = 62.0, 168.0
cw = (HX1 - HX0) / len(ORDER)
rh, HY0 = 4.0, 62.0
aucs = SSR.set_index("set").auc.to_dict(); pss = SSR.set_index("set").p.to_dict()
for k, name in enumerate(SEL):
    yy = HY0 - k * rh
    v = SS[name].to_numpy()[ORDER]
    v = np.clip(v / (np.abs(v).max() or 1), -1, 1)
    for i, val in enumerate(v):
        M.add_patch(Rectangle((HX0 + i * cw, yy - rh / 2 + 0.25), cw, rh - 0.5,
                              facecolor=EFFECT(0.5 + 0.5 * val), edgecolor="none"))
    C.text(HX0 - 1.5, yy, "\n".join(wrap(name, 44)[:2]), ha="right", fontsize=5.0, linespacing=1.0)
    C.text(HX1 + 2.0, yy, f"{aucs.get(name, float('nan')):.2f}", ha="left", fontsize=5.2)
    C.text(HX1 + 10.0, yy, f"{pss.get(name, float('nan')):.3f}", ha="left", fontsize=5.2, color=MUTED)
C.text(HX1 + 2.0, HY0 + 3.2, "AUC", ha="left", fontsize=5.4)
C.text(HX1 + 10.0, HY0 + 3.2, "P", ha="left", fontsize=5.4, color=MUTED)
ay = HY0 + 2.2
for i, idx in enumerate(ORDER):
    M.add_patch(Rectangle((HX0 + i * cw, ay), cw, 1.8, facecolor=PD_C if SS.y[idx] else CT_C, edgecolor="none"))
studies = sorted(SS.dataset.unique())
sh = {d: f"#{h}" for d, h in zip(studies, ["D9DCE0", "BFC5CB", "A5ADB5", "8B949E"])}
for i, idx in enumerate(ORDER):
    M.add_patch(Rectangle((HX0 + i * cw, ay + 2.0), cw, 1.8, facecolor=sh[SS.dataset[idx]], edgecolor="none"))
C.text(HX0 - 1.5, ay + 0.9, "diagnosis", ha="right", fontsize=5.0, color=MUTED)
C.text(HX0 - 1.5, ay + 2.9, "study", ha="right", fontsize=5.0, color=MUTED)
kx = HX0
for d in studies:
    M.add_patch(Rectangle((kx, HY0 - 12 * rh - 4.0), 2.6, 1.8, facecolor=sh[d], edgecolor="none"))
    C.text(kx + 3.2, HY0 - 12 * rh - 3.1, d, ha="left", fontsize=5.0, color=MUTED); kx += 22.0
for x, col, t in ((HX0, CT_C, "control"), (HX0 + 16.0, PD_C, "PD")):
    M.add_patch(Rectangle((x, HY0 - 12 * rh - 8.0), 2.6, 1.8, facecolor=col, edgecolor="none"))
    C.text(x + 3.2, HY0 - 12 * rh - 7.1, t, ha="left", fontsize=5.0, color=MUTED)
cax = C.ax(HX0 + 60.0, HY0 - 12 * rh - 7.6, 22.0, 1.6)
cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=EFFECT); cax.set_xticks([]); cax.set_yticks([])
for s_ in cax.spines.values():
    s_.set_linewidth(0.3); s_.set_color("#B7BDC4")
C.text(HX0 + 59.0, HY0 - 12 * rh - 6.8, "low", ha="right", fontsize=5.0, color=MUTED)
C.text(HX0 + 83.0, HY0 - 12 * rh - 6.8, "high   pathway score within study", ha="left", fontsize=5.0, color=MUTED)
C.text(2.0, HY0 - 12 * rh - 12.0, f"The {len(SEL)} of the {len(TOP)} gene sets that separate PD from control best when scored donor by "
       f"donor; {SUM['ssgsea_p_lt_0.05']} of {SUM['sets_tested']:,} pathways reach P < 0.05 this way, none survives BH correction.",
       ha="left", fontsize=5.3, color=MUTED)
C.save("Figure12_gsea_per_donor")
