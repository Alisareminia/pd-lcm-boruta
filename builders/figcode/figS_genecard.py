from matplotlib.colors import LinearSegmentedColormap
UPC, DNC = PANEL_C, CORE_C
DIV = LinearSegmentedColormap.from_list("div", ["#1E3350", "#4C6583", "#9AAABB", "#EEECE7", "#C3A09E", "#8C4B52", "#551C28"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#F4F5F6", "#C9CED4", "#8C9299", "#4A4F56", "#23282D"])
G = GTB.copy()
G["name_full"] = G["name"].fillna("")
G["grp"] = np.where(G.direction == "down", 0, 1)
G = G.sort_values(["grp", "hedges_g_discovery"], key=lambda s: s if s.name == "grp" else s.abs(), ascending=[True, False]).reset_index(drop=True)
C = Canvas(183.0, 134.0); M = C.M
C.text(2.0, 130.0, "Panel gene summary", ha="left", va="baseline", fontsize=7.2)
COLS = [  # key, header, kind, scale
    ("hedges_g_discovery", "PD effect\n(discovery g)", "div", 1.3),
    ("p_discovery", "Discovery\n$-\\log_{10}$ P", "seqlog", 4.0),
    ("single_gene_auc", "Single-gene\nAUC", "seq", (0.5, 0.8)),
    ("shap_rank", "SHAP\nrank", "rank", 30),
    ("boruta_fold_frequency", "Boruta fold\nfrequency", "seq", (0, 1)),
    ("g_external_neuron_adjusted", "Bulk nigra g\n(neuron-adj.)", "div", 0.7),
    ("dopamine_lineage_score", "Lineage\nscore", "lin", 16),
    ("top_subtype_marker", "Strongest\nsubtype marker", "text", None),
    ("pd_association_open_targets", "Open Targets\nPD score", "seq", (0, 0.6)),
    ("pd_genetic_association", "PD genetic\nevidence", "seq", (0, 0.8))]
X_SYM, X_NAME, X0 = 3.0, 17.0, 68.0
widths = [10.5, 10.5, 10.5, 9.0, 10.5, 11.5, 10.5, 17.0, 11.5, 11.0]
xs = np.cumsum([X0] + widths[:-1]); xc = xs + np.array(widths) / 2
HY = 122.5
C.text(X_SYM, HY, "Gene", ha="left", fontsize=5.8); C.text(X_NAME, HY, "Name", ha="left", fontsize=5.8)
for (k, h, kind, sc_), x in zip(COLS, xc):
    C.text(x, HY, h, ha="center", fontsize=5.4, linespacing=1.0)
C.rule(2.0, 181.0, HY - 3.4, lw=0.5, color=INK)
RH = 3.3
y_ = HY - 6.0
tr = lambda v: np.sign(v) * np.log10(1 + np.abs(v) / 2.0)
for grp, lab, col in ((0, f"Lower in PD ({len(DOWN)})", DNC), (1, f"Higher in PD ({len(UP)})", UPC)):
    C.text(X_SYM, y_, lab, ha="left", fontsize=5.8, color=col, fontweight="bold"); y_ -= 3.8
    sub = G[G.grp == grp]
    y_top = y_ + RH / 2
    for r in sub.itertuples():
        C.text(X_SYM + 1.4, y_, r.symbol, ha="left", fontsize=5.6, fontstyle="italic", color=col)
        C.text(X_NAME, y_, r.name_full[0].upper() + r.name_full[1:] if r.name_full else "", ha="left", fontsize=4.9, color="#3E454D")
        for (k, h, kind, sc_), x0, w in zip(COLS, xs, widths):
            v = getattr(r, k)
            face, txt, tcol = "#FFFFFF", "", INK
            if kind == "div":
                face = DIV(0.5 + 0.5 * np.clip(v / sc_, -1, 1)) if pd.notna(v) else "#FFFFFF"
                txt = f"{v:+.2f}" if pd.notna(v) else "n/a"
                tcol = "white" if pd.notna(v) and abs(v / sc_) > 0.55 else INK
                if k == "g_external_neuron_adjusted" and pd.notna(v) and np.sign(v) == np.sign(r.hedges_g_discovery):
                    txt += "*"
            elif kind == "seqlog":
                lv = -np.log10(v); face = SEQ(min(lv / sc_, 1)); txt = f"{lv:.1f}"; tcol = "white" if lv / sc_ > 0.6 else INK
            elif kind == "seq":
                a_, b_ = sc_; f = np.clip((v - a_) / (b_ - a_), 0, 1) if pd.notna(v) else 0
                face = SEQ(f); txt = (f"{v:.2f}" if pd.notna(v) else ""); tcol = "white" if f > 0.6 else INK
            elif kind == "rank":
                f = 1 - (v - 1) / (sc_ - 1); face = SEQ(f); txt = f"{int(v)}"; tcol = "white" if f > 0.6 else INK
            elif kind == "lin":
                face = DIV(0.5 + 0.5 * np.clip(tr(v) / tr(sc_), -1, 1)); txt = f"{v:+.1f}" if abs(v) >= 0.05 else "0"
                tcol = "white" if abs(tr(v) / tr(sc_)) > 0.55 else INK
            elif kind == "text":
                txt = str(v).replace("_", " ") if isinstance(v, str) and v else "-"
                tcol = DNC if txt.startswith("SOX6") else (UPC if txt.startswith("CALB1") else MUTED)
            M.add_patch(Rectangle((x0 + 0.25, y_ - RH / 2 + 0.2), w - 0.5, RH - 0.4, facecolor=face, edgecolor="none"))
            C.text(x0 + w / 2, y_, txt, ha="center", fontsize=4.9, color=tcol)
        y_ -= RH
    M.plot([2.0, 2.0], [y_ + RH / 2, y_top], color=col, lw=1.4, solid_capstyle="butt")
    y_ -= 1.2
C.rule(2.0, 181.0, y_ + 0.4, lw=0.5, color=INK)
C.text(2.0, y_ - 2.4, "Bulk nigra g: pooled over the 8 external cohorts after neuron content was regressed out; * same sign as discovery. "
       "Lineage score: CALB1 minus SOX6 subtype markers (Kamath et al. 2022).", ha="left", fontsize=5.2, color=MUTED)
C.text(2.0, y_ - 5.0, "Open Targets: association with Parkinson disease (overall and genetic evidence). SHAP rank 1 = most important "
       "gene in the panel forest; Boruta fold frequency = share of cross-validation folds selecting the gene.", ha="left", fontsize=5.2, color=MUTED)
C.save("FigureS_panel_gene_summary")
