co = SUM["composition"]; lr = SUM["logistic"]
auc = pd.DataFrame(SUM["oof_auc"]); cv = pd.DataFrame(SUM["cv"])
core = auc[auc.model == "core"].set_index("variant")
o, a = core.loc["original"], core.loc["composition (50 markers)"]
ge = pd.read_csv(FIG_IN / "composition_gene_effects.csv")
pf = lambda p: f"P = {p:.4f}" if p >= 0.0001 else "P < 0.0001"
TEXT = f"""FIGURE 9 LEGEND - Pathway over-representation with g:Profiler
(a) g:GOSt run with the statistical domain set to the {5622:,} genes measured in the discovery data and Benjamini-Hochberg FDR as
the significance threshold, for the genes higher in PD (18), lower in PD (12) and all 30; the six smallest FDR values per list,
with the panel genes behind each pathway. Dots are filled at FDR < 0.05; dot area is the number of panel genes. "P vs random
lists" is how often 100 random gene lists of the same size reached a smaller FDR anywhere under identical settings. (b) The
number of "significant" pathways those random lists returned, with the panel's count marked. Terms holding only 2-4 background
genes reach very small FDR values when both of their genes are in the panel and name the genes rather than a pathway.

FIGURE 10 LEGEND - Does the classifier see more than the neuron mix?
(a) Composition score of every donor: mean z-score of the {len(SUM['markers']['calb1'])} strongest CALB1-lineage markers minus the {len(SUM['markers']['sox6'])} strongest SOX6-lineage
markers of Kamath et al. (2022), computed within each study with the 30 panel genes excluded. PD donors are more CALB1-like
(AUC {co['auc_composition_pd_more_CALB1_like']:.2f}, Mann-Whitney {pf(co['composition_mwu_p'])}), as expected when SOX6/AGTR1 neurons are lost. (b) The classifier's out-of-fold
score against the same composition (Spearman {co['rho_oof_composition']:.2f}; {co['rho_oof_composition_controls']:.2f} within controls alone). (c) Out-of-fold AUC of the core classifier on
the same folds after the composition is regressed out of every gene, with 95% bootstrap CI and P from label shuffles within
study; the marker count and the alternatives test how the estimate depends on that choice. (d) PD effect of each panel gene
before (open) and after (filled) the composition is removed, with the share of the effect kept and the gene's lineage score.

METHODS - Neuron-mix (composition) analysis
Kamath et al. (2022) report markers of ten human dopamine-neuron subtypes; the SOX6 subtypes, in particular SOX6_AGTR1, are the
ones lost in PD, and CALB1 subtypes are relatively spared. Each gene was given a lineage score (mean marker z across the six
CALB1 subtypes minus the mean across the four SOX6 subtypes). A donor's composition score is the mean expression z-score of the
50 strongest CALB1-lineage genes minus that of the 50 strongest SOX6-lineage genes, computed within each study; the 30 panel
genes were excluded, and no diagnosis labels are used. The composition was then regressed out of every gene within each study,
and the core classifier (within-person ranks, 30 principal components, Random Forest) was re-run on exactly the folds of the
main analysis. Significance of the adjusted out-of-fold AUC comes from 10,000 label shuffles within study, its interval from
4,000 bootstrap resamples of donors. Sensitivity analyses used 25, 100 and 200 markers, the SOX6_AGTR1 markers alone, and the
composition together with a dopamine-neuron purity score. Per gene, the pooled within-study PD effect (Hedges' g) was compared
before and after adjustment.

RESULTS - numbers
Composition differs by diagnosis: PD donors are more CALB1-like, AUC {co['auc_composition_pd_more_CALB1_like']:.2f} ({pf(co['composition_mwu_p'])}); SOX6/AGTR1 markers alone {co['auc_AGTR1_markers_pd_lower']:.2f}.
The classifier partly follows it: Spearman {co['rho_oof_composition']:.2f} overall, {co['rho_oof_composition_controls']:.2f} within controls.
With the composition regressed out of every gene, the core classifier keeps a significant part of its accuracy:
  out-of-fold AUC {o.oof_auc:.2f} [{o.ci_lo:.2f}-{o.ci_hi:.2f}] before, {a.oof_auc:.2f} [{a.ci_lo:.2f}-{a.ci_hi:.2f}] after ({pf(a.shuffle_p)}); cross-validated AUC {cv[(cv.variant == 'original') & (cv.model == 'core classifier')].cv_auc.iloc[0]:.3f} -> {cv[(cv.variant == 'composition (50 markers)') & (cv.model == 'core classifier')].cv_auc.iloc[0]:.3f}.
  Sensitivity (out-of-fold AUC, label-shuffle P): """ + "; ".join(f"{v} {core.loc[v].oof_auc:.2f} ({pf(core.loc[v].shuffle_p)})"
        for v in ["composition (25 markers)", "composition (100 markers)", "composition (200 markers)", "SOX6_AGTR1 markers only",
                  "composition + dopamine purity"]) + f"""
  The classifier's score adds to the composition: leave-one-out AUC {lr['loo_auc_composition']:.2f} -> {lr['loo_auc_composition_plus_score']:.2f}""" + (
    f", likelihood-ratio {pf(lr['lr_p'])}, score coefficient {pf(lr['score_coef_p'])}." if "lr_p" in lr else ".") + f"""
  Genes: {int((ge.retained >= 0.5).sum())}/30 keep at least half of their PD effect (median {100 * ge.retained.median():.0f}%). Most composition-dependent: """ + ", ".join(
    f"{r.symbol} {100 * r.retained:.0f}%" for r in ge.nsmallest(5, "retained").itertuples()) + "; least: " + ", ".join(
    f"{r.symbol} {100 * r.retained:.0f}%" for r in ge.nlargest(5, "retained").itertuples()) + """.
CONCLUSION: the panel reflects both which dopamine neurons survive and changes inside the surviving neurons - the classifier
stays above chance after the neuron mix is removed, and most panel genes keep most of their PD effect.
"""
print(TEXT)
open(OUT / "pathway_composition_legends.txt", "w").write(TEXT)
