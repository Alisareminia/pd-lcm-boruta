U, A = POOL["unseen"], POOL["all"]
def ci(d): return f"{d['auc']:.2f} (95% CI {d['ci_lo']:.2f}-{d['ci_hi']:.2f})"
cu = U["calls@oob"]
OVI = OV.set_index("cohort"); r1, r2 = OVI.loc["GSE20292"], OVI.loc["GSE20163"]
ga, gr = AGREE["g_external_neuron_adj"], AGREE["g_external"]
TEXT = f"""FIGURE LEGEND
External validation in eight independent bulk substantia nigra cohorts. Both models were frozen before these cohorts
were opened (GSE7621, marked with a dagger, had been examined in earlier versions of the project) and were applied without
refitting; each cohort's genes were z-scored within the cohort and ranked within each person, exactly as in discovery.
(a) AUC of the core classifier and of the Boruta-panel forest in each cohort. Filled circle, model as reported, with 95%
bootstrap CI; open circle, strictly independent version of the same recipe, retrained without any discovery study that
could share donors with that cohort's brain bank; grey tick, the eight dopamine-neuron marker genes alone. Donors whose
brain-bank IDs matched a discovery donor were removed before testing (GSE20292, {r1.shared_with_discovery} of {r1.people};
GSE20163, {r2.shared_with_discovery} of {r2.people}; diagnosis agreed for every match). Pooled rows: cohort AUCs weighted by the number of
PD-control pairs, each donor counted once; diamond width, 95% CI. (b) Pooled over the {len(U['cohorts'])} cohorts never examined before the
model was locked ({U['people']} people): AUC; AUC after the neuron-marker score was regressed out of each model's score within each
cohort; and accuracy, sensitivity and specificity at the out-of-bag cut-off fixed on the 63 discovery people. (c) PD-versus-
control effect (Hedges' g) of each of the {ga['n']} panel genes in the laser-capture discovery cohort and, pooled by inverse variance, in the
eight bulk cohorts after adjustment for neuron content (filled) and without it (open). Shaded quadrants, same direction.

METHODS - External validation
Model lock. The core classifier (within-person gene ranks, 30 principal components, Random Forest of 1,000 trees) was chosen by
a sweep over 97 Random Forest variants that used only the 63 discovery people; the Boruta panel and its forest were fitted
on the same 63 people. Both were frozen, together with their decision thresholds, before any external cohort was analysed.
Cohorts. Eight public bulk substantia nigra cohorts were downloaded from GEO: GSE7621, GSE20292, GSE20163, GSE20164 and GSE8397
(Affymetrix HG-U133A or Plus 2; for GSE8397 the lateral and medial nigra of each case were averaged and frontal cortex excluded),
GSE49036 (HG-U133 Plus 2; controls and Braak 3-6 PD, with incidental Lewy body cases analysed separately), GSE114517 (RNA-seq
counts, nigra only; PD with dementia) and GSE168496 (RNA-seq; transcript abundances summed to genes). Probe sets were mapped to
Ensembl genes with g:Profiler (the highest-mean probe set per gene) and array data were log2-transformed. Each cohort was
processed without its labels: every gene was z-scored within the cohort and ranked within each person; genes not measured
were set to the cohort mean (coverage {COV.share.min():.0%}-{COV.share.max():.0%} of the {len(GENES):,} discovery genes).
Donor overlap. Where external donor IDs were public in a comparable format they were matched to the discovery donors
(GSE20292 and GSE20163 against GSE20141; GSE168496 against GSE182622). Matched donors ({int(OV.shared_with_discovery.sum())} in total) were removed
before testing; every match agreed in diagnosis. One GSE20163 donor also present in GSE20292 was counted once in pooled
analyses. Because not every cohort reports donor IDs, each cohort was also scored by strictly independent models: the same
recipe (for the panel, Boruta itself) re-run without every discovery study from a potentially shared brain bank (without
GSE20141 and GSE24378 for the Harvard-series cohorts, {STRICT['harvard']['n']} people; without GSE182622 for the Netherlands Brain Bank
cohorts, {STRICT['nbb']['n']} people).
Thresholds and statistics. Two thresholds were fixed on the discovery people before testing: 0.5, and each forest's out-of-bag
accuracy optimum (core {THRESHOLDS['core']:.2f}; panel {THRESHOLDS['panel']:.2f}). AUC 95% CIs came from 4,000 bootstrap resamples of people and
P values from 10,000 label permutations within each cohort. Pooled AUCs were cohort AUCs weighted by the number of PD-control
pairs, with CIs from resampling people within cohorts. Neuron content was the mean z-score of eight dopamine-neuron marker
genes (TH, SLC6A3, SLC18A2, DDC, KCNJ6, ALDH1A1, NR4A2, EN1); neuron-adjusted AUCs used the residual of each score after linear
regression on this estimate within the cohort. Gene-level effects were Hedges' g (PD minus control), pooled across cohorts
by inverse-variance weighting, with and without the neuron estimate regressed out of each gene.

RESULTS - numbers
Seven unseen cohorts ({U['people']} people, {U['control']} control, {U['PD']} PD):
  core classifier       AUC {ci(U['auc']['frozen'])}; neuron-adjusted {U['auc']['frozen|neuron_removed']['auc']:.2f}; strictly independent {U['auc']['strict']['auc']:.2f}
  Boruta-panel forest   AUC {ci(U['auc']['panel'])}; neuron-adjusted {ci(U['auc']['panel|neuron_removed'])}
                        strictly independent {ci(U['auc']['panel_strict'])}
  neuron markers alone  AUC {ci(U['auc']['neuron'])}
  at the discovery cut-off: core accuracy {cu['frozen']['accuracy']:.2f}, sensitivity {cu['frozen']['sensitivity']:.2f}, specificity {cu['frozen']['specificity']:.2f}
                            panel accuracy {cu['panel']['accuracy']:.2f}, sensitivity {cu['panel']['sensitivity']:.2f}, specificity {cu['panel']['specificity']:.2f}
All eight cohorts ({A['people']} people): core {ci(A['auc']['frozen'])}; panel {ci(A['auc']['panel'])}; neuron markers {ci(A['auc']['neuron'])}
Gene level: {ga['same_sign']}/{ga['n']} panel genes in the discovery direction after neuron adjustment (binomial P = {ga['binom_p']:.4f},
  Spearman {ga['spearman']:.2f}); {gr['same_sign']}/{gr['n']} without adjustment.
Early stage (GSE49036, {EARLY['ilbd']} incidental Lewy body vs {EARLY['control']} control): core {EARLY['frozen']['auc']:.2f}, core strict {EARLY['strict']['auc']:.2f},
  panel {EARLY['panel']['auc']:.2f}, panel strict {EARLY['panel_strict']['auc']:.2f}, neuron markers {EARLY['auc_neuron_markers']:.2f}
"""
print(TEXT)
open(OUT / "external_multi_legend_methods.txt", "w").write(TEXT)
