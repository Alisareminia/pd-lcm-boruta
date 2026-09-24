cF, cP, cN = A["frozen"], A["panel"], A["neuron"]
ga, gr = SUM["gene_agreement"]["g_external_neuron_adj"], SUM["gene_agreement"]["g_external"]
co5, cob = POOL["calls@0.5"], POOL["calls@oob"]
nshared = int(OVL.shared_with_discovery.sum())
sw = UNI["sensitivity_without_GSE7621"]
ci = lambda d: f"{d['auc']:.2f} (95% CI {d['ci_lo']:.2f}-{d['ci_hi']:.2f})"
TEXT = f"""FIGURE 6 LEGEND - External validation in eight independent bulk substantia nigra cohorts
Both models were frozen before any external cohort was analysed and were applied without refitting; each cohort's genes were
z-scored within the cohort and ranked within each person, as in discovery. {len(COHORTS)} cohorts, {POOL['people']} people ({POOL['control']} control, {POOL['PD']} PD),
after removal of {nshared} donors shared with the discovery data. (a) ROC of the core classifier, the Boruta-panel forest and an eight-gene
dopamine-neuron marker score, each cohort's curve averaged with weights equal to its number of PD-control pairs; pooled AUCs
{cF['auc']:.2f}, {cP['auc']:.2f} and {cN['auc']:.2f} (label shuffles within cohorts: P {pfmt(PERM['core'])} and P {pfmt(PERM['panel'])} for the two models). (b) Core
classifier score (centred within each cohort) against dopamine-neuron content (Spearman {RHO['core']:.2f}). Table: pooled AUCs with the
neuron-content component regressed out of each score within each cohort, and leave-one-out logistic models within each cohort
using neuron content alone or with a model score. (c) PD-versus-control effect (Hedges' g) of each Boruta gene in the laser-capture
discovery cohort, in each bulk cohort after neuron content is regressed out, and pooled across the bulk cohorts by inverse variance
without and with this adjustment; struck-through cells have the opposite sign to discovery, grey cells were not measured.

FIGURE 7 LEGEND - The frozen models cohort by cohort, calls at fixed cut-offs, and donor overlap
(a) AUC in each cohort (filled, model as reported, with 95% bootstrap CI; open, strictly independent version of the same recipe
retrained without the discovery studies from the same source; grey tick, neuron markers alone) and pooled over the {len(COHORTS)} cohorts
(cohort AUCs weighted by PD-control pairs, each donor counted once; diamond width, 95% CI). (b) Accuracy, sensitivity and
specificity at two cut-offs fixed on the 63 discovery people before testing: 0.5, and each forest's out-of-bag accuracy optimum;
below, balanced accuracy at the out-of-bag cut-off in each cohort. (c) Donor overlap between the discovery studies and the bulk
cohorts. Donor IDs were compared where they were public in a comparable format; shared donors ({nshared}) were removed before testing,
and one donor present in two bulk cohorts was counted once when cohorts were pooled. Shaded groups may share a brain source, so
their cohorts were also scored by the strictly independent models.

METHODS - External validation
Model lock. The core classifier (within-person gene ranks, 30 principal components, Random Forest of 1,000 trees) was chosen by a
sweep over 97 Random Forest variants that used only the 63 discovery people; the Boruta panel and its forest were fitted on the
same people. Both were frozen, with their decision thresholds, before the external cohorts were analysed.
Cohorts. Eight public bulk substantia nigra cohorts were obtained from GEO and analysed together: GSE7621, GSE8397, GSE20163,
GSE20164 and GSE20292 (Affymetrix HG-U133A or Plus 2; for GSE8397, lateral and medial nigra averaged per case), GSE49036
(HG-U133 Plus 2; controls and PD, Braak 3-6), GSE114517 (RNA-seq counts, nigra only; PD with dementia) and GSE168496 (RNA-seq,
transcript abundances summed to genes). Probe sets were mapped to Ensembl genes with g:Profiler (highest-mean probe set per gene)
and array data were log2-transformed. Each cohort was processed without its labels: genes z-scored within the cohort, then ranked
within each person; unmeasured genes were set to the cohort mean.
Donor overlap. Donor IDs, where public in a comparable format, were matched to the discovery donors (GSE20292 and GSE20163
against GSE20141; GSE168496 against GSE182622); the {nshared} matched donors (all with the same diagnosis) were removed. Because not
every cohort reports IDs, each cohort was also scored by strictly independent models - the same recipe, including Boruta for the
panel, re-run without every discovery study from a potentially shared source (GSE20141 and GSE24378 for GSE20163, GSE20164 and
GSE20292; GSE182622 for GSE49036 and GSE168496).
Statistics. Per-cohort AUCs have 95% CIs from 4,000 bootstrap resamples of people. Pooled AUCs are cohort AUCs weighted by the
number of PD-control pairs (CIs from resampling people within cohorts; P from 10,000 label shuffles within cohorts); ROC curves
were averaged vertically with the same weights. Calls used thresholds fixed on the discovery people: 0.5 and each forest's
out-of-bag accuracy optimum (core {SUM['thresholds']['core']:.2f}, panel {SUM['thresholds']['panel']:.2f}). Neuron content was the mean z-score of eight
dopamine-neuron marker genes (TH, SLC6A3, SLC18A2, DDC, KCNJ6, ALDH1A1, NR4A2, EN1); neuron-adjusted AUCs used each score's residual
after linear regression on it within the cohort; leave-one-out logistic models were fitted within each cohort and pooled like the
AUCs. Gene effects were Hedges' g (PD minus control) with and without neuron content regressed out, pooled by inverse variance;
agreement with discovery was tested with a one-sided binomial test on the signs. GSE7621 had been examined in an earlier version of
this project; without it the pooled AUCs are {sw['frozen']['auc']:.2f} (core), {sw['panel']['auc']:.2f} (panel) and {sw['neuron']['auc']:.2f} (neuron markers).

RESULTS - numbers ({len(COHORTS)} cohorts, {POOL['people']} people)
  core classifier      AUC {ci(cF)}, label shuffles P {pfmt(PERM['core'])}; neuron-adjusted {A['frozen|neuron_removed']['auc']:.2f}; strictly independent {A['strict']['auc']:.2f}
  Boruta-panel forest  AUC {ci(cP)}, label shuffles P {pfmt(PERM['panel'])}; neuron-adjusted {ci(A['panel|neuron_removed'])}
                       strictly independent {ci(A['panel_strict'])}
  neuron markers alone AUC {ci(cN)}
  leave-one-out: neurons {LOO['neurons']:.2f}; neurons + core {LOO['neurons+core']:.2f}; neurons + panel {LOO['neurons+panel']:.2f}
  calls, cut-off 0.5:   core acc {co5['frozen']['accuracy']:.2f} sens {co5['frozen']['sensitivity']:.2f} spec {co5['frozen']['specificity']:.2f}; panel acc {co5['panel']['accuracy']:.2f} sens {co5['panel']['sensitivity']:.2f} spec {co5['panel']['specificity']:.2f}
  calls, OOB cut-off:   core acc {cob['frozen']['accuracy']:.2f} sens {cob['frozen']['sensitivity']:.2f} spec {cob['frozen']['specificity']:.2f}; panel acc {cob['panel']['accuracy']:.2f} sens {cob['panel']['sensitivity']:.2f} spec {cob['panel']['specificity']:.2f}
  genes: {ga['same_sign']}/{ga['n']} same sign after neuron adjustment (binomial P = {ga['binom_p']:.4f}, Spearman {ga['spearman']:.2f}); {gr['same_sign']}/{gr['n']} unadjusted
"""
print(TEXT)
open(OUT / "external_unified_legends_methods.txt", "w").write(TEXT)
