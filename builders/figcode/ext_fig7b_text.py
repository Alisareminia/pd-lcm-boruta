nshared = int(OVL.shared_with_discovery.sum())
TEXT = f"""FIGURE 7 LEGEND - External validation, cohort by cohort
(a) Discrimination of the frozen core classifier (navy) and Boruta-panel forest (burgundy) in each of {len(COHORTS)} bulk substantia nigra
cohorts. Squares, model as reported (trained on all 63 discovery donors), with 95% bootstrap CI; square area proportional to the
cohort's weight (PD-control pairs). Open circles, strictly independent version of the same recipe, retrained without the discovery
studies that may share a brain source with that cohort. Grey ticks, eight-gene dopamine-neuron marker score alone. Diamonds,
pooled estimate over the {len(COHORTS)} cohorts ({POOL['people']} donors; cohort AUCs weighted by PD-control pairs, each donor counted once), width 95% CI.
Shared donors: donors matched to a discovery donor by ID and removed before testing. (b) Accuracy, sensitivity and specificity
pooled over the cohorts at two thresholds fixed on the discovery donors before testing (0.5, and each forest's out-of-bag
accuracy optimum; strictly independent models use their own out-of-bag optimum); below, balanced accuracy of each model at the
out-of-bag threshold in each cohort (squares, models as reported; open circles, strictly independent). (c) Donor overlap between the discovery studies (left) and the external cohorts
(right), grouped by possible common brain source (shaded groups). Solid lines, donors with matching IDs (all with the same
diagnosis; {nshared} in total, removed before testing); dashed line, IDs compared and none shared; bracket, one donor present in two
external cohorts, counted once in pooled estimates. Numbers in boxes, donors deposited -> analysed. Cohorts in shaded groups were
also scored by the strictly independent models."""
print(TEXT)
open(OUT / "figure07_v2_legend.txt", "w").write(TEXT)
