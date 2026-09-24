def pf(p): return f"P = {p:.2g}" if p >= 0.001 else f"P = {p:.1e}"
sig_sub = SUBTAB[SUBTAB.q_BH < 0.05].sort_values("p")
best = {k: ORA[ORA.list == k].nsmallest(1, "p").iloc[0] for k in LISTS}
nnb = CT.iloc[0]
TEXT = f"""FIGURE LEGEND - Biology of the Boruta panel
(a) Each panel gene on a dopamine-neuron lineage axis built from Kamath et al. (2022, Supplementary Table 8): the gene's mean
marker z across the six CALB1 subtypes minus its mean across the four SOX6 subtypes (0 when the gene marks no subtype).
Upper tracks, the {len(DOWN)} genes lower and the {len(UP)} genes higher in PD dopamine neurons; below, all {N:,} measured genes (square-root
counts, symmetric-log axis). Shaded, the outer 5% of all genes. Down genes sit on the vulnerable SOX6 side and up genes on the
resilient CALB1 side (Mann-Whitney {pf(SUB['up_vs_down'])}; against gene sets matched on the strength and direction of their PD
change, {pf(SUB['p_vs_DE_matched'])}). (b) Overlap of the down and up genes with the 200 strongest markers of each subtype
(hypergeometric test against the {N:,} measured genes, Benjamini-Hochberg across the 20 tests); filled circles, FDR < 0.05. SOX6_AGTR1
is the population that degenerates in PD. (c) Over-representation of the up, down and all 30 genes among Gene Ontology,
Reactome, KEGG and WikiPathways terms of 10-500 measured genes; the four smallest P values per list, with the genes behind each.
Dashed lines, the family-wise 5% threshold: the P value that random gene sets of the same size reach anywhere among the
{CAL['all']['terms']:,} terms in 5% of draws. No term crosses it, and none reaches FDR < 0.05.

METHODS - Enrichment and pathway analysis
The 30 Boruta genes were split by the sign of their pooled PD effect in the discovery donors ({len(UP)} higher, {len(DOWN)} lower in PD)
and analysed separately, with all 30 as a secondary analysis. The background was the {N:,} genes measured in the discovery data.
Over-representation: gene sets from Enrichr (GO Biological Process, Cellular Component and Molecular Function 2023, Reactome
2022, KEGG 2021, WikiPathways 2023), restricted to measured genes, kept at 10-500 genes, identical sets counted once
({CAL['all']['terms']:,} terms); one-sided hypergeometric tests; Benjamini-Hochberg over every term, not only the terms a list touches. Each
result was calibrated against 2,000 random gene sets of the same size (family-wise threshold) and 2,000 sets matched gene by
gene on the magnitude and direction of the PD effect; with this procedure random sets returned any FDR < 0.05 term in
{100 * max(v['random_sets_with_any_BH_hit'] for v in CAL.values()):.1f}% of draws or fewer.
Dopamine-neuron subtypes: the subtype markers of Kamath et al. (2022; MAST, each of ten subtypes against the other dopamine
neurons, FDR < 0.05) gave each gene a lineage score (mean z over CALB1 subtypes minus mean z over SOX6 subtypes). The scores of
the down and up genes were compared with each other (Mann-Whitney) and against 2,000 gene sets matched on PD effect; the
transcriptome-wide association between each gene's PD effect and its lineage score was tested with 2,000 label shuffles within
study. The 200 strongest markers of each subtype were tested against the down and up lists (hypergeometric, BH over 20 tests).
Pathway neighbours: for every term of 10-150 measured genes holding a panel gene ({len(CT)} terms), the mean PD effect of its other
members, signed by the panel genes' direction, was compared with 5,000 within-study label shuffles; whole-search FWER from the
minimum P per shuffle and empirical FDR. Gene annotation: mygene.info (names, RefSeq summaries) and the Open Targets Platform
(association with Parkinson disease, MONDO_0005180).

RESULTS - numbers
Over-representation: best terms - up: {best['up'].term} (P = {best['up'].p:.1e}, FDR {best['up'].q_BH:.2f}); down: {best['down'].term}
  (P = {best['down'].p:.1e}, FDR {best['down'].q_BH:.2f}); all: {best['all'].term} (P = {best['all'].p:.1e}, FDR {best['all'].q_BH:.2f},
  family-wise P = {best['all'].fwer_random:.3f}). No term at FDR < 0.05 in any list.
Subtypes: down vs up lineage scores {pf(SUB['up_vs_down'])}; down genes vs all genes {pf(SUB['down_vs_background'])}; up genes vs all genes
  {pf(SUB['up_vs_background'])}; beyond DE-matched genes {pf(SUB['p_vs_DE_matched'])}; transcriptome-wide rho = {SUB['rho_transcriptome']:.2f}
  (label-shuffle {pf(SUB['rho_p_label_shuffle'])}){' - the lineage signal is concentrated in the panel, not spread over all genes' if SUB['rho_p_label_shuffle'] > 0.05 else ' - the shift also runs through the whole transcriptome'}.
  Subtype marker overlaps at FDR < 0.05: """ + "; ".join(f"{r.subtype} x {r.list} ({r.hits}: {r.genes}; FDR {r.q_BH:.3f})" for r in sig_sub.itertuples()) + f"""
Pathway neighbours: {len(CT)} terms; best {nnb.term} ({nnb.anchors}; P = {nnb.p:.3f}, FDR {nnb.fdr:.2f}); none at FWER or FDR < 0.05.
Known PD association (Open Targets): """ + "; ".join(f"{r.symbol} {r.pd_association_open_targets:.2f}" + (f" (genetic {r.pd_genetic_association:.2f})" if r.pd_genetic_association > 0 else "")
                                                    for r in GT.sort_values("pd_association_open_targets", ascending=False).head(8).itertuples()) + "\n"
print(TEXT)
open(OUT / "enrich_legend_methods.txt", "w").write(TEXT)
