# Boruta-Driven Explainable Machine Learning for Identifying Gene Signatures in Laser-Captured Dopamine Neurons of Parkinson's Disease

Code, notebooks and result tables behind the manuscript. Everything here runs on public data: four
laser-capture microdissection (LCM) studies of human nigral dopamine neurons are merged at one profile
per person, a Random Forest and Boruta select a 30-gene panel, and both models are then frozen and
tested in eight independent bulk substantia nigra cohorts.

Manuscript PDF: [`manuscript/boruta_lcm_pd_signatures.pdf`](manuscript/boruta_lcm_pd_signatures.pdf)

## What is in here

| Folder | Contents |
| --- | --- |
| `notebooks/` | The 17 Kaggle notebooks, exactly as run |
| `builders/` | The Python scripts that generate each notebook, the figure code (`figcode/`) and the Kaggle metadata |
| `results/` | Every table the figures read: model performance, the Boruta panel, external cohorts, enrichment, composition, deconvolution and panel stability |
| `figures/` | Figures 1–10 as PDF and PNG |
| `manuscript/` | The paper, the standalone gene table, the gene master table and the literature-search record |
| `manuscript/source/` | The LaTeX source, `build.sh`, the bibliography and every script that generates a table |

Notebooks are written by the scripts in `builders/`, not by hand: `python3 builders/build_fig1.py`
rewrites `PD_LCM_rf_figure1.ipynb`, and `push_kernel.py` uploads it. Credentials are read from
`~/.kaggle/access_token` at run time and are not in this repository.

## Notebooks and what they produce

| Notebook | Kaggle | Output |
| --- | --- | --- |
| `PD_LCM_merged_pipeline` | [pd-lcm-merged-pipeline](https://www.kaggle.com/code/alisaremi/pd-lcm-merged-pipeline) | Merge of the four LCM cohorts, gene filtering, differential expression |
| `PD_LCM_rf_core` | [pd-lcm-rf-core](https://www.kaggle.com/code/alisaremi/pd-lcm-rf-core) | The core classifier: within-person ranks, PCA, Random Forest |
| `PD_LCM_rf_boruta_panel` | [pd-lcm-rf-boruta-panel](https://www.kaggle.com/code/alisaremi/pd-lcm-rf-boruta-panel) | Boruta selection and SHAP, the 30-gene panel |
| `PD_LCM_rf_confirm` | [pd-lcm-rf-confirm](https://www.kaggle.com/code/alisaremi/pd-lcm-rf-confirm) | Nested model choice, label permutation, leave-one-study-out |
| `PD_LCM_rf_figure1` | [pd-lcm-rf-figure-1](https://www.kaggle.com/code/alisaremi/pd-lcm-rf-figure-1) | Figure 1, study design |
| `PD_LCM_rf_figure2` | [pd-lcm-rf-figure-2](https://www.kaggle.com/code/alisaremi/pd-lcm-rf-figure-2) | Figure 2, classifier performance |
| `PD_LCM_rf_figures34_print` | [pd-lcm-rf-figures-3-4](https://www.kaggle.com/code/alisaremi/pd-lcm-rf-figures-3-4) | Figures 3 and 4, SHAP ranking and beeswarm |
| `PD_LCM_rf_volcano_venn` | [pd-lcm-rf-volcano-venn](https://www.kaggle.com/code/alisaremi/pd-lcm-rf-volcano-venn) | Figure 5, volcano, panel overlap and per-cohort effects |
| `PD_LCM_rf_external_multi` | [pd-lcm-rf-external-multi](https://www.kaggle.com/code/alisaremi/pd-lcm-rf-external-multi) | The eight external cohorts, donor-overlap check, frozen and strictly independent models |
| `PD_LCM_rf_external_unified` | [pd-lcm-rf-external-unified](https://www.kaggle.com/code/alisaremi/pd-lcm-rf-external-unified) | Figure 6, pooled external validation |
| `PD_LCM_rf_external_figure7` | [pd-lcm-rf-external-figure-7](https://www.kaggle.com/code/alisaremi/pd-lcm-rf-external-figure-7) | Figure 7, cohort by cohort |
| `PD_LCM_rf_enrichment` | [pd-lcm-rf-enrichment](https://www.kaggle.com/code/alisaremi/pd-lcm-rf-enrichment) | Dopamine-neuron subtype scoring against Kamath et al. |
| `PD_LCM_rf_enrichment_figures` | [pd-lcm-rf-enrichment-figures](https://www.kaggle.com/code/alisaremi/pd-lcm-rf-enrichment-figures) | Figure 8, the panel against subtypes |
| `PD_LCM_rf_pathway_composition` | [pd-lcm-rf-pathway-composition](https://www.kaggle.com/code/alisaremi/pd-lcm-rf-pathway-composition) | Over-representation with the measured background, and the neuron-mix adjustment |
| `PD_LCM_rf_composition_figures` | [pd-lcm-rf-composition-figures](https://www.kaggle.com/code/alisaremi/pd-lcm-rf-composition-figures) | Figures 9 and 10 |
| `PD_LCM_rf_gsea` | [pd-lcm-rf-gsea](https://www.kaggle.com/code/alisaremi/pd-lcm-rf-gsea) | Exploratory gene set enrichment over the full ranking |
| `PD_LCM_rf_gsea_figures` | [pd-lcm-rf-gsea-figures](https://www.kaggle.com/code/alisaremi/pd-lcm-rf-gsea-figures) | Figures for the exploratory analysis (not in the paper) |
| `PD_LCM_rf_deconvolution` | [pd-lcm-rf-deconvolution](https://www.kaggle.com/code/alisaremi/pd-lcm-rf-deconvolution) | Reference-based deconvolution of the external cohorts against a single-nucleus midbrain atlas |
| `PD_LCM_rf_panel_stability` | [pd-lcm-rf-panel-stability](https://www.kaggle.com/code/alisaremi/pd-lcm-rf-panel-stability) | Does a smaller, more stable panel transfer as well? Every fold-frequency threshold, against random panels |

## The twelve cohorts

All expression data are public in the Gene Expression Omnibus.

**Discovery, laser-captured dopamine neurons (63 donors, one profile each)**

| Accession | Platform | Control / PD | Source |
| --- | --- | --- | --- |
| [GSE182622](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE182622) | RNA-seq | 10 / 12 | Tïklová et al. 2021, doi:[10.3389/fnmol.2021.763777](https://doi.org/10.3389/fnmol.2021.763777) |
| [GSE20141](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE20141) | Affymetrix U133 Plus 2.0 | 8 / 10 | Zheng et al. 2010, doi:[10.1126/scitranslmed.3001059](https://doi.org/10.1126/scitranslmed.3001059) |
| [GSE24378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE24378) | Affymetrix U133 X3P | 9 / 8 | Zheng et al. 2010, doi:[10.1126/scitranslmed.3001059](https://doi.org/10.1126/scitranslmed.3001059) |
| [GSE169755](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE169755) | RNA-seq | 3 / 3 | Zaccaria et al. 2021, doi:[10.1007/s10571-021-01146-8](https://doi.org/10.1007/s10571-021-01146-8) |

**External validation, bulk substantia nigra (158 donors after 14 shared brains were removed)**

| Accession | Platform | Source |
| --- | --- | --- |
| [GSE7621](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE7621) | U133 Plus 2.0 | Lesnick et al. 2007, doi:[10.1371/journal.pgen.0030098](https://doi.org/10.1371/journal.pgen.0030098) |
| [GSE8397](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE8397) | U133A | Moran et al. 2006, doi:[10.1007/s10048-005-0020-2](https://doi.org/10.1007/s10048-005-0020-2) |
| [GSE20163](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE20163) | U133A | Zheng et al. 2010, doi:[10.1126/scitranslmed.3001059](https://doi.org/10.1126/scitranslmed.3001059) |
| [GSE20164](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE20164) | U133A | Zheng et al. 2010, doi:[10.1126/scitranslmed.3001059](https://doi.org/10.1126/scitranslmed.3001059) |
| [GSE20292](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE20292) | U133A | Zhang et al. 2005, doi:[10.1002/ajmg.b.30195](https://doi.org/10.1002/ajmg.b.30195) |
| [GSE49036](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE49036) | U133 Plus 2.0 | Dijkstra et al. 2015, doi:[10.1371/journal.pone.0128651](https://doi.org/10.1371/journal.pone.0128651) |
| [GSE114517](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114517) | RNA-seq | Simchovitz et al. 2020, doi:[10.1111/acel.13115](https://doi.org/10.1111/acel.13115) |
| [GSE168496](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE168496) | RNA-seq | Tranchevent et al. 2023, doi:[10.1038/s41531-023-00446-8](https://doi.org/10.1038/s41531-023-00446-8) |

Dopamine-neuron subtype markers come from the single-nucleus atlas of Kamath et al. 2022,
doi:[10.1038/s41593-022-01061-1](https://doi.org/10.1038/s41593-022-01061-1) (Supplementary Table 8).

**Single-nucleus references**

| Accession | Use | Source |
| --- | --- | --- |
| — | Dopamine-neuron subtype markers (Supplementary Table 8) | Kamath et al. 2022, doi:[10.1038/s41593-022-01061-1](https://doi.org/10.1038/s41593-022-01061-1) |
| [GSE157783](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE157783) | Cell-type reference for deconvolution: 41,434 nuclei, 6 control and 5 PD | Smajić et al. 2022, doi:[10.1093/brain/awab446](https://doi.org/10.1093/brain/awab446) |

The deconvolution notebook downloads GSE157783 itself and writes the signature matrix, the per-donor
proportions and the agreement with the eight-marker score to `results/deconvolution/`.

## Reproducing the manuscript

`manuscript/source/` holds everything the paper is built from. With a TeX installation that has
`pdflatex` and `bibtex`, `./build.sh` regenerates the PDF and the DOCX, including Table 3 and the
supplementary search record, from the tables in `results/`.

## Reproducing

Every notebook reads the outputs of the ones before it and refits nothing that is already fixed.
The order is: `merged_pipeline` → `rf_core` → `rf_boruta_panel` → `rf_confirm` → the figure notebooks,
then `rf_external_multi` → `rf_external_unified`, `rf_enrichment`, `rf_pathway_composition` and their
figure notebooks. Random seeds are set inside each notebook. `results/` holds the tables they produce,
so a figure can be redrawn without rerunning the analysis.

## Contact

Ali Sareminia — questions and requests through the repository issues.
