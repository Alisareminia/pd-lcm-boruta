"""Builds PD_LCM_rf_gsea_figures.ipynb - the figure of the GSEA analysis, drawn from pd-lcm-rf-gsea.

Reads that notebook's outputs (and re-reads the Enrichr libraries only to draw the running-score curves); nothing is recomputed.
  Figure 11 - a the gene sets below FDR 0.25 with their normalised enrichment, both FDRs and the panel genes in their leading
              edges; b the running enrichment score of the three strongest sets along the ranking of all genes; c the same
              pathways scored in every donor, with each pathway's AUC.
"""
import pathlib
from kaggle_nb import write_nb
HERE = pathlib.Path(__file__).parent
FC = HERE / "figcode"
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))
md("""# Figure: pathways from the whole-transcriptome analysis

Reads only `pd-lcm-rf-gsea`. Print size, 183 mm wide, gene-set names written in full.""")
code((FC / "gseafig_data.py").read_text())
code((FC / "ext_unified_style.py").read_text())
md("## Figure 11 (183 x 212 mm)")
code((FC / "fig_gsea.py").read_text())
write_nb(HERE / "PD_LCM_rf_gsea_figures.ipynb", CELLS, "gsf")
