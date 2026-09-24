"""Builds PD_LCM_rf_enrichment_figures.ipynb - the figures of the enrichment analysis, drawn from pd-lcm-rf-enrichment.

Reads only the outputs of pd-lcm-rf-enrichment (and the Kamath et al. 2022 workbook it saved); nothing is recomputed.
  Figure 8  - the panel on the dopamine-neuron lineage axis: a every measured gene, PD effect against lineage, all 30 panel
              genes labelled; b marker overlap by subtype; c marker strength of each panel gene in each subtype
  Figure 9  - pathways: a over-representation with the panel genes behind each pathway; b pathway partners (label-shuffle test)
  Figure S  - panel gene summary: effect, importance, replication, lineage and known PD association of all 30 genes
Pathway and gene names are written in full (wrapped, never truncated).
"""
import pathlib
from kaggle_nb import write_nb
HERE = pathlib.Path(__file__).parent
FC = HERE / "figcode"
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))
md("""# Figures of the enrichment analysis

Reads only `pd-lcm-rf-enrichment`; nothing is recomputed. Print size, 183 mm wide.

* **Figure 8** - the 30 Boruta genes on the dopamine-neuron lineage axis (Kamath et al. 2022)
* **Figure 9** - pathway over-representation and pathway partners, every pathway named in full
* **Figure S** - a one-page summary of all 30 panel genes""")
code((FC / "strict_xlsx.py").read_text())
code((FC / "enrfig_data.py").read_text())
code((FC / "ext_unified_style.py").read_text())
md("## Figure 8 - dopamine-neuron subtypes (183 x 180 mm)")
code((FC / "fig8_subtypes.py").read_text())
md("## Figure 9 - pathways (183 x 195 mm)")
code((FC / "fig9_pathways.py").read_text())
md("## Figure S - panel gene summary (183 x 134 mm)")
code((FC / "figS_genecard.py").read_text())
write_nb(HERE / "PD_LCM_rf_enrichment_figures.ipynb", CELLS, "enf")
