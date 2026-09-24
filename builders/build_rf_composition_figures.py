"""Builds PD_LCM_rf_composition_figures.ipynb - figures for pd-lcm-rf-pathway-composition.

Reads only that notebook's outputs; nothing is recomputed.
  Figure 9  - g:Profiler with the 5,622-gene background and Benjamini-Hochberg FDR, with what random gene lists of the
              same size get under identical settings
  Figure 10 - the neuron-mix test: every donor's SOX6-vs-CALB1 composition, how much of the classifier follows it, and
              what survives when it is regressed out of every gene
"""
import pathlib
from kaggle_nb import write_nb
HERE = pathlib.Path(__file__).parent
FC = HERE / "figcode"
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))
md("""# Figures: pathways and the neuron mix

Reads only `pd-lcm-rf-pathway-composition`. Print size, 183 mm wide, pathway names written in full.""")
code((FC / "pwcfig_data.py").read_text())
code((FC / "ext_unified_style.py").read_text())
md("## Figure 9 - g:Profiler pathways with the requested settings (183 x 196 mm)")
code((FC / "fig_gprofiler.py").read_text())
md("## Figure 10 - the neuron-mix test (183 x 165 mm)")
code((FC / "fig_composition.py").read_text())
md("## Ready-to-paste legends, methods and numbers")
code((FC / "pwc_text.py").read_text())
write_nb(HERE / "PD_LCM_rf_composition_figures.ipynb", CELLS, "cfg")
