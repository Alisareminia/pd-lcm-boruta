"""Builds PD_LCM_rf_external_unified.ipynb - the external validation redrawn with every cohort treated alike.

Reads only the finished outputs of pd-lcm-rf-external-multi (per-person scores, per-cohort results, gene
effects, donor overlap, pooled summaries); nothing is downloaded or refitted, so it runs in about a minute.
The two figures of the external validation are redrawn with no lead cohort: all eight bulk cohorts are
analysed together.
  Figure 6 (unified) - a ROC averaged over the cohorts; b how much of it is neuron loss; c every Boruta gene in
                       every cohort
  Figure 7 (unified) - a forest plot, every cohort alike; b calls at cut-offs fixed before testing; c donor overlap
"""
import json, pathlib
import pandas as pd
from kaggle_nb import write_nb
HERE = pathlib.Path(__file__).parent
FC = HERE / "figcode"
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

FP = pd.read_csv(HERE.parent / "data/lcm/harmonised/fingerprints.csv")
DISC_N = {d: int((FP.dataset == d).sum()) for d in ("GSE20141", "GSE24378", "GSE182622", "GSE169755")}

md("""# External validation, every cohort alike

Redraws the external validation with **no lead cohort**: the eight bulk substantia nigra cohorts found for
`pd-lcm-rf-external-multi` are analysed together (GSE7621 is one of the eight, nothing more). This notebook
reads only that notebook's outputs - nothing is downloaded or refitted.

* **Figure 6** - a: ROC of the frozen models, averaged over the cohorts; b: how much of the signal is neuron loss;
  c: every Boruta gene in every cohort
* **Figure 7** - a: every cohort's AUC; b: calls at cut-offs fixed before testing; c: which brains were shared with
  the discovery data""")

code("DISC_N = " + repr(DISC_N) + "   # discovery people per study")
md("## 1. Read the multi-cohort outputs and pool them")
code((FC / "ext_unified_compute.py").read_text())
code((FC / "ext_unified_style.py").read_text())
md("## 2. Figure 6 - the frozen models across the eight cohorts (183 x 155 mm)")
code((FC / "fig_ext_unified6.py").read_text())
md("## 3. Figure 7 - cohort by cohort, calls, donor overlap (183 x 150 mm)")
code((FC / "fig_ext_unified7.py").read_text())
md("## 4. Ready-to-paste legends, methods and numbers")
code((FC / "ext_unified_text.py").read_text())

write_nb(HERE / "PD_LCM_rf_external_unified.ipynb", CELLS, "exu")
