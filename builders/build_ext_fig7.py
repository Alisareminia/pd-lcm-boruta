"""Builds PD_LCM_rf_external_figure7.ipynb - Figure 7 of the external validation, redrawn (version 2).

Reads only the outputs of pd-lcm-rf-external-multi; nothing is downloaded or refitted. Same content as Figure 7 of
pd-lcm-rf-external-unified (left as it is, so the two can be compared), in journal conventions: a meta-analysis
forest plot (square area = cohort weight, AUC [95% CI] columns), a key that defines every mark in both colours,
capitalised table headers, and a donor-overlap matrix instead of a box diagram.
"""
import pathlib
import pandas as pd
from kaggle_nb import write_nb
HERE = pathlib.Path(__file__).parent
FC = HERE / "figcode"
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))
FP = pd.read_csv(HERE.parent / "data/lcm/harmonised/fingerprints.csv")
DISC_N = {d: int((FP.dataset == d).sum()) for d in ("GSE20141", "GSE24378", "GSE182622", "GSE169755")}

md("""# Figure 7 (version 2) - external validation, cohort by cohort

Reads only `pd-lcm-rf-external-multi`; nothing is downloaded or refitted. The earlier Figure 7 in
`pd-lcm-rf-external-unified` is left unchanged for comparison.

* **a** discrimination in each of the eight bulk cohorts and pooled (forest plot)
* **b** accuracy, sensitivity and specificity at thresholds fixed on the discovery donors; balanced accuracy per cohort
* **c** donor overlap between the external cohorts and the discovery studies""")
code("DISC_N = " + repr(DISC_N) + "   # discovery donors per study")
code((FC / "ext_unified_compute.py").read_text())
code((FC / "ext_unified_style.py").read_text())
md("## Figure 7 (183 x 150 mm)")
code((FC / "fig_ext_fig7b.py").read_text())
md("## Legend")
code((FC / "ext_fig7b_text.py").read_text())
write_nb(HERE / "PD_LCM_rf_external_figure7.ipynb", CELLS, "ex7")
