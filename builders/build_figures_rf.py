"""Builds PD_LCM_rf_figures.ipynb - secondary notebook: figures for the Random Forest analysis.

Reads the outputs of pd-lcm-rf-boruta-panel (tables in the pipeline format), pd-lcm-rf-core
(the core model) and pd-lcm-rf-confirm (the sweep's strict checks). Draws Figure 4
(DE landscape with the Boruta panel), Figure 5 (each Boruta gene alone), the performance
table, and the core-model figure, with the same design system as the merged figures notebook.
"""
import pathlib
from kaggle_nb import write_nb
HERE = pathlib.Path(__file__).parent
FC = HERE / "figcode"
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md("""# Figures: Random Forest on laser-captured dopamine neurons in Parkinson's disease

Reads three notebooks' outputs: **`pd-lcm-rf-boruta-panel`** (gene panel tables),
**`pd-lcm-rf-core`** (the core Random Forest) and **`pd-lcm-rf-confirm`** (the sweep's
strict checks). Figures are written to `outputs/figures` as PDF and PNG.""")

code((FC / "design.py").read_text())

loader = (FC / "loader.py").read_text()
loader = loader.replace("SRC = find_outputs_dir()",
                        'import os\nSRC = Path(os.environ["FIG_SRC"]) if "FIG_SRC" in os.environ else find_outputs_dir()')
loader = loader.replace("'alisaremi/pd-lcm-merged-pipeline'", "'alisaremi/pd-lcm-rf-boruta-panel'")
code(loader)

code(r'''# ============================== CORE MODEL AND STRICT CHECKS ==============================
import json, os
def find_any(pattern):
    roots = ["/kaggle/input"] if Path("/kaggle/input").exists() else os.environ.get("FIG_EXTRA", ".").split(":")
    for root in roots:
        hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True), key=len)
        if hits:
            return hits[0]
    return None
HEAD_NAME  = "Ranks + PCA 30 + Random Forest"
PERF_ALL   = pd.read_csv(find_any("performance_all_models.csv"))
CORE_ROC   = pd.read_csv(find_any("core_roc_curve.csv"))
CORE_COMP  = pd.read_csv(find_any("core_component_importance.csv"))
CORE_GENES = pd.read_csv(find_any("core_gene_importance.csv"))
_y = pd.read_csv(find_any("core_oof_scores.csv"))["y"]
MAJ = float(max(_y.mean(), 1 - _y.mean()))
_c = find_any("S10_confirmation.json"); CONF = json.load(open(_c)) if _c else None
_a = find_any("panel_core_agreement.json"); AGREE = json.load(open(_a)) if _a else None
print(PERF_ALL[["model", "cv_auc", "cv_accuracy", "lodo_auc"]].round(3).to_string(index=False))
print("strict checks from pd-lcm-rf-confirm:", "found" if CONF else "not attached")''')

md("## Figure 4 - the differential-expression landscape with the Boruta panel")
fig4 = (FC / "fig4.py").read_text()
fig4 = fig4.replace('f"the {n_bor_t}-gene Boruta panel selected independently of them"',
                    'f"the {n_bor_t} genes Boruta + Random Forest confirmed on all 63 people, chosen without DE"')
code(fig4)

md("## Figure 5 - each Boruta gene on its own")
fig5 = (FC / "fig5.py").read_text()
fig5 = fig5.replace('N_TEST = int(T("02_baseline_rf_performance").set_index("metric").loc["n_test", "value"])', 'N_TEST = N_PEOPLE')
fig5 = fig5.replace('"Each gene alone, on the held-out test set"', '"Each gene alone, out of fold across all 63 people"')
fig5 = fig5.replace("""fig.text(0.055, 0.913, f"direction-corrected   ·   ranked by AUC   ·   the "
         f"{len(order) - n_sig} genes whose bootstrap interval includes 0.5 are drawn as one "
         f"grey thicket   ·   test n = {N_TEST}", ha="left", va="center", fontsize=9.5, color=MUTED)""",
"""fig.text(0.055, 0.913, f"direction-corrected   ·   ranked by AUC   ·   10 x 5-fold CV over all "
         f"{N_TEST} people   ·   the strongest are coloured, the rest are one grey thicket",
         ha="left", va="center", fontsize=9.5, color=MUTED)""")
fig5 = fig5.replace("""fig.text(0.980, 0.940, f"{n_sig} of {len(order)} clear chance on their own",
         ha="right", va="center", fontsize=10.6, color=DEEP_RED,
         fontweight="bold")""",
"""fig.text(0.980, 0.940, f"{n_sig} of {len(order)} have a bootstrap interval above 0.5",
         ha="right", va="center", fontsize=10.6, color=DEEP_RED, fontweight="bold")""")
fig5 = fig5.replace("""fig.text(0.980, 0.913, "colour = bootstrap 95% CI excludes AUC 0.5",
         ha="right", va="center", fontsize=9.5, color=MUTED)""",
"""fig.text(0.055, 0.032, "Boruta chose these genes on all 63 people, so a gene's own interval is not an independent test; "
         "the ranking between them is what this figure is for.", ha="left", va="center", fontsize=9.4, color=MUTED)""")
code(fig5)

md("## Performance table")
code((FC / "perf_table_rf.py").read_text())

md("## The core Random Forest")
code((FC / "fig_core.py").read_text())

code(r'''print("\n".join(f"{i + 1}. {f['figure']}: {f['headline']}" for i, f in enumerate(FIGURES)))''')

write_nb(HERE / "PD_LCM_rf_figures.ipynb", CELLS, "fig")
