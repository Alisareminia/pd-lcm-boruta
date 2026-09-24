"""Builds PD_LCM_rf_shap_figures.ipynb - secondary notebook that only draws the paper's
SHAP figures for the Boruta + Random Forest panel:

  paper Figure 4 (file Figure02_shap_directional) - directional SHAP blades, mean |SHAP|
      over ten Random Forest runs with the SD across runs as a halo;
  paper Figure 5 (file Figure03_shap_beeswarm)    - per-person SHAP, PD above each line
      and controls below, dumbbell between the class means.

The figure code is taken verbatim from the merged figures notebook. The only change is
where the SHAP values come from: there is no single held-out test set any more, so each
person's values come from forests trained on the core notebook's folds without them
(10 runs x 5 repeats of 5-fold), which puts all 63 people on the page.
"""
import pathlib, re
from kaggle_nb import write_nb
HERE = pathlib.Path(__file__).parent
FC = HERE / "figcode"
SRC = (HERE.parent / "merged/build_figures_part1.py").read_text()

def grab(start, end):
    a = SRC.index(start); b = SRC.index(end, a)
    return SRC[a:b]
FIG2 = grab("# ============================== FIGURE 2 ==============================",
            "''')\n# ---------------------------------------------------------------- FIG 3")
FIG3 = grab("# ============================== FIGURE 3 ==============================",
            "''')\n\n# ---------------------------------------------------------------- FIG 4")

def swap(s, a, b):
    assert a in s, f"not found: {a[:70]}"
    return s.replace(a, b)

# ---- paper Figure 4: blades --------------------------------------------------------------
FIG2 = swap(FIG2, 'imp = T("05_boruta_shap_importance").copy()', "imp = IMP.copy()")
FIG2 = swap(FIG2, "FW, FH = 13.0, 9.0", "FW, FH = 13.0, max(9.0, 3.2 + 0.25 * len(IMP))")
FIG2 = swap(FIG2, '"higher expression pushes towards, with its spread across ten model runs"',
            '"higher expression pushes towards, with its spread across ten Random Forest runs"')
FIG2 = swap(FIG2, 'f"across 10 model runs   ·   a bold name',
            'f"across 10 runs   ·   every person scored out of fold   ·   a bold name')

# ---- paper Figure 5: split beeswarm ----------------------------------------------------------
FIG3 = swap(FIG3, "fig = plt.figure(figsize=(13.4, 9.3))", "fig = plt.figure(figsize=(13.4, max(9.3, 4.0 + 0.26 * len(IMP))))")
FIG3 = swap(FIG3, "if not HAS_FIGDATA:", "if SV.empty:")
FIG3 = swap(FIG3, 'sv = T("15_shap_values_test")', "sv = SV")
FIG3 = swap(FIG3, 'imp3 = T("05_boruta_shap_importance").set_index("gene")', 'imp3 = IMP.set_index("gene")')
FIG3 = swap(FIG3, "s=31, alpha=0.95", "s=31 if len(PERSON) <= 20 else 17, alpha=0.95")
FIG3 = swap(FIG3, '"Held-out test people   ·   one point per person, PD "',
            '"Every person, scored by forests that never saw them   ·   one point per person, PD "')

CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md("""# SHAP figures for the Boruta + Random Forest panel

Draws the paper's two SHAP figures for the Boruta panel of **`pd-lcm-rf-boruta-panel`**,
using the people and folds of **`pd-lcm-rf-core`**.

* **Paper Figure 4** (`Figure02_shap_directional`) - each gene's mean |SHAP| on the side of
  the PD call that higher expression pushes towards; the pale halo is the SD across ten
  Random Forest runs; bold names with a node on the axis are also nominally differentially
  expressed.
* **Paper Figure 5** (`Figure03_shap_beeswarm`) - one point per person, PD above each
  gene's line and controls below, coloured by that gene's expression; the dumbbell joins
  the two class means.

**Where the SHAP values come from.** For each of ten runs (different forest seeds), a
Random Forest on the Boruta genes is trained on every training fold of the core notebook's
5 x 5-fold split and explained with TreeSHAP on that fold's held-out people, so every person
is explained by forests that never saw them. Boruta itself chose the genes on all 63 people.""")

code((FC / "design.py").read_text())

loader = (FC / "loader.py").read_text()
loader = loader.replace("SRC = find_outputs_dir()",
                        'import os\nSRC = Path(os.environ["FIG_SRC"]) if "FIG_SRC" in os.environ else find_outputs_dir()')
loader = loader.replace("'alisaremi/pd-lcm-merged-pipeline'", "'alisaremi/pd-lcm-rf-boruta-panel'")
code(loader)

code(r'''# ============================== OUT-OF-FOLD SHAP, TEN RUNS ==============================
import os, json, shap
from scipy import stats
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier

def find_any(pattern):
    roots = ["/kaggle/input"] if Path("/kaggle/input").exists() else os.environ.get("FIG_EXTRA", ".").split(":")
    for root in roots:
        hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True), key=len)
        if hits:
            return hits[0]
    raise FileNotFoundError(pattern)

SMOKE   = not Path("/kaggle/input").exists()
SEED    = 42
N_RUNS  = 2 if SMOKE else 10
N_TREES = 50 if SMOKE else 800            # same forest settings as the paper's SHAP runs

cz = np.load(find_any("core_data.npz"), allow_pickle=True)
X, y = cz["X"], cz["y"].astype(int)
PERSON, ALL_GENES = cz["person"].astype(str), [str(g) for g in cz["genes"]]
FOLDS = [f for f in json.load(open(find_any("core_folds.json"))) if f["kind"] == "cv"]
REPS = sorted({f["rep"] for f in FOLDS})
PANEL = T("04_boruta_selected_genes")["gene"].tolist()
XP = X[:, [ALL_GENES.index(g) for g in PANEL]]
print(f"{len(PANEL)} Boruta genes, {len(y)} people, {len(FOLDS)} folds x {N_RUNS} runs")

def one(run, f):
    tr, te = np.array(f["train"]), np.array(f["test"])
    m = RandomForestClassifier(n_estimators=N_TREES, max_depth=10, min_samples_split=4, min_samples_leaf=2,
                               class_weight="balanced", random_state=SEED + 11 * run, n_jobs=1).fit(XP[tr], y[tr])
    v = shap.TreeExplainer(m).shap_values(XP[te])
    v = v[1] if isinstance(v, list) else v
    return run, te, (v[..., 1] if v.ndim == 3 else v)
RUNS = np.zeros((N_RUNS, len(y), len(PANEL)))
for run, te, v in Parallel(n_jobs=os.cpu_count())(delayed(one)(r, f) for r in range(N_RUNS) for f in FOLDS):
    RUNS[run, te] += v / len(REPS)            # each person: mean over the CV repeats inside a run
mean_shap, std_shap = RUNS.mean(0), RUNS.std(0)

de = T("03_de_results_full").set_index("gene")
dirs = []
for k, g in enumerate(PANEL):                 # does HIGHER expression push the call towards PD?
    rho = stats.spearmanr(XP[:, k], mean_shap[:, k])[0]
    dirs.append(np.sign(rho) if np.isfinite(rho) and rho != 0 else np.sign(de.loc[g, "log2FC"]))
IMP = pd.DataFrame({"gene": PANEL, "symbol": [sym(g) for g in PANEL], "mean_abs_shap": np.abs(mean_shap).mean(0),
                    "mean_shap": mean_shap.mean(0), "shap_std": std_shap.mean(0), "direction_sign": np.array(dirs, int)})
IMP["signed_importance"] = IMP.direction_sign * IMP.mean_abs_shap
IMP["direction"] = np.where(IMP.direction_sign > 0, "Higher expression raises PD probability",
                            "Higher expression lowers PD probability")
IMP = IMP.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
IMP["shap_rank"] = range(1, len(IMP) + 1)
SV = pd.DataFrame([{"gene": g, "person": PERSON[i], "shap_value": mean_shap[i, k], "feature_value": XP[i, k],
                    "true_label": int(y[i])} for k, g in enumerate(PANEL) for i in range(len(y))])
IMP.to_csv(OUT_DIR / "shap_oof_importance.csv", index=False)
SV.to_csv(OUT_DIR / "shap_oof_values.csv", index=False)
print(IMP[["shap_rank", "symbol", "mean_abs_shap", "shap_std", "direction_sign"]].round(4).to_string(index=False))''')

md("## Paper Figure 4 - directional SHAP importance")
code(FIG2)
md("## Paper Figure 5 - per-person SHAP, split by diagnosis")
code(FIG3)

write_nb(HERE / "PD_LCM_rf_shap_figures.ipynb", CELLS, "shp")
