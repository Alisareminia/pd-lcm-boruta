import os, glob, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
warnings.filterwarnings("ignore")
ON_KAGGLE = Path("/kaggle/input").exists()
OUT = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("FIG_OUT", "."))
def find_in(name):
    if not ON_KAGGLE:
        return Path(os.environ["EXT_IN"]) / name
    hits = sorted((h for h in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if "external-multi" in h), key=len)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]
RES = pd.read_csv(find_in("external_multi_results.csv")).set_index(["cohort", "model"])
SCO = pd.read_csv(find_in("external_multi_scores.csv"))
GBC = pd.read_csv(find_in("external_multi_gene_by_cohort.csv"))
GMT = pd.read_csv(find_in("external_multi_gene_meta.csv"))
OVL = pd.read_csv(find_in("donor_overlap.csv")).set_index("cohort")
MATCH = pd.read_csv(find_in("donor_overlap_matches.csv"))
SUM = json.load(open(find_in("external_multi_summary.json")))
POOL = SUM["pooled"]["all"]                       # every cohort, each donor once - the analysis reported
COHORTS = ["GSE7621", "GSE8397", "GSE20163", "GSE20164", "GSE20292", "GSE49036", "GSE114517", "GSE168496"]   # by accession
assert sorted(COHORTS) == sorted(POOL["cohorts"])
P = SCO[SCO.counted_in_pool].copy()               # shared donors already removed; the one repeated external donor counted once

def fast_auc(yv, s):
    r = rankdata(s); n1 = int(yv.sum()); n0 = len(yv) - n1
    return (r[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
BY = {c: P[P.cohort == c] for c in COHORTS}
W = np.array([d.y.sum() * (1 - d.y).sum() for d in BY.values()], float)       # PD-control pairs per cohort
def pooled(fn):
    return float(np.sum(W * np.array([fn(d) for d in BY.values()])) / W.sum())
SCORE = {"core": lambda d: d.frozen.to_numpy(), "panel": lambda d: d.panel.to_numpy(), "neuron": lambda d: -d.neuron_score.to_numpy()}

# ROC averaged over cohorts: each cohort's curve, averaged vertically with the same pair weights as the pooled AUC
GRID = np.linspace(0, 1, 1001)
def roc_xy(yv, s):
    order = np.argsort(-s, kind="mergesort"); ys, ss = yv[order], s[order]
    keep = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]                    # one point per distinct score
    tp, fp = np.cumsum(ys)[keep], np.cumsum(1 - ys)[keep]
    return np.r_[0, fp / fp[-1]], np.r_[0, tp / tp[-1]]
ROC = {}
for key, fn in SCORE.items():
    tprs = []
    for d in BY.values():
        f, t = roc_xy(d.y.to_numpy(), fn(d))
        tprs.append(np.interp(GRID, f + np.arange(len(f)) * 1e-9, t))
    ROC[key] = np.sum(W[:, None] * np.array(tprs), axis=0) / W.sum()

# label shuffles within cohorts, for the pooled AUCs
rng = np.random.default_rng(42)
PERM = {}
for key in ("core", "panel"):
    obs = pooled(lambda d: fast_auc(d.y.to_numpy(), SCORE[key](d)))
    null = np.array([pooled(lambda d: fast_auc(rng.permutation(d.y.to_numpy()), SCORE[key](d))) for _ in range(10000)])
    PERM[key] = (np.sum(null >= obs) + 1) / (len(null) + 1)

# neuron content: leave-one-out logistic models inside each cohort, then pooled like the AUCs
def loo(F, yv):
    return fast_auc(yv, cross_val_predict(LogisticRegression(), F, yv, cv=LeaveOneOut(), method="predict_proba")[:, 1])
LOO = {"neurons": pooled(lambda d: loo(d[["neuron_score"]].to_numpy(), d.y.to_numpy())),
       "neurons+core": pooled(lambda d: loo(d[["neuron_score", "frozen"]].to_numpy(), d.y.to_numpy())),
       "neurons+panel": pooled(lambda d: loo(d[["neuron_score", "panel"]].to_numpy(), d.y.to_numpy()))}
P["core_centred"] = P.frozen - P.groupby("cohort").frozen.transform("mean")
P["panel_centred"] = P.panel - P.groupby("cohort").panel.transform("mean")
RHO = {"core": spearmanr(P.neuron_score, P.core_centred)[0], "panel": spearmanr(P.neuron_score, P.panel_centred)[0]}

A = POOL["auc"]
NEURO_TABLE = [("neuron markers alone", A["neuron"]["auc"], False),
               ("core classifier", A["frozen"]["auc"], False),
               ("core, neuron content removed", A["frozen|neuron_removed"]["auc"], True),
               ("panel forest", A["panel"]["auc"], False),
               ("panel forest, neuron content removed", A["panel|neuron_removed"]["auc"], True),
               ("neurons alone, leave-one-out", LOO["neurons"], False),
               ("neurons + core score, leave-one-out", LOO["neurons+core"], False),
               ("neurons + panel score, leave-one-out", LOO["neurons+panel"], False)]
UNI = {"people": POOL["people"], "control": POOL["control"], "PD": POOL["PD"], "cohorts": COHORTS,
       "auc": A, "calls@0.5": POOL["calls@0.5"], "calls@oob": POOL["calls@oob"], "perm_p_within_cohorts": PERM,
       "loo": LOO, "spearman_with_neuron_centred": RHO, "gene_agreement": SUM["gene_agreement"],
       "sensitivity_without_GSE7621": {k: SUM["pooled"]["unseen"]["auc"][k] for k in ("frozen", "panel", "neuron")}}
json.dump(UNI, open(OUT / "external_unified_summary.json", "w"), indent=1, default=float)
print(f"{len(COHORTS)} cohorts, {POOL['people']} people ({POOL['control']} control, {POOL['PD']} PD)")
for lab, v, _ in NEURO_TABLE:
    print(f"  {lab:40s} {v:.3f}")
print("  label shuffles within cohorts:", {k: f"{v:.5f}" for k, v in PERM.items()}, " rho with neuron content:", {k: round(v, 2) for k, v in RHO.items()})
print("  area under the averaged curves:", {k: round(float(np.trapz(v, GRID)), 3) for k, v in ROC.items()})
