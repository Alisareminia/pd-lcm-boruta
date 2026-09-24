"""Shared pieces for the core / secondary Kaggle notebooks of the Random Forest project.

Each builder imports SETUP (imports, Kaggle-or-smoke switch, find_input) and write_nb.
Off Kaggle a notebook runs in a tiny smoke mode, reading local files instead of
/kaggle/input; the core's smoke output folder stands in for the core notebook.
"""
import ast, json, pathlib

SETUP = r'''import subprocess, sys, os, time, json, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
for _a, _t in [("float", float), ("int", int), ("bool", bool), ("object", object), ("str", str)]:
    if not hasattr(np, _a):
        setattr(np, _a, _t)
ON_KAGGLE = Path("/kaggle/input").exists()
SMOKE = not ON_KAGGLE
OUT = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("SMOKE_OUT", "smoke_out"))
OUT.mkdir(exist_ok=True, parents=True)
LOCAL_IN = os.environ.get("SMOKE_IN", "data").split(":")
N_CPU = os.cpu_count()
SEED = 42
t0 = time.time()
def log(m): print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

def find_input(pattern):
    roots = ("/kaggle/input",) if ON_KAGGLE else tuple(LOCAL_IN)
    for root in roots:
        hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True), key=len)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{pattern} not found under {roots}")
print("Kaggle" if ON_KAGGLE else "LOCAL SMOKE RUN", "| CPUs:", N_CPU)'''

# the headline model, fixed by the sweep in pd-lcm-rf-confirm (top AUC of 97 variants)
BLOCKS = r'''from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, balanced_accuracy_score, recall_score,
                             f1_score, matthews_corrcoef, brier_score_loss)

HEAD_NAME = "Ranks + PCA 30 + Random Forest"
HEAD = ("rpca", 30, dict(max_features=0.5, min_samples_leaf=1))

def rf(n_jobs=1, **kw):
    p = dict(n_estimators=N_TREES, max_features="sqrt", min_samples_leaf=1, class_weight="balanced",
             oob_score=True, n_jobs=n_jobs, random_state=SEED)
    p.update(kw); return RandomForestClassifier(**p)

def features(kind, arg, tr, te):
    if kind == "all":
        return X[tr], X[te]
    if kind == "genes":
        return X[tr][:, arg], X[te][:, arg]
    src = XR if kind == "rpca" else X
    p = PCA(arg, random_state=SEED).fit(src[tr])
    return p.transform(src[tr]), p.transform(src[te])

def oob_threshold(m, yt):
    """Accuracy cut-off chosen on the forest's out-of-bag predictions for the training people."""
    s = m.oob_decision_function_[:, 1]; ok = np.isfinite(s); s, yt = s[ok], yt[ok]
    u = np.unique(np.round(s, 6))
    cuts = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]]) if len(u) > 1 else np.array([0.5])
    acc = [accuracy_score(yt, (s > c).astype(int)) for c in cuts]
    best = np.flatnonzero(np.isclose(acc, max(acc)))
    return float(cuts[best[len(best) // 2]])

def fit_score(spec, tr, te, yy, n_jobs=1):
    kind, arg, kw = spec
    Xtr, Xte = features(kind, arg, tr, te)
    m = rf(n_jobs=n_jobs, **kw).fit(Xtr, yy[tr])
    return m.predict_proba(Xte)[:, 1], oob_threshold(m, yy[tr])

def metrics(yt, s, th=None):
    yh = (s >= 0.5).astype(int)
    out = {"auc": roc_auc_score(yt, s), "accuracy": accuracy_score(yt, yh), "bal_accuracy": balanced_accuracy_score(yt, yh),
           "sensitivity": recall_score(yt, yh), "specificity": recall_score(1 - yt, 1 - yh),
           "f1": f1_score(yt, yh, zero_division=0), "mcc": matthews_corrcoef(yt, yh), "brier": brier_score_loss(yt, s)}
    if th is not None:
        out["accuracy_oob_cut"] = accuracy_score(yt, (s > th).astype(int))
    return out

def summarise(name, recs, key):
    """Metrics of one model over the fold records: mean over CV repeats, pooled over LODO folds."""
    reps = sorted({o["rep"] for o in recs if o["kind"] == "cv"})
    per = []
    for r in reps:
        fr = [o for o in recs if o["kind"] == "cv" and o["rep"] == r]
        idx = np.concatenate([o["test"] for o in fr]); s = np.concatenate([o[key]["score"] for o in fr])
        th = np.concatenate([np.full(len(o["test"]), o[key]["thr"]) for o in fr])
        per.append(metrics(y[idx], s, th))
    d = pd.DataFrame(per)
    lo = [o for o in recs if o["kind"] == "lodo"]
    li = np.concatenate([o["test"] for o in lo]); ls = np.concatenate([o[key]["score"] for o in lo])
    row = {"model": name, **{f"cv_{k}": d[k].mean() for k in d.columns},
           "cv_auc_sd": d.auc.std(ddof=1) if len(d) > 1 else np.nan,
           "cv_accuracy_sd": d.accuracy.std(ddof=1) if len(d) > 1 else np.nan,
           "lodo_auc": roc_auc_score(y[li], ls), "lodo_accuracy": accuracy_score(y[li], (ls >= 0.5).astype(int))}
    for o in lo:
        row[f"lodo_auc_{o['tag'][5:]}"] = roc_auc_score(y[np.array(o["test"])], o[key]["score"])
    return row'''

LOAD_CORE = r'''# ============================== LOAD THE CORE NOTEBOOK'S OUTPUT ==============================
cz = np.load(find_input("core_data.npz"), allow_pickle=True)
X, XR, y = cz["X"], cz["XR"], cz["y"].astype(int)
DS, PERSON, GENES = cz["ds"].astype(str), cz["person"].astype(str), [str(g) for g in cz["genes"]]
STRATA = np.array([f"{d}_{v}" for d, v in zip(DS, y)])
FOLDS = json.load(open(find_input("core_folds.json")))
for f in FOLDS:
    f["train"], f["test"] = np.array(f["train"]), np.array(f["test"])
SYM = pd.read_csv(find_input("15_gene_symbol_map.csv")).set_index("gene")["symbol"].to_dict()
sym = lambda g: SYM[g] if isinstance(SYM.get(g), str) else g
print(f"core data: {len(y)} people, {X.shape[1]:,} genes, {sum(f['kind'] == 'cv' for f in FOLDS)} CV folds + "
      f"{sum(f['kind'] == 'lodo' for f in FOLDS)} leave-one-study-out folds")'''


def write_nb(path, cells, prefix):
    for k, s in cells:
        if k == "code":
            ast.parse(s)

    def to_source(t):
        ls = t.split("\n")
        return [l + "\n" for l in ls[:-1]] + ([ls[-1]] if ls[-1] else [])
    nb = {"cells": [{"cell_type": k, "id": f"{prefix}{i:03d}", "metadata": {}, "source": to_source(s),
                     **({"execution_count": None, "outputs": []} if k == "code" else {})}
                    for i, (k, s) in enumerate(cells)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.12.13"}},
          "nbformat": 4, "nbformat_minor": 5}
    pathlib.Path(path).write_text(json.dumps(nb, indent=1))
    print(f"wrote {pathlib.Path(path).name}: {len(cells)} cells, syntax ok")


def smoke_source(path):
    """All code cells of a notebook joined into one script, for a local smoke run."""
    nb = json.loads(pathlib.Path(path).read_text())
    return "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
