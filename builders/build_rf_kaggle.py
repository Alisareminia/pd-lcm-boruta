"""Builds PD_LCM_rf_boruta.ipynb: Boruta + Random Forest with Boruta re-run inside every fold.

Runs on Kaggle next to the local sweep (step9_rf_optimise.py) and uses the same
25 folds, so the numbers can be put in one table. The merged matrix is read from
the pipeline kernel's 18_hvg_expression.npz and reordered to the local person and
gene order; row sums are compared with the local copy before anything is fitted.
"""
import json, pathlib
HERE = pathlib.Path(__file__).parent
REF = json.loads((HERE / "data/mdata_reference.json").read_text())
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md("""# Boruta + Random Forest on the merged LCM cohort, Boruta inside every fold

63 people from four laser-capture studies, one profile per person. Every number
below comes from people the model never saw: Boruta is re-run from scratch on each
training fold, the forest is fitted on the genes it keeps, and the accuracy cut-off
comes from the forest's out-of-bag predictions on the training people.""")

code(r'''import subprocess, sys, time, json, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try:
    import boruta
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "Boruta==0.4.3"], check=True)
for _a, _t in [("float", float), ("int", int), ("bool", bool), ("object", object), ("str", str)]:
    if not hasattr(np, _a):
        setattr(np, _a, _t)
from boruta import BorutaPy
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
import os
N_CPU = os.cpu_count(); print("CPUs:", N_CPU)
OUT = Path("/kaggle/working"); t0 = time.time()
def log(m): print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)''')

code("REF = json.loads(r'''" + json.dumps(REF) + "''')")

code(r'''# ============================== LOAD, REORDER, CHECK ==============================
hit = sorted(glob.glob("/kaggle/input/**/18_hvg_expression.npz", recursive=True), key=len)[0]
z = np.load(hit, allow_pickle=True)
Xk = pd.DataFrame(z["X"].astype(float), index=z["person"].astype(str), columns=[str(g) for g in z["genes"]])
X = Xk.loc[REF["person"], REF["genes"]].to_numpy()
y = np.array(REF["y"]); DS = np.array(REF["ds"])
yk = pd.Series(z["y"].astype(int), index=z["person"].astype(str)).loc[REF["person"]].to_numpy()
diff = np.abs(X.sum(1) - np.array(REF["row_sum"])).max()
print(f"{X.shape[0]} people x {X.shape[1]:,} genes; labels identical: {bool((yk == y).all())}; "
      f"largest row-sum difference from the local matrix: {diff:.4f}")
print("SAME MATRIX AS LOCAL" if diff < 0.05 and (yk == y).all() else "WARNING: matrix differs from the local copy")
STRATA = np.array([f"{d}_{v}" for d, v in zip(DS, y)])''')

code(r'''# ============================== BORUTA INSIDE EVERY FOLD ==============================
SEED, N_REP, N_TREES = 42, 5, 1000
def rf(**kw):
    p = dict(n_estimators=N_TREES, max_features="sqrt", min_samples_leaf=1, class_weight="balanced",
             oob_score=True, n_jobs=N_CPU, random_state=SEED)
    p.update(kw); return RandomForestClassifier(**p)

def oob_threshold(m, yt):
    s = m.oob_decision_function_[:, 1]; ok = np.isfinite(s); s, yt = s[ok], yt[ok]
    u = np.unique(np.round(s, 6))
    cuts = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]]) if len(u) > 1 else np.array([0.5])
    acc = [accuracy_score(yt, (s > c).astype(int)) for c in cuts]
    best = np.flatnonzero(np.isclose(acc, max(acc)))
    return float(cuts[best[len(best) // 2]])

def boruta_genes(tr, perc, seed):
    b = BorutaPy(RandomForestClassifier(max_features="sqrt", class_weight="balanced", n_jobs=N_CPU),
                 n_estimators=500, perc=perc, alpha=0.05, two_step=True, max_iter=100,
                 random_state=seed, verbose=0).fit(X[tr], y[tr])
    conf = np.where(b.support_)[0]
    used = conf if len(conf) >= 2 else np.where(b.support_ | b.support_weak_)[0]
    if len(used) < 2:
        used = np.argsort(b.ranking_)[:2]
    return conf, used

FOLDS = [(f"r{r}", tr, te, SEED + 100 * r + k) for r in range(N_REP)
         for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(X, STRATA))]
FOLDS += [(f"lodo_{d}", np.where(DS != d)[0], np.where(DS == d)[0], SEED) for d in sorted(set(DS))]
RES = []
for i, (tag, tr, te, seed) in enumerate(FOLDS):
    for perc in (100, 99):
        conf, used = boruta_genes(tr, perc, seed)
        for mf in ("sqrt", 0.33):
            m = rf(max_features=mf).fit(X[tr][:, used], y[tr])
            RES.append({"tag": tag, "perc": perc, "mtry": str(mf), "te": te.tolist(),
                        "score": m.predict_proba(X[te][:, used])[:, 1].tolist(),
                        "thr": oob_threshold(m, y[tr]), "genes": [REF["genes"][j] for j in conf]})
    log(f"fold {i + 1}/{len(FOLDS)} ({tag}): Boruta kept {len(RES[-1]['genes'])} (perc 99) / {len(RES[-3]['genes'])} (perc 100)")
json.dump(RES, open(OUT / "boruta_rf_folds.json", "w"))''')

code(r'''# ============================== SUMMARY ==============================
rows = []
for (perc, mf), g in pd.DataFrame(RES).groupby(["perc", "mtry"]):
    per = []
    for r in range(N_REP):
        gr = g[g.tag == f"r{r}"]
        idx = np.concatenate(gr.te.tolist()); s = np.concatenate(gr.score.tolist())
        th = np.concatenate([np.full(len(a), t) for a, t in zip(gr.te, gr.thr)])
        per.append({"auc": roc_auc_score(y[idx], s), "acc_05": accuracy_score(y[idx], (s >= 0.5).astype(int)),
                    "acc_oob": accuracy_score(y[idx], (s > th).astype(int)),
                    "bal_oob": balanced_accuracy_score(y[idx], (s > th).astype(int))})
    lo = g[g.tag.str.startswith("lodo_")]
    li = np.concatenate(lo.te.tolist()); ls = np.concatenate(lo.score.tolist())
    d = pd.DataFrame(per)
    sizes = [len(x) for x in g[g.tag.str.startswith("r")].genes]
    rows.append({"variant": f"Boruta perc {perc} (in fold) | mtry {mf}", "auc": d.auc.mean(), "auc_sd": d.auc.std(ddof=1),
                 "acc_0.5": d.acc_05.mean(), "acc_oob_cut": d.acc_oob.mean(), "acc_oob_sd": d.acc_oob.std(ddof=1),
                 "bal_acc_oob_cut": d.bal_oob.mean(), "lodo_auc": roc_auc_score(y[li], ls),
                 "genes_per_fold": float(np.mean(sizes))})
S = pd.DataFrame(rows).sort_values("auc", ascending=False)
S.to_csv(OUT / "boruta_rf_summary.csv", index=False)
print(S.round(3).to_string(index=False))
log("done")''')

def to_source(t):
    ls = t.split("\n")
    return [l + "\n" for l in ls[:-1]] + ([ls[-1]] if ls[-1] else [])
nb = {"cells": [{"cell_type": k, "id": f"rfb{i:03d}", "metadata": {}, "source": to_source(s),
                 **({"execution_count": None, "outputs": []} if k == "code" else {})}
                for i, (k, s) in enumerate(CELLS)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12.13"}},
      "nbformat": 4, "nbformat_minor": 5}
(HERE / "PD_LCM_rf_boruta.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote PD_LCM_rf_boruta.ipynb", len(CELLS), "cells")
