"""Builds PD_LCM_rf_confirm.ipynb: the step-9b sweep and the step-10 check in one Kaggle run.

The winner and the candidate set are fixed by rule before anything is seen:
winner = highest cross-validated AUC in the sweep, candidates = the top six. The
nested check then pays for that choice. Code is taken verbatim from the local
scripts; only data loading (reordered to the local person/gene order and checked)
and the output folder differ.
"""
import json, pathlib, textwrap
HERE = pathlib.Path(__file__).parent
REF = json.loads((HERE / "data/mdata_reference.json").read_text())
S9 = (HERE / "step9b_rf_pca.py").read_text()
S10 = (HERE / "step10_confirm_rf.py").read_text()

defs = S9[S9.index("# ---------------------------------------------------------------- in-fold gene scores"):S9.index('if __name__ == "__main__":')]
defs = defs.replace("n_estimators=2000", "n_estimators=1000")
sweep = textwrap.dedent(S9[S9.index('if __name__ == "__main__":') + len('if __name__ == "__main__":\n'):])
sweep = sweep.replace('names = [n for n in V if len(sys.argv) < 2 or any(s in n for s in sys.argv[1:])]', "names = list(V)")
sweep = sweep.replace('tag = "_".join(sys.argv[1:]).replace(" ", "") or "all"', 'tag = "kaggle"')
check = S10[S10.index("N_JOBS, N_PERM = 4, 200"):]
check = check.replace("N_JOBS, N_PERM = 4, 200", "N_PERM = 200")
check = check.replace('''SPEC = json.loads((OUT / "S10_candidates.json").read_text())
CAND = {k: (v[0], tuple(v[1]) if isinstance(v[1], list) else v[1], v[2]) for k, v in SPEC["candidates"].items()}
CHOSEN = SPEC["chosen"]
''', "")

CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))
md("""# Random Forest on the merged LCM cohort: sweep, then a strict check of the winner

63 people from four laser-capture studies, one profile per person. Part 1 scores
50 Random Forest variants on the same 5 x 5-fold split, with every gene filter and
PCA fitted inside the training fold. Part 2 re-tests the winner on fresh folds, with
the choice among the top six made inside each fold, 200 label permutations, and
leave one dataset out.""")
code(r'''import time, json, sys, glob, os, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from scipy.stats import rankdata
warnings.filterwarnings("ignore")
OUT = Path("/kaggle/working"); SEED, N_REP = 42, 5
N_JOBS = os.cpu_count(); print("CPUs:", N_JOBS)
t0 = time.time()
def log(m): print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)''')
code("REF = json.loads(r'''" + json.dumps(REF) + "''')")
code(r'''hit = sorted(glob.glob("/kaggle/input/**/18_hvg_expression.npz", recursive=True), key=len)[0]
z = np.load(hit, allow_pickle=True)
Xk = pd.DataFrame(z["X"].astype(float), index=z["person"].astype(str), columns=[str(g) for g in z["genes"]])
X = Xk.loc[REF["person"], REF["genes"]].to_numpy()
y = np.array(REF["y"]); DS = np.array(REF["ds"])
yk = pd.Series(z["y"].astype(int), index=z["person"].astype(str)).loc[REF["person"]].to_numpy()
diff = np.abs(X.sum(1) - np.array(REF["row_sum"])).max()
print("SAME MATRIX AS LOCAL" if diff < 0.05 and (yk == y).all() else f"WARNING: matrix differs (max row diff {diff:.4f})")
STRATA = np.array([f"{d}_{v}" for d, v in zip(DS, y)])
XR = np.apply_along_axis(rankdata, 1, X) / X.shape[1]''')
code(defs)
md("## Part 1 - the sweep")
code(sweep)
md("## Part 2 - strict check of the winner (rule fixed in advance: top AUC wins, top six are the candidates)")
code(r'''CHOSEN = R.loc[0, "variant"]
CAND = {n: V[n] for n in R.variant.head(6)}
print("chosen:", CHOSEN); print("candidates:", list(CAND))''')
code(check)

def to_source(t):
    ls = t.split("\n")
    return [l + "\n" for l in ls[:-1]] + ([ls[-1]] if ls[-1] else [])
nb = {"cells": [{"cell_type": k, "id": f"rfc{i:03d}", "metadata": {}, "source": to_source(s),
                 **({"execution_count": None, "outputs": []} if k == "code" else {})}
                for i, (k, s) in enumerate(CELLS)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12.13"}},
      "nbformat": 4, "nbformat_minor": 5}
(HERE / "PD_LCM_rf_confirm.ipynb").write_text(json.dumps(nb, indent=1))
import ast
for k, s in CELLS:
    if k == "code":
        ast.parse(s)
print("wrote PD_LCM_rf_confirm.ipynb", len(CELLS), "cells; syntax ok")
