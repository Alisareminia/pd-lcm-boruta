"""Builds PD_LCM_accuracy.ipynb - the accuracy-ceiling notebook for Kaggle.

Question it answers: can a model reach 80% accuracy on the merged 63-person LCM
cohort without cheating, and if not, what would it take. Runs on the merged
pipeline kernel's output, uses the GPU for the boosted-tree candidates.
"""

CELLS = []
def md(src): CELLS.append(("markdown", src))
def code(src): CELLS.append(("code", src))


md(r"""# How accurate can PD vs control get on 63 laser-captured people?

Companion to **`pd-lcm-merged-pipeline`**, which harmonises four laser-capture
datasets into one row per person. This notebook asks one question: **how high can
accuracy go without cheating, and what would 80% require?**

Three things are kept strictly apart, because mixing them is what produces
published accuracies that never reproduce:

1. **Choosing** a model family, its hyperparameters and its decision threshold.
2. **Measuring** how well the chosen thing works on people it has never seen.
3. **Choosing among measurements** - picking the best of many models is itself a
   fitted step and needs its own outer loop.

| Section | What it does |
|---|---|
| 1 | Loads the harmonised matrix (63 people, 5,622 genes) and builds pathway and cell-type feature sets |
| 2 | Nineteen model families (twenty with a GPU), hyperparameters **and** threshold tuned inside each training fold |
| 3 | Learning curve: accuracy against training size, and the size 80% would need |
| 4 | Bulk substantia nigra (GSE7621), where 80% is reachable, and why |
| 5 | The same data scored the three common leaky ways, to show where 80% comes from |

**GPU note.** scikit-learn runs on the CPU, so the GPU only accelerates the
XGBoost candidate. It is the boosting model in the line-up for that reason; the
run as a whole is CPU-bound and takes roughly half an hour.""")

md("""## 0. Setup""")

code(r'''# ============================== CONFIG ==============================
SEED       = 42
OUTER_REP  = 3        # outer repeats x 5 folds = 15 honest evaluations per family
N_PERM     = 200      # label permutations for the winner
CURVE_REPS = 30       # resamples per training size in the learning curve
TARGET     = 0.80     # the accuracy being asked about

F_GSE7621  = "GSE7621_series_matrix.txt"
GMT_URL    = "https://biit.cs.ut.ee/gprofiler/static/gprofiler_full_hsapiens.ENSG.gmt"
SET_MIN, SET_MAX = 15, 200
print("target accuracy:", TARGET)''')

code(r'''# ============================== IMPORTS ==============================
import os, io, gzip, json, glob, time, warnings, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
try:
    get_ipython(); IN_NOTEBOOK = True
except NameError:
    IN_NOTEBOOK = False; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, StackingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, balanced_accuracy_score,
                             recall_score, f1_score, matthews_corrcoef, roc_curve)
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    GPU = False
    try:                                   # does this machine actually have a GPU?
        xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=2).fit(
            np.random.rand(8, 3), np.array([0, 1] * 4))
        GPU = True
    except Exception as exc:
        print("XGBoost present but no usable GPU:", type(exc).__name__)
    HAS_XGB = True
except ImportError:
    HAS_XGB, GPU = False, False
print(f"XGBoost available: {HAS_XGB} | GPU in use: {GPU}")

ON_KAGGLE = Path("/kaggle/working").exists()
OUT = Path("/kaggle/working/outputs") if ON_KAGGLE else Path("./outputs")
FIG = OUT / "figures"
for d in (OUT, FIG):
    d.mkdir(parents=True, exist_ok=True)
N_JOBS = max(1, (os.cpu_count() or 2))
print("output:", OUT.resolve(), "| cores:", N_JOBS)

# ---- a quiet, print-first look, same palette as the manuscript figures ----
INK, MUTED, FAINT, BORDER = "#1A1A1A", "#5F6469", "#9AA0A6", "#C9CFD5"
NAVY, STEEL, LIGHTBL = "#26405C", "#4C6E9C", "#A8BED6"
DEEP_RED, CORAL, FOREST, AMBER, PLUM = "#A64236", "#C0785F", "#5F7F63", "#B08442", "#6E5F7E"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 10.5, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.linewidth": 0.9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#3F3F3F", "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})
FIGURES = []
def dashgrid(ax, axis="both"):
    ax.set_axisbelow(True); ax.grid(True, axis=axis, ls="-", lw=0.5, color="#E4E7EA")
def rule(fig, x0, x1, yv, lw=0.8, color=BORDER):
    fig.add_artist(Line2D([x0, x1], [yv, yv], transform=fig.transFigure, color=color, lw=lw, zorder=0))
def head(fig, x0, x1, letter, ttl, sub, yv=0.838):
    rule(fig, x0, x1, yv)
    fig.text(x0, yv + 0.040, f"$\\bf{{{letter}}}$   {ttl}", ha="left", va="center", fontsize=11.4, color=INK)
    fig.text(x0, yv + 0.015, sub, ha="left", va="center", fontsize=9.4, color=MUTED)
def finish(fig, name, caption):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    FIGURES.append({"figure": name, "caption": caption})
    print(f"  [fig] {name}")
    plt.show() if IN_NOTEBOOK else plt.close(fig)
WRITTEN = []
def save_csv(df, name, desc, preview=None):
    df.to_csv(OUT / name, index=False)
    WRITTEN.append({"file": name, "rows": len(df), "description": desc})
    print(f"  [csv] {name} ({len(df)} rows)")
    if preview and IN_NOTEBOOK:
        display(df.head(preview))
t0 = time.time()
def log(m): print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)''')

md(r"""## 1. The cohort, and three ways to describe each person

The merged matrix comes from the pipeline kernel: 63 people, 5,622 genes measured
on every platform and expressed in every dataset, each gene z-scored **within its
own dataset without using the labels**.

Three feature sets are built from it, all label-free:

- **genes** - all 5,622.
- **pathways** - the mean z-score of each GO, Reactome, KEGG or WikiPathways gene
  set holding 15 to 200 of those genes. Averaging cancels some noise.
- **cell types** - mean z-score of marker genes for dopamine neurons, neurons in
  general, astrocytes and oligodendrocytes. If the classifier were only reading
  glial contamination, these four numbers would be enough.""")

code(r'''# ============================== LOAD ==============================
def find_input(pattern):
    for root in ("/kaggle/input", ".", "..", "../.."):
        if Path(root).exists():
            hits = glob.glob(f"{root}/**/{pattern}", recursive=True)
            if hits:
                return sorted(hits, key=len)[0]
    raise FileNotFoundError(f"{pattern} not found. Attach the notebook "
                            "alisaremi/pd-lcm-merged-pipeline and the dataset "
                            "alisaremi/externalvalidation2.")

z = np.load(find_input("18_hvg_expression.npz"), allow_pickle=True)
XG = z["X"].astype(float); y = z["y"].astype(int)
DS = z["dataset"].astype(str); PERSON = z["person"].astype(str)
GENES = [str(g) for g in z["genes"]]
STRATA = np.array([f"{d}_{v}" for d, v in zip(DS, y)])
gi = {g: i for i, g in enumerate(GENES)}
MAJORITY = float(max(y.mean(), 1 - y.mean()))
print(f"{len(y)} people ({int((y == 0).sum())} control, {int(y.sum())} PD), {XG.shape[1]:,} genes")
print(pd.Series(DS).value_counts().rename("people").to_frame().T.to_string())
print(f"majority-class accuracy (call everyone PD): {MAJORITY:.3f}")''')

code(r'''# ============================== PATHWAY AND CELL-TYPE FEATURES ==============================
gmt_path = Path("/tmp/gprofiler_hsapiens.ENSG.gmt")
if not gmt_path.exists():
    for k in range(4):
        try:
            urllib.request.urlretrieve(GMT_URL, gmt_path); break
        except Exception as exc:
            print("  GMT download retry", k + 1, type(exc).__name__); time.sleep(8)
sets, seen = {}, set()
if gmt_path.exists():
    for line in open(gmt_path, errors="replace"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 3:
            continue
        if not p[0].startswith(("GO:", "REAC:", "KEGG:", "WP:")):
            continue                      # skip TF, miRNA, HPA, CORUM, HP sources
        idx = tuple(sorted(gi[g] for g in p[2:] if g in gi))
        if SET_MIN <= len(idx) <= SET_MAX and idx not in seen:
            seen.add(idx); sets[f"{p[0]}|{p[1]}"] = list(idx)
SET_NAMES = list(sets)
XS = (np.column_stack([XG[:, i].mean(1) for i in sets.values()])
      if sets else np.zeros((len(y), 0)))
if XS.shape[1]:
    XS = (XS - XS.mean(0)) / XS.std(0).clip(1e-6)

SYMBOL = {}
try:
    smap = pd.read_csv(find_input("15_gene_symbol_map.csv"))
    SYMBOL = dict(zip(smap["gene"], smap["symbol"]))
except Exception as exc:
    print("symbol map not found:", exc)
S2E = {v: k for k, v in SYMBOL.items() if isinstance(v, str)}
CELLS = {
    "dopamine neuron": ["TH", "SLC6A3", "SLC18A2", "DDC", "KCNJ6", "ALDH1A1", "NR4A2", "EN1", "FOXA2", "LMX1B"],
    "pan-neuron":      ["SNAP25", "SYT1", "RBFOX3", "STMN2", "GAP43", "SYN1"],
    "astrocyte":       ["GFAP", "AQP4", "SLC1A2", "SLC1A3", "ALDH1L1", "GJA1"],
    "microglia":       ["CX3CR1", "P2RY12", "AIF1", "CSF1R", "C1QA", "C1QB", "TYROBP"],
    "oligodendrocyte": ["MBP", "MOG", "PLP1", "MOBP", "MAG", "OLIG2"],
}
cell_used = {c: [s for s in m if s in S2E and S2E[s] in gi] for c, m in CELLS.items()}
cell_used = {c: m for c, m in cell_used.items() if len(m) >= 2}
XC = np.column_stack([XG[:, [gi[S2E[s]] for s in m]].mean(1) for m in cell_used.values()])
XA = np.hstack([XG, XS]) if XS.shape[1] else XG
print(f"pathway scores: {XS.shape[1]:,} sets of {SET_MIN}-{SET_MAX} genes")
print("cell-type scores: " + ", ".join(f"{c} ({len(m)} markers)" for c, m in cell_used.items()))''')

md(r"""## 2. Every model family, each tuned honestly

For every family: an outer stratified 5-fold split (repeated 3 times, stratified by
dataset **and** diagnosis) holds out people. Inside the remaining people, a grid
search with its own 5-fold cross-validation picks the hyperparameters, and the
**decision threshold** is picked on inner out-of-fold predictions. Only then is the
held-out fold scored. No number below has seen its own test people.

The threshold matters for accuracy: 0.5 is arbitrary when the classes are nearly
balanced but the scores are not centred.""")

code(r'''# ============================== FAMILIES ==============================
class RankRows(BaseEstimator, TransformerMixin):
    """Within-person quantile rank: removes each person own scale."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        return (np.argsort(np.argsort(X, axis=1), axis=1) / (X.shape[1] - 1)).astype(float)

class PLSda(BaseEstimator):
    def __init__(self, n_components=2): self.n_components = n_components
    def fit(self, X, yf):
        self.pls_ = PLSRegression(n_components=self.n_components).fit(X, yf.astype(float))
        s = self.pls_.predict(X).ravel()
        self.lo_, self.hi_ = s[yf == 0].mean(), s[yf == 1].mean()
        self.classes_ = np.array([0, 1]); return self
    def predict_proba(self, X):
        s = self.pls_.predict(X).ravel()
        p = np.clip((s - self.lo_) / max(self.hi_ - self.lo_, 1e-9), 0, 1)
        return np.column_stack([1 - p, p])

def P(*steps): return Pipeline([(f"s{i}", s) for i, s in enumerate(steps)])
def LRB(**kw):
    kw.setdefault("max_iter", 5000); kw.setdefault("class_weight", "balanced")
    return LogisticRegression(**kw)

FAMILIES = {
    "Logistic L2, all genes": (XG, P(StandardScaler(), LRB()), {"s1__C": [0.003, 0.01, 0.03, 0.1, 1.0, 10.0]}),
    "Logistic L1, all genes": (XG, P(StandardScaler(), LRB(penalty="l1", solver="liblinear")), {"s1__C": [0.03, 0.1, 0.5, 2.0]}),
    "Elastic net (SGD)": (XG, P(StandardScaler(), SGDClassifier(loss="log_loss", penalty="elasticnet",
                          max_iter=3000, tol=1e-3, class_weight="balanced", random_state=SEED)),
                          {"s1__alpha": [1e-4, 1e-3, 1e-2], "s1__l1_ratio": [0.2, 0.8]}),
    "Logistic L2, F-test filter": (XG, P(StandardScaler(), SelectKBest(f_classif), LRB()),
                                   {"s1__k": [100, 500, 2000], "s2__C": [0.01, 0.1, 1.0]}),
    "Logistic L2, genes + pathways": (XA, P(StandardScaler(), LRB()), {"s1__C": [0.003, 0.01, 0.1, 1.0]}),
    "Logistic L2, pathways only": (XS if XS.shape[1] else XG, P(StandardScaler(), LRB()), {"s1__C": [0.003, 0.01, 0.1, 1.0]}),
    "Logistic L2, within-person ranks": (XG, P(RankRows(), StandardScaler(), LRB()), {"s2__C": [0.01, 0.1, 1.0]}),
    "Logistic L2, cell types only": (XC, P(StandardScaler(), LRB()), {"s1__C": [0.1, 1.0, 10.0]}),
    "Ridge classifier": (XG, P(StandardScaler(), RidgeClassifier(class_weight="balanced")),
                         {"s1__alpha": [1.0, 10.0, 100.0, 1000.0]}),
    "Linear SVM": (XG, P(StandardScaler(), SVC(kernel="linear", class_weight="balanced", random_state=SEED)),
                   {"s1__C": [0.0003, 0.001, 0.01, 0.1]}),
    "RBF SVM, F-test filter": (XG, P(StandardScaler(), SelectKBest(f_classif), SVC(class_weight="balanced", random_state=SEED)),
                               {"s1__k": [100, 500], "s2__C": [1.0, 10.0], "s2__gamma": ["scale", 0.001]}),
    "PLS discriminant analysis": (XG, P(StandardScaler(), PLSda()), {"s1__n_components": [1, 2, 3, 5]}),
    "Shrinkage LDA, F-test filter": (XG, P(StandardScaler(), SelectKBest(f_classif), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
                                     {"s1__k": [50, 100, 500]}),
    "PCA + logistic": (XG, P(StandardScaler(), PCA(random_state=SEED), LRB()),
                       {"s1__n_components": [5, 10, 20], "s2__C": [0.1, 1.0]}),
    "Random Forest": (XG, RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced",
                      random_state=SEED, n_jobs=1), {"min_samples_leaf": [1, 3]}),
    "Extra Trees": (XG, ExtraTreesClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced",
                    random_state=SEED, n_jobs=1), {"min_samples_leaf": [1, 3]}),
    "kNN, F-test filter": (XG, P(StandardScaler(), SelectKBest(f_classif), KNeighborsClassifier(weights="distance")),
                           {"s1__k": [50, 200], "s2__n_neighbors": [5, 9, 15]}),
    "Stacked ensemble": (XA, StackingClassifier(
        estimators=[("lr", P(StandardScaler(), LRB(C=0.01))),
                    ("lr_filter", P(StandardScaler(), SelectKBest(f_classif, k=200), LRB(C=0.1))),
                    ("lda", P(StandardScaler(), SelectKBest(f_classif, k=100), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))),
                    ("ridge", P(StandardScaler(), RidgeClassifier(alpha=100.0, class_weight="balanced")))],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"), cv=3, n_jobs=1),
        {"final_estimator__C": [1.0]}),
}
if HAS_XGB:
    # on a GPU this family is cheap, so it gets a grid; on CPU it would dominate
    # the whole run, so it gets a single setting instead
    xgb_kw = dict(n_estimators=300 if GPU else 200, subsample=0.8, colsample_bytree=0.3,
                  eval_metric="logloss", random_state=SEED, n_jobs=1, tree_method="hist")
    if GPU:
        xgb_kw["device"] = "cuda"
    FAMILIES["XGBoost" + (" (GPU)" if GPU else " (CPU)")] = (
        XG, xgb.XGBClassifier(**xgb_kw),
        {"max_depth": [2, 3], "learning_rate": [0.05, 0.1]} if GPU else {"max_depth": [2], "learning_rate": [0.1]})
print(f"{len(FAMILIES)} families")''')

code(r'''# ============================== HONEST EVALUATION ==============================
def scores_of(m, X):
    return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X)

def best_threshold(s, yt):
    """Threshold maximising accuracy, chosen on training-fold predictions only."""
    u = np.unique(np.round(s, 6))
    cuts = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]]) if len(u) > 1 else np.array([0.5])
    return float(cuts[int(np.argmax([accuracy_score(yt, (s > c).astype(int)) for c in cuts]))])

def outer_eval(fam, r, k, tr, te):
    tic = time.time()
    X, est, grid = FAMILIES[fam]
    inner = StratifiedKFold(5, shuffle=True, random_state=7000 + 13 * r + k)
    gs = GridSearchCV(clone(est), grid, scoring="roc_auc", cv=inner, n_jobs=1, refit=True).fit(X[tr], y[tr])
    p_in = np.full(len(tr), np.nan)
    for itr, ite in inner.split(X[tr], STRATA[tr]):
        p_in[ite] = scores_of(clone(gs.best_estimator_).fit(X[tr][itr], y[tr][itr]), X[tr][ite])
    thr = best_threshold(p_in, y[tr])
    s = scores_of(gs.best_estimator_, X[te])
    return {"family": fam, "repeat": r, "fold": k, "threshold": thr, "seconds": time.time() - tic,
            "params": json.dumps({a: str(b) for a, b in gs.best_params_.items()}),
            "idx": te.tolist(), "score": s.tolist()}

OUTER = [(r, k, tr, te) for r in range(OUTER_REP)
         for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=900 + r).split(XG, STRATA))]
log(f"tuned search: {len(FAMILIES)} families x {len(OUTER)} outer folds")
res = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(outer_eval)(f, *o) for f in FAMILIES for o in OUTER)
R = pd.DataFrame(res)
log("tuned search done")

rows, POOLED = [], {}
for fam, g in R.groupby("family"):
    per = []
    for r, gr in g.groupby("repeat"):
        idx = np.concatenate([np.asarray(x) for x in gr["idx"]])
        sc = np.concatenate([np.asarray(x) for x in gr["score"]])
        th = np.concatenate([np.full(len(x), t) for x, t in zip(gr["idx"], gr["threshold"])])
        yh = (sc > th).astype(int)
        per.append({"acc": accuracy_score(y[idx], yh), "bal": balanced_accuracy_score(y[idx], yh),
                    "sens": recall_score(y[idx], yh), "spec": recall_score(1 - y[idx], 1 - yh),
                    "f1": f1_score(y[idx], yh, zero_division=0), "mcc": matthews_corrcoef(y[idx], yh),
                    "auc": roc_auc_score(y[idx], sc)})
        if r == 0:
            POOLED[fam] = (idx, sc, th)
    d = pd.DataFrame(per)
    rows.append({"family": fam, "accuracy": d.acc.mean(), "accuracy_sd": d.acc.std(ddof=1),
                 "bal_accuracy": d.bal.mean(), "sensitivity": d.sens.mean(), "specificity": d.spec.mean(),
                 "f1": d.f1.mean(), "mcc": d.mcc.mean(), "auc": d.auc.mean(), "auc_sd": d.auc.std(ddof=1),
                 "reaches_target": bool(d.acc.mean() >= TARGET), "params_mode": g["params"].mode().iloc[0]})
A = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
A.insert(0, "rank", range(1, len(A) + 1))
save_csv(A, "A1_tuned_family_accuracy.csv", "Every family: accuracy, balanced accuracy and AUC, all tuned inside folds")
save_csv(R.drop(columns=["idx", "score"]), "A2_tuned_folds.csv", "Chosen hyperparameters, threshold and runtime per outer fold")
print("\nslowest families (seconds per outer fold):")
print(R.groupby("family")["seconds"].mean().sort_values(ascending=False).head(5).round(1).to_string())
BEST = A.loc[0, "family"]
print(A[["rank", "family", "accuracy", "accuracy_sd", "bal_accuracy", "auc"]].round(3).to_string(index=False))
log(f"best honest accuracy: {A.loc[0, 'accuracy']:.3f} +/- {A.loc[0, 'accuracy_sd']:.3f} ({BEST}); "
    f"majority baseline {MAJORITY:.3f}; target {TARGET}")''')

code(r'''# ============================== IS THE WINNER ABOVE CHANCE? ==============================
Xb, est_b, grid_b = FAMILIES[BEST]
def tuned_cv_accuracy(yy, seed=SEED):
    st = np.array([f"{d}_{v}" for d, v in zip(DS, yy)])
    acc_idx, acc_sc, acc_th = [], [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xb, st):
        inner = StratifiedKFold(4, shuffle=True, random_state=seed + 1)
        gs = GridSearchCV(clone(est_b), grid_b, scoring="roc_auc", cv=inner, n_jobs=1).fit(Xb[tr], yy[tr])
        p_in = np.full(len(tr), np.nan)
        for itr, ite in inner.split(Xb[tr], st[tr]):
            p_in[ite] = scores_of(clone(gs.best_estimator_).fit(Xb[tr][itr], yy[tr][itr]), Xb[tr][ite])
        acc_idx.append(te); acc_sc.append(scores_of(gs.best_estimator_, Xb[te]))
        acc_th.append(np.full(len(te), best_threshold(p_in, yy[tr])))
    idx = np.concatenate(acc_idx); sc = np.concatenate(acc_sc); th = np.concatenate(acc_th)
    yh = (sc > th).astype(int)
    return accuracy_score(yy[idx], yh), roc_auc_score(yy[idx], sc)

obs_acc, obs_auc = tuned_cv_accuracy(y)
def perm_run(b):
    r = np.random.default_rng(20_000 + b); yp = y.copy()
    for d in np.unique(DS):
        m = np.where(DS == d)[0]; yp[m] = yp[r.permutation(m)]
    return tuned_cv_accuracy(yp, seed=SEED)
null = np.array(Parallel(n_jobs=N_JOBS)(delayed(perm_run)(b) for b in range(N_PERM)))
p_acc = (np.sum(null[:, 0] >= obs_acc) + 1) / (N_PERM + 1)
p_auc = (np.sum(null[:, 1] >= obs_auc) + 1) / (N_PERM + 1)
save_csv(pd.DataFrame({"perm_accuracy": null[:, 0], "perm_auc": null[:, 1]}),
         "A3_permutation_null.csv", f"{N_PERM} label permutations of the winning family, labels shuffled within dataset")
log(f"{BEST}: accuracy {obs_acc:.3f} (null mean {null[:, 0].mean():.3f}, 95th pct {np.percentile(null[:, 0], 95):.3f}, "
    f"p = {p_acc:.4f}) | AUC {obs_auc:.3f} (p = {p_auc:.4f})")''')

md(r"""## 3. What would 80% take?

The honest ceiling above is a property of this sample size as much as of the
biology. Accuracy is therefore measured again with fewer training people, and a
power-law curve, `accuracy(n) = a - b n^-c`, is fitted to those points. Solving it
for 0.80 gives an order-of-magnitude answer to "how many people".

Extrapolation is not evidence, so the fitted ceiling `a` is reported next to it:
if `a` sits below 0.80, no sample size reaches the target with these features.""")

code(r'''# ============================== LEARNING CURVE ==============================
gs_full = GridSearchCV(clone(est_b), grid_b, scoring="roc_auc",
                       cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=N_JOBS).fit(Xb, y)
fixed = clone(gs_full.best_estimator_)
print("hyperparameters used for the curve:", gs_full.best_params_)

def curve_point(n, b):
    tr, te = train_test_split(np.arange(len(y)), train_size=n, random_state=3000 + b, stratify=STRATA)
    m = clone(fixed).fit(Xb[tr], y[tr])
    thr = best_threshold(scores_of(m, Xb[tr]), y[tr])
    s = scores_of(m, Xb[te])
    return {"n_train": n, "rep": b, "acc": accuracy_score(y[te], (s > thr).astype(int)),
            "auc": roc_auc_score(y[te], s) if len(set(y[te])) == 2 else np.nan}
SIZES = [20, 25, 30, 35, 40, 45, 50]
lc = pd.DataFrame(Parallel(n_jobs=N_JOBS)(delayed(curve_point)(n, b) for n in SIZES for b in range(CURVE_REPS)))
LC = lc.groupby("n_train").agg(acc=("acc", "mean"), acc_sd=("acc", "std"),
                               auc=("auc", "mean"), auc_sd=("auc", "std")).reset_index()
save_csv(LC, "B1_learning_curve.csv", "Accuracy and AUC against training-set size")
print(LC.round(3).to_string(index=False))

def powerlaw(n, a, b_, c): return a - b_ * np.power(n, -c)
NEED, CEIL = np.nan, np.nan
try:
    popt, _ = curve_fit(powerlaw, LC.n_train, LC.acc, p0=[0.8, 1.0, 0.5], maxfev=40000,
                        bounds=([0.5, 0, 0.05], [1.0, 50, 3]))
    CEIL = float(popt[0])
    NEED = float((popt[1] / (CEIL - TARGET)) ** (1 / popt[2])) if CEIL > TARGET else np.inf
    log(f"fitted ceiling {CEIL:.3f}; people needed for {TARGET:.0%}: "
        + ("out of reach with these features" if not np.isfinite(NEED) else f"about {NEED:.0f}"))
except Exception as exc:
    log(f"curve fit failed: {type(exc).__name__}")''')

md(r"""## 4. Where 80% is reachable, and why

GSE7621 is **bulk** substantia nigra: dissected tissue, not captured neurons. In
PD that tissue has lost most of its dopamine neurons, so a classifier can succeed
by counting neurons rather than by reading a state inside them. The same tuned
logistic model is cross-validated there, and then the eight dopamine-neuron marker
genes are used alone for comparison.""")

code(r'''# ============================== BULK NIGRA ==============================
gse = find_input(F_GSE7621)
lines = open(gse, encoding="utf-8", errors="replace").read().split("\n")
titles = [t.strip('"') for t in next(l for l in lines if l.startswith("!Sample_title")).split("\t")[1:]]
ye = np.array([0 if "normal" in t.lower() else 1 for t in titles])
b0 = next(i for i, l in enumerate(lines) if "!series_matrix_table_begin" in l)
b1 = next(i for i, l in enumerate(lines) if "!series_matrix_table_end" in l)
M = pd.read_csv(io.StringIO("\n".join(lines[b0 + 1:b1])), sep="\t", index_col=0).apply(pd.to_numeric, errors="coerce")
LOGE = np.log2(M.clip(lower=1))
print(f"GSE7621: {M.shape[1]} samples x {M.shape[0]:,} probes; control {int((ye == 0).sum())}, PD {int(ye.sum())}; "
      f"majority {max(ye.mean(), 1 - ye.mean()):.2f}")

def gconvert(ids, target):
    out = {}
    for s in range(0, len(ids), 3000):
        body = {"organism": "hsapiens", "target": target, "query": list(ids[s:s + 3000])}
        for k in range(5):
            try:
                req = urllib.request.Request("https://biit.cs.ut.ee/gprofiler/api/convert/convert/",
                                             data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=300) as fh:
                    for rec in json.loads(fh.read())["result"]:
                        c = rec.get("converted")
                        if c and c not in ("None", "N/A"):
                            out.setdefault(rec["incoming"], set()).add(c)
                break
            except Exception as exc:
                print("  g:Convert retry", k + 1, type(exc).__name__); time.sleep(8)
    return out

pm = LOGE.mean(1)
conv = gconvert(GENES, "AFFY_HG_U133_PLUS_2")
best_probe = {}
for g, ps in conv.items():
    ps = [p for p in ps if p in LOGE.index]
    if ps:
        best_probe[g] = max(ps, key=lambda p: pm[p])
EB = pd.DataFrame({g: LOGE.loc[p].to_numpy(float) for g, p in best_probe.items()})
EBz = ((EB - EB.mean(0)) / EB.std(0, ddof=1).clip(0.05)).to_numpy()
print(f"{EB.shape[1]:,} of {len(GENES):,} harmonised genes map to a probe")

def bulk_cv(seed, Xb_, yb_):
    idx, sc, th = [], [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xb_, yb_):
        inner = StratifiedKFold(4, shuffle=True, random_state=seed + 1)
        gs = GridSearchCV(P(StandardScaler(), LRB()), {"s1__C": [0.003, 0.01, 0.1, 1.0]},
                          scoring="roc_auc", cv=inner, n_jobs=1).fit(Xb_[tr], yb_[tr])
        p_in = np.full(len(tr), np.nan)
        for itr, ite in inner.split(Xb_[tr], yb_[tr]):
            p_in[ite] = scores_of(clone(gs.best_estimator_).fit(Xb_[tr][itr], yb_[tr][itr]), Xb_[tr][ite])
        idx.append(te); sc.append(scores_of(gs.best_estimator_, Xb_[te])); th.append(np.full(len(te), best_threshold(p_in, yb_[tr])))
    i = np.concatenate(idx); s = np.concatenate(sc); t = np.concatenate(th)
    yh = (s > t).astype(int)
    return accuracy_score(yb_[i], yh), balanced_accuracy_score(yb_[i], yh), roc_auc_score(yb_[i], s)

bulk = np.array(Parallel(n_jobs=N_JOBS)(delayed(bulk_cv)(s, EBz, ye) for s in range(10)))
mk = [S2E[s] for s in CELLS["dopamine neuron"] if s in S2E and S2E[s] in EB.columns]
NEURON = EBz[:, [EB.columns.get_loc(g) for g in mk]].mean(1)
neuron_auc = roc_auc_score(1 - ye, NEURON)
save_csv(pd.DataFrame(bulk, columns=["accuracy", "bal_accuracy", "auc"]),
         "C1_bulk_nigra_cv.csv", "Tuned logistic model cross-validated on bulk GSE7621")
log(f"bulk nigra: accuracy {bulk[:, 0].mean():.3f} +/- {bulk[:, 0].std(ddof=1):.3f} | "
    f"balanced {bulk[:, 1].mean():.3f} | AUC {bulk[:, 2].mean():.3f}")
log(f"{len(mk)} dopamine-neuron markers alone separate the same samples with AUC {neuron_auc:.3f}")''')

md(r"""## 5. The three ways 80% appears in papers

Each variant below is run on exactly the same 63 people as section 2. Only the
bookkeeping changes. They are reported so the gap is visible, not as results.

| Variant | What leaks |
|---|---|
| Genes filtered on everyone, then cross-validated | the test people helped choose the genes |
| Threshold chosen on the test fold | the cut-off is fitted to the answer |
| Best of many random five-gene panels, scored on the test fold | the winner is chosen by the test set |
""")

code(r'''# ============================== HONEST VS LEAKY ==============================
def honest_reference():
    return float(A.loc[0, "accuracy"])

def leak_filter_on_all(k=200):
    """Rank genes on all 63 people once, then cross-validate the model on that subset."""
    keep = np.argsort(f_classif(XG, y)[0])[::-1][:k]
    accs = []
    for r in range(3):
        idx, sc, th = [], [], []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=900 + r).split(XG, STRATA):
            m = P(StandardScaler(), LRB(C=0.1)).fit(XG[tr][:, keep], y[tr])
            idx.append(te); sc.append(scores_of(m, XG[te][:, keep]))
            th.append(np.full(len(te), best_threshold(scores_of(m, XG[tr][:, keep]), y[tr])))
        i, s, t = np.concatenate(idx), np.concatenate(sc), np.concatenate(th)
        accs.append(accuracy_score(y[i], (s > t).astype(int)))
    return float(np.mean(accs))

def leak_threshold_on_test():
    """Honest model, but the cut-off is chosen on the test fold itself."""
    accs = []
    for r in range(3):
        acc_fold = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=900 + r).split(XG, STRATA):
            m = P(StandardScaler(), LRB(C=1.0)).fit(XG[tr], y[tr])
            s = scores_of(m, XG[te])
            acc_fold.append(accuracy_score(y[te], (s > best_threshold(s, y[te])).astype(int)))
        accs.append(np.mean(acc_fold))
    return float(np.mean(accs))

def leak_best_panel_on_test(n_panels=500, size=5):
    """Many small panels, and the one that scores best on the test fold is reported."""
    rng = np.random.default_rng(SEED)
    panels = [rng.choice(XG.shape[1], size, replace=False) for _ in range(n_panels)]
    accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=900).split(XG, STRATA):
        best = 0.0
        for cols in panels:
            m = P(StandardScaler(), LRB(C=1.0)).fit(XG[tr][:, cols], y[tr])
            s = scores_of(m, XG[te][:, cols])
            best = max(best, accuracy_score(y[te], (s > best_threshold(s, y[te])).astype(int)))
        accs.append(best)
    return float(np.mean(accs))

comp = [("Honest: everything tuned inside the training folds", honest_reference(), "none"),
        ("Genes filtered on all 63 people, then cross-validated", leak_filter_on_all(), "feature selection"),
        ("Threshold chosen on the test fold", leak_threshold_on_test(), "threshold"),
        ("Best of 500 random five-gene panels, chosen on the test fold", leak_best_panel_on_test(), "model choice")]
CMP = pd.DataFrame(comp, columns=["variant", "accuracy", "what_leaks"])
CMP["above_target"] = CMP["accuracy"] >= TARGET
save_csv(CMP, "D1_honest_vs_leaky.csv", "The same cohort scored honestly and the three common leaky ways")
print(CMP.round(3).to_string(index=False))''')

md("""## 6. Figures""")

code(r'''# ============================== FIGURE: ACCURACY CEILING ==============================
fig = plt.figure(figsize=(15.0, 8.6))
nR = len(A)
ax = fig.add_axes([0.030, 0.085, 0.430, 0.700]); ax.axis("off")
ax.set_xlim(0.30, 1.16); ax.set_ylim(nR - 0.4, -1.6)
for i, r in A.iterrows():
    best = i == 0
    ax.add_patch(Rectangle((0.30, i - 0.46), 0.86, 0.92, zorder=0, edgecolor="none",
                           facecolor="#FBEEEA" if best else ("#F5F7F9" if i % 2 == 0 else "white")))
    a, sd = float(r["accuracy"]), float(r["accuracy_sd"])
    ax.text(0.325, i, r["family"], ha="left", va="center", fontsize=9.8, color=INK,
            fontweight="bold" if best else "regular")
    ax.plot([a - sd, a + sd], [i, i], color=STEEL, lw=2.0, solid_capstyle="round", zorder=3)
    ax.plot([a], [i], marker="D", ms=7.2, color=DEEP_RED if best else NAVY, mec="white", mew=1.1, ls="none", zorder=4)
    ax.text(1.145, i, f"{a:.3f}", ha="right", va="center", fontsize=9.5,
            color=INK if best else MUTED, fontweight="bold" if best else "regular", zorder=4)
ax.plot([MAJORITY] * 2, [-0.6, nR - 0.5], ls=(0, (3, 3)), lw=1.0, color="#8E959B", zorder=2)
ax.text(MAJORITY, -0.8, f"call everyone PD  {MAJORITY:.2f}", ha="center", va="bottom", fontsize=9.0, color=MUTED)
ax.plot([TARGET] * 2, [-0.6, nR - 0.5], lw=1.3, color=DEEP_RED, alpha=0.75, zorder=2)
ax.text(TARGET, -0.8, f"{TARGET:.0%}", ha="center", va="bottom", fontsize=9.8, color=DEEP_RED, fontweight="bold")
for tick in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    ax.plot([tick, tick], [nR - 0.58, nR - 0.45], color="#8E959B", lw=0.8)
    ax.text(tick, nR - 0.32, f"{tick:g}", ha="center", va="top", fontsize=8.4, color=MUTED)

axB = fig.add_axes([0.545, 0.355, 0.435, 0.430])
n = LC["n_train"].to_numpy(float); acc = LC["acc"].to_numpy(float); sd = LC["acc_sd"].to_numpy(float)
axB.fill_between(n, acc - sd, acc + sd, color=LIGHTBL, alpha=0.45, lw=0)
axB.plot(n, acc, color=STEEL, lw=2.2, zorder=4)
axB.plot(n, acc, marker="o", ms=5.5, color=STEEL, mec="white", mew=1.0, ls="none", zorder=5)
if np.isfinite(CEIL):
    xf = np.linspace(n.min(), max(260.0, (NEED if np.isfinite(NEED) else 260.0) * 1.1), 400)
    axB.plot(xf, powerlaw(xf, *popt), ls=(0, (4, 3)), lw=1.4, color=NAVY, alpha=0.85, zorder=3)
    axB.axhline(CEIL, lw=0.9, color=FOREST, ls=(0, (2, 3)))
    axB.text(xf[-1], CEIL + 0.005, f"fitted ceiling {CEIL:.2f}", ha="right", va="bottom", fontsize=9.2, color=FOREST)
axB.axhline(TARGET, lw=1.2, color=DEEP_RED, alpha=0.8)
axB.text(n.min(), TARGET + 0.006, f"{TARGET:.0%} accuracy", ha="left", va="bottom", fontsize=9.6,
         color=DEEP_RED, fontweight="bold")
axB.axhline(MAJORITY, lw=0.9, color="#8E959B", ls=(0, (3, 3)))
axB.text(n.min(), MAJORITY - 0.012, "call everyone PD", ha="left", va="top", fontsize=9.0, color=MUTED)
if np.isfinite(NEED):
    axB.plot([NEED], [TARGET], marker="v", ms=9, color=DEEP_RED, mec="white", mew=1.0, ls="none", zorder=6)
    axB.annotate(f"about {NEED:.0f} people", xy=(NEED, TARGET), xytext=(NEED, TARGET - 0.075), ha="center", va="top",
                 fontsize=9.6, color=DEEP_RED, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=DEEP_RED, lw=0.9))
axB.set_xlabel("People used for training"); axB.set_ylabel("Accuracy on the people left out")
axB.set_ylim(0.40, 0.95); dashgrid(axB)

axC = fig.add_axes([0.545, 0.085, 0.435, 0.150]); axC.axis("off")
axC.set_xlim(0, 1); axC.set_ylim(0, 1)
for j, (lab, v, col) in enumerate([("laser-captured neurons, 63 people", float(A.loc[0, "accuracy"]), STEEL),
                                   ("bulk substantia nigra, 25 samples", float(bulk[:, 0].mean()), DEEP_RED)]):
    yb = 0.70 - j * 0.42
    axC.text(0.0, yb, lab, ha="left", va="center", fontsize=9.8, color=INK)
    axC.add_patch(Rectangle((0.46, yb - 0.10), 0.40 * (v - 0.3) / 0.7, 0.20, facecolor=col, alpha=0.85, edgecolor="none"))
    axC.text(0.90, yb, f"{v:.2f}", ha="left", va="center", fontsize=10.0, color=col, fontweight="bold")
axC.plot([0.46 + 0.40 * (TARGET - 0.3) / 0.7] * 2, [0.05, 0.92], color=DEEP_RED, lw=1.2, alpha=0.7)

head(fig, 0.030, 0.470, "a", f"{nR} tuned model families",
     "hyperparameters and cut-off chosen by inner cross-validation, scored once on the held-out fold")
head(fig, 0.545, 0.980, "b", f"What would reach {TARGET:.0%}",
     "accuracy against training size, with a power-law fit")
fig.text(0.545, 0.275, "$\\bf{c}$   Where the target is already reachable", ha="left", va="center",
         fontsize=10.6, color=INK)
fig.text(0.545, 0.252, "bulk tissue separates the groups largely by neuron loss: the dopamine-neuron markers "
         f"alone reach AUC {neuron_auc:.2f}", ha="left", va="center", fontsize=9.2, color=MUTED)
finish(fig, "Accuracy_ceiling", "Honest accuracy of every tuned family, the training size the target needs, "
                                "and the bulk-tissue comparison")''')

code(r'''# ============================== FIGURE: HONEST VS LEAKY ==============================
fig = plt.figure(figsize=(12.6, 4.9))
ax = fig.add_axes([0.035, 0.150, 0.930, 0.560]); ax.axis("off")
ax.set_xlim(0.35, 1.02); ax.set_ylim(len(CMP) - 0.4, -0.8)
for i, r in CMP.iterrows():
    col = STEEL if i == 0 else DEEP_RED
    v = float(r["accuracy"])
    ax.add_patch(Rectangle((0.35, i - 0.32), v - 0.35, 0.64, facecolor=col, alpha=0.20 if i == 0 else 0.75,
                           edgecolor="none", zorder=2))
    ax.text(0.36, i - 0.42, r["variant"], ha="left", va="bottom", fontsize=10.2, color=INK,
            fontweight="bold" if i == 0 else "regular")
    ax.text(v + 0.006, i, f"{v:.2f}", ha="left", va="center", fontsize=10.6, color=col, fontweight="bold", zorder=4)
    if r["what_leaks"] != "none":
        ax.text(0.995, i - 0.42, f"leaks: {r['what_leaks']}", ha="right", va="bottom", fontsize=9.0, color=MUTED)
ax.plot([TARGET] * 2, [-0.55, len(CMP) - 0.5], lw=1.3, color=DEEP_RED, alpha=0.75, zorder=3)
ax.text(TARGET, -0.62, f"{TARGET:.0%}", ha="center", va="bottom", fontsize=9.8, color=DEEP_RED, fontweight="bold")
ax.plot([MAJORITY] * 2, [-0.55, len(CMP) - 0.5], ls=(0, (3, 3)), lw=1.0, color="#8E959B", zorder=3)
ax.text(MAJORITY, -0.62, "chance", ha="center", va="bottom", fontsize=9.0, color=MUTED)
for tick in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    ax.plot([tick, tick], [len(CMP) - 0.5, len(CMP) - 0.4], color="#8E959B", lw=0.8)
    ax.text(tick, len(CMP) - 0.28, f"{tick:g}", ha="center", va="top", fontsize=8.6, color=MUTED)
rule(fig, 0.035, 0.965, 0.790)
fig.text(0.035, 0.885, "The same 63 people, four sets of bookkeeping", ha="left", va="center", fontsize=11.8, color=INK)
fig.text(0.035, 0.840, "each leaky variant uses the test people for one decision it should not: which genes, "
         "which cut-off, or which panel", ha="left", va="center", fontsize=9.6, color=MUTED)
fig.text(0.035, 0.045, "None of the lower three is a result. They are printed because this is how a published "
         "80% is usually produced on a cohort this size.", ha="left", va="center", fontsize=9.4, color=MUTED)
finish(fig, "Honest_vs_leaky", "The same cohort scored honestly and the three common leaky ways")''')

md("""## 7. Summary""")

code(r'''# ============================== SUMMARY ==============================
summary = {
    "n_people": int(len(y)), "n_genes": int(XG.shape[1]), "majority_accuracy": MAJORITY, "target": TARGET,
    "best_family": BEST, "best_accuracy": float(A.loc[0, "accuracy"]), "best_accuracy_sd": float(A.loc[0, "accuracy_sd"]),
    "best_balanced_accuracy": float(A.loc[0, "bal_accuracy"]), "best_auc": float(A.loc[0, "auc"]),
    "families_reaching_target": int(A["reaches_target"].sum()),
    "winner_permutation_p_accuracy": float(p_acc), "winner_permutation_p_auc": float(p_auc),
    "fitted_ceiling": None if not np.isfinite(CEIL) else float(CEIL),
    "people_needed_for_target": None if not np.isfinite(NEED) else float(NEED),
    "bulk_accuracy": float(bulk[:, 0].mean()), "bulk_auc": float(bulk[:, 2].mean()),
    "bulk_neuron_marker_auc": float(neuron_auc),
    "leaky": {r["variant"]: float(r["accuracy"]) for _, r in CMP.iterrows()},
    "gpu_used": bool(GPU),
}
json.dump(summary, open(OUT / "Z_summary.json", "w"), indent=1)
man = pd.DataFrame(WRITTEN); man.insert(0, "n", range(1, len(man) + 1))
man.to_csv(OUT / "00_manifest.csv", index=False)
pd.DataFrame(FIGURES).to_csv(OUT / "00_figures.csv", index=False)

print("=" * 66); print("ANSWER"); print("=" * 66)
print(f"  people                      : {len(y)} ({int((y == 0).sum())} control / {int(y.sum())} PD)")
print(f"  call-everyone-PD accuracy   : {MAJORITY:.3f}")
print(f"  best honest accuracy        : {A.loc[0, 'accuracy']:.3f} +/- {A.loc[0, 'accuracy_sd']:.3f}  ({BEST})")
print(f"  balanced accuracy / AUC     : {A.loc[0, 'bal_accuracy']:.3f} / {A.loc[0, 'auc']:.3f}")
print(f"  above chance?               : permutation p = {p_acc:.4f} (accuracy), {p_auc:.4f} (AUC)")
print(f"  families reaching {TARGET:.0%}      : {int(A['reaches_target'].sum())} of {len(A)}")
print(f"  fitted ceiling              : {CEIL:.3f}" if np.isfinite(CEIL) else "  fitted ceiling              : n/a")
print(f"  people needed for {TARGET:.0%}      : " + ("out of reach with these features"
      if not np.isfinite(NEED) else f"about {NEED:.0f}"))
print(f"  bulk nigra (25 samples)     : accuracy {bulk[:, 0].mean():.3f}, AUC {bulk[:, 2].mean():.3f} "
      f"(neuron markers alone AUC {neuron_auc:.3f})")
for _, r in CMP.iloc[1:].iterrows():
    print(f"  leaky: {r['variant'][:52]:54s} {r['accuracy']:.3f}")
print("=" * 66)
print(f"{len(man)} files and {len(FIGURES)} figures in {OUT}")''')
