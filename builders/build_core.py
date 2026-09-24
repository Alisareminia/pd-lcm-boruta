"""Builds PD_LCM_rf_core.ipynb - the core notebook every other analysis reads.

The high-AUC Random Forest (within-person ranks -> 30 principal components ->
Random Forest), evaluated on fresh folds and leave-one-study-out, then fitted on all
63 people and explained with TreeSHAP traced back to genes. Saves the data, the
folds, the model and the tables the secondary notebooks need.
"""
import json, pathlib
from kaggle_nb import SETUP, BLOCKS, write_nb
HERE = pathlib.Path(__file__).parent
REF = json.loads((HERE / "data/mdata_reference.json").read_text())
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md("""# Core: Random Forest on laser-captured dopamine neurons in Parkinson's disease

**63 people** from four laser-capture studies (GSE182622, GSE20141, GSE24378, GSE169755),
one profile per person, each study z-scored on its own without looking at diagnosis.

**The model.** Each person's 5,622 genes are ranked within that person, the ranks are
compressed into 30 principal components (fitted on training people only), and a Random
Forest (1,000 trees, half the components tried at each split) classifies. It was the
best of 97 Random Forest variants in `pd-lcm-rf-confirm`; here it is re-scored on folds
that sweep never used.

**What this notebook saves** for the secondary notebooks: the data (`core_data.npz`),
the folds (`core_folds.json`), the fitted model (`core_model.joblib`), its performance,
its out-of-fold scores, and its SHAP importance for every component and every gene.""")

code(SETUP + r'''
import joblib, shap
from scipy.stats import rankdata
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
N_REP   = 1 if SMOKE else 5       # repeats of stratified 5-fold
N_TREES = 40 if SMOKE else 1000''')

code("REF = json.loads(r'''" + json.dumps(REF) + "''')")

code(r'''# ============================== LOAD AND CHECK ==============================
if ON_KAGGLE:
    z = np.load(find_input("18_hvg_expression.npz"), allow_pickle=True)
    Xk = pd.DataFrame(z["X"].astype(float), index=z["person"].astype(str), columns=[str(g) for g in z["genes"]])
    X = Xk.loc[REF["person"], REF["genes"]].to_numpy()
    yk = pd.Series(z["y"].astype(int), index=z["person"].astype(str)).loc[REF["person"]].to_numpy()
else:
    z = np.load(find_input("mdata.npz"), allow_pickle=True)
    X = z["X"].astype(float); yk = z["y"].astype(int)
y = np.array(REF["y"]); DS = np.array(REF["ds"]); PERSON = np.array(REF["person"]); GENES = list(REF["genes"])
diff = np.abs(X.sum(1) - np.array(REF["row_sum"])).max()
assert diff < 0.05 and (yk == y).all(), f"matrix differs from the reference ({diff:.4f})"
print("matrix identical to the reference copy (row sums and labels)")
STRATA = np.array([f"{d}_{v}" for d, v in zip(DS, y)])
XR = np.apply_along_axis(rankdata, 1, X) / X.shape[1]          # within-person ranks, label-free
SYM = pd.read_csv(find_input("15_gene_symbol_map.csv")).set_index("gene")["symbol"].to_dict()
sym = lambda g: SYM[g] if isinstance(SYM.get(g), str) else g
print(f"{len(y)} people ({(y == 0).sum()} control, {y.sum()} PD), {X.shape[1]:,} genes")
print(pd.crosstab(DS, np.where(y == 1, "PD", "control")).to_string())''')

code(BLOCKS + r'''

def tree_shap(model, Z):
    v = shap.TreeExplainer(model).shap_values(Z)
    v = v[1] if isinstance(v, list) else v
    return v[..., 1] if v.ndim == 3 else v

def head_explain(idx):
    """Fit the headline model on idx. SHAP on its components, shared out to genes by squared loadings
    (each loading vector has unit length, so a component's importance is split, not created)."""
    p = PCA(30, random_state=SEED).fit(XR[idx]); Z = p.transform(XR[idx])
    m = rf(n_jobs=N_CPU, **HEAD[2]).fit(Z, y[idx])
    phi = tree_shap(m, Z)
    axis_imp = np.abs(phi).mean(0)
    gene_imp = (p.components_ ** 2 * axis_imp[:, None]).sum(0)
    push = phi.sum(1)                                                # each person's SHAP push towards PD
    Rg = np.apply_along_axis(rankdata, 0, XR[idx]); Rt = rankdata(push)
    Rg = (Rg - Rg.mean(0)) / (Rg.std(0) + 1e-12); Rt = (Rt - Rt.mean()) / Rt.std()
    rho = (Rg * Rt[:, None]).mean(0)                                 # Spearman: gene rank vs PD push
    return p, m, phi, axis_imp, gene_imp, rho''')

md("""## 1. The folds every notebook uses

5 x 5-fold cross-validation by person, stratified by study and diagnosis (seeds 1000-1004,
never used in the sweep), plus leave one study out. Saved so every secondary notebook
scores on exactly the same people.""")

code(r'''FOLDS = []
for r in range(N_REP):
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=1000 + r).split(X, STRATA)):
        FOLDS.append({"tag": f"r{r}k{k}", "kind": "cv", "rep": r, "seed": 1000 + 100 * r + k, "train": tr, "test": te})
for d in sorted(set(DS)):
    FOLDS.append({"tag": f"lodo_{d}", "kind": "lodo", "rep": -1, "seed": SEED,
                  "train": np.where(DS != d)[0], "test": np.where(DS == d)[0]})
json.dump([{**f, "train": f["train"].tolist(), "test": f["test"].tolist()} for f in FOLDS],
          open(OUT / "core_folds.json", "w"))
print(len(FOLDS), "folds saved")''')

md("""## 2. How well it classifies

The headline model and, for reference, a plain Random Forest on all 5,622 genes, on the
same folds. Accuracy uses a 0.5 cut; `accuracy_oob_cut` uses a cut chosen on the
training people's out-of-bag predictions. For every CV fold the model is also explained
on its training people, to measure how stable its top genes are.""")

code(r'''def run_fold(f):
    tr, te = f["train"], f["test"]
    rec = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in f.items() if k != "train"}
    s, t = fit_score(HEAD, tr, te, y); rec["head"] = {"score": s.tolist(), "thr": t}
    s, t = fit_score(("all", None, {}), tr, te, y); rec["rf_all"] = {"score": s.tolist(), "thr": t}
    return rec
REC = Parallel(n_jobs=N_CPU)(delayed(run_fold)(f) for f in FOLDS)
for rec, f in zip(REC, FOLDS):
    rec["test"] = np.array(rec["test"])
    if f["kind"] == "cv":
        *_, gene_imp, _ = head_explain(f["train"])
        rec["head_top100"] = [GENES[j] for j in np.argsort(-gene_imp)[:100]]
PERF = pd.DataFrame([summarise(HEAD_NAME, REC, "head"), summarise("Random Forest, all genes", REC, "rf_all")])
PERF.to_csv(OUT / "core_performance.csv", index=False)
json.dump([{**r, "test": r["test"].tolist()} for r in REC], open(OUT / "core_fold_records.json", "w"))
pd.set_option("display.width", 220)
print(PERF[["model", "cv_auc", "cv_auc_sd", "cv_accuracy", "cv_accuracy_sd", "cv_accuracy_oob_cut", "cv_bal_accuracy",
            "cv_sensitivity", "cv_specificity", "lodo_auc"]].round(3).to_string(index=False))

oof = np.zeros(len(y)); cnt = np.zeros(len(y))
for rec in REC:
    if rec["kind"] == "cv":
        oof[rec["test"]] += rec["head"]["score"]; cnt[rec["test"]] += 1
oof /= np.maximum(cnt, 1)
pd.DataFrame({"person": PERSON, "dataset": DS, "y": y, "oof_score": oof}).to_csv(OUT / "core_oof_scores.csv", index=False)
fpr, tpr, _ = roc_curve(y, oof)
pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(OUT / "core_roc_curve.csv", index=False)
log(f"out-of-fold AUC of the averaged scores: {roc_auc_score(y, oof):.3f}")''')

md("""## 3. The final model on all 63 people, and what drives it""")

code(r'''pH, mH, phiH, axis_imp, gene_imp, rho = head_explain(np.arange(len(y)))
joblib.dump({"pca": pH, "rf": mH, "genes": GENES, "note": "input = within-person ranks of the z-scored genes / n_genes"},
            OUT / "core_model.joblib")
head_rank = pd.Series(gene_imp, index=GENES).rank(ascending=False, method="first").astype(int)
cv_recs = [r for r in REC if r["kind"] == "cv"]
freq50 = pd.Series(0.0, index=GENES)
for r in cv_recs:
    freq50[r["head_top100"][:50]] += 1
freq50 /= len(cv_recs)
G = pd.DataFrame({"gene": GENES, "symbol": [sym(g) for g in GENES], "importance": gene_imp,
                  "rank": head_rank.to_numpy(), "direction_rho": rho,
                  "direction": np.where(rho > 0, "higher rank -> more PD-like", "higher rank -> less PD-like"),
                  "top50_fold_frequency": freq50.to_numpy()}).sort_values("rank")
G.to_csv(OUT / "core_gene_importance.csv", index=False)
C = pd.DataFrame({"component": [f"PC{k + 1}" for k in range(len(axis_imp))], "mean_abs_shap": axis_imp,
                  "variance_explained": pH.explained_variance_ratio_}).sort_values("mean_abs_shap", ascending=False)
C.to_csv(OUT / "core_component_importance.csv", index=False)
pd.DataFrame(phiH, columns=[f"PC{k + 1}" for k in range(phiH.shape[1])]).assign(person=PERSON, dataset=DS, y=y) \
  .to_csv(OUT / "core_component_shap.csv", index=False)
print(C.head(8).round(4).to_string(index=False))
print(G.head(25)[["rank", "symbol", "importance", "direction_rho", "top50_fold_frequency"]].round(4).to_string(index=False))''')

code(r'''# ============================== SAVE FOR THE SECONDARY NOTEBOOKS ==============================
np.savez_compressed(OUT / "core_data.npz", X=X, XR=XR, y=y, ds=DS, person=PERSON, genes=np.array(GENES))
for f in ("03_de_results_full.csv", "15_gene_symbol_map.csv", "01_preprocessing_summary.csv", "01_cohort_by_dataset.csv"):
    pd.read_csv(find_input(f)).to_csv(OUT / f, index=False)
hp = PERF.set_index("model").loc[HEAD_NAME]
json.dump({"model": HEAD_NAME, "spec": {"features": "within-person ranks -> PCA(30), fitted on training people",
                                        "forest": {"n_estimators": N_TREES, "max_features": 0.5, "min_samples_leaf": 1,
                                                   "class_weight": "balanced"}},
           "n_people": int(len(y)), "n_genes": int(X.shape[1]),
           "cv_auc": float(hp.cv_auc), "cv_auc_sd": float(hp.cv_auc_sd), "cv_accuracy": float(hp.cv_accuracy),
           "cv_accuracy_sd": float(hp.cv_accuracy_sd), "lodo_auc": float(hp.lodo_auc),
           "chosen_in": "alisaremi/pd-lcm-rf-confirm (97 Random Forest variants)"},
          open(OUT / "core_summary.json", "w"), indent=1)
print(sorted(p.name for p in OUT.iterdir()))
log("core done")''')

write_nb(HERE / "PD_LCM_rf_core.ipynb", CELLS, "core")
