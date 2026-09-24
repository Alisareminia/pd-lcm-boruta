"""Builds PD_LCM_rf_boruta_panel.ipynb - secondary notebook: the Boruta + Random Forest gene panel.

Reads only the core notebook's output. Boruta (Random Forest importance) is re-run
inside each of the core's folds to score the panel honestly, then run on all 63 people
for the reported panel, explained with TreeSHAP, compared with the core model's genes,
and written out in the table format the figures notebook reads.
"""
import pathlib
from kaggle_nb import SETUP, BLOCKS, LOAD_CORE, write_nb
HERE = pathlib.Path(__file__).parent
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md("""# Boruta + Random Forest gene panel (secondary notebook)

Reads the output of **`pd-lcm-rf-core`**: the same 63 people, the same folds, and the
core model's gene importance.

1. **Honest panel performance** - Boruta is re-run from scratch inside every training fold
   (standard setting perc 100, and the broadened perc 99 used for the reported panel); a
   Random Forest on the confirmed genes scores the held-out people.
2. **The reported panel** - Boruta on all 63 people, a Random Forest on the confirmed
   genes, explained with TreeSHAP; how often each gene was also confirmed inside the folds.
3. **Agreement** with the core model's top genes, and single-gene ROCs.
4. **Tables** for Figures 4 and 5 and the performance table.""")

code(SETUP + r'''
try:
    import boruta
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "Boruta==0.4.3"], check=True)
from boruta import BorutaPy
import shap
from scipy.stats import hypergeom
from sklearn.model_selection import StratifiedKFold
N_TREES   = 40 if SMOKE else 1000   # trees in the panel forest
B_TREES   = 40 if SMOKE else 500    # trees per Boruta iteration
B_ITER    = 25 if SMOKE else 100     # Boruta iterations
MAIN_PERC = 99                      # reported panel; perc 100 (strictest) is scored alongside''')

code(LOAD_CORE)

code(BLOCKS + r'''

def boruta(idx, perc, seed=SEED):
    b = BorutaPy(RandomForestClassifier(max_features="sqrt", class_weight="balanced", n_jobs=N_CPU),
                 n_estimators=B_TREES, perc=perc, alpha=0.05, two_step=True, max_iter=B_ITER,
                 random_state=seed, verbose=0).fit(X[idx], y[idx])
    conf = np.where(b.support_)[0]
    used = conf if len(conf) >= 2 else np.where(b.support_ | b.support_weak_)[0]
    if len(used) < 2:
        used = np.argsort(b.ranking_)[:2]
    return b, conf, used

def tree_shap(model, Z):
    v = shap.TreeExplainer(model).shap_values(Z)
    v = v[1] if isinstance(v, list) else v
    return v[..., 1] if v.ndim == 3 else v''')

md("## 1. Boruta inside every fold")

code(r'''PERCS = sorted({MAIN_PERC, 100})
REC = []
for i, f in enumerate(FOLDS):
    rec = {k: v for k, v in f.items() if k not in ("train", "test")}; rec["test"] = f["test"]
    for perc in PERCS:
        _, conf, used = boruta(f["train"], perc, f["seed"])
        s, t = fit_score(("genes", used, {}), f["train"], f["test"], y, n_jobs=N_CPU)
        rec[f"perc{perc}"] = {"score": s.tolist(), "thr": t, "genes": [GENES[j] for j in conf]}
    REC.append(rec)
    log(f"fold {i + 1}/{len(FOLDS)} {f['tag']}: " + ", ".join(f"{len(rec[f'perc{p}']['genes'])} genes (perc {p})" for p in PERCS))
json.dump([{**r, "test": r["test"].tolist()} for r in REC], open(OUT / "panel_fold_records.json", "w"))

rows = [summarise(f"Boruta (perc {p}) + Random Forest", REC, f"perc{p}") for p in PERCS]
CORE = pd.read_csv(find_input("core_performance.csv"))
PERF = pd.concat([CORE, pd.DataFrame(rows)], ignore_index=True).sort_values("cv_auc", ascending=False).reset_index(drop=True)
PERF.insert(0, "rank", range(1, len(PERF) + 1))
PERF.to_csv(OUT / "performance_all_models.csv", index=False)
pd.set_option("display.width", 220)
print(PERF[["rank", "model", "cv_auc", "cv_auc_sd", "cv_accuracy", "cv_bal_accuracy", "lodo_auc"]].round(3).to_string(index=False))''')

md("## 2. The reported panel on all 63 people")

code(r'''ALL = np.arange(len(y))
_, conf100, _ = boruta(ALL, 100)
bor, conf, used_all = boruta(ALL, MAIN_PERC)
tent = np.where(bor.support_weak_)[0]
if len(conf) == 0:                                   # never expected on the full run; keeps smoke runs going
    print("no gene confirmed - using confirmed + tentative / best-ranked instead"); conf = used_all
panel = [GENES[j] for j in conf]
log(f"Boruta on all 63: {len(conf)} confirmed at perc {MAIN_PERC} ({len(tent)} tentative); {len(conf100)} at perc 100")
print(", ".join(sym(g) for g in panel))

panel_rf = rf(n_jobs=N_CPU).fit(X[:, conf], y)
SHAP = tree_shap(panel_rf, X[:, conf])
mean_abs = np.abs(SHAP).mean(0)
rho = np.array([pd.Series(X[:, j]).corr(pd.Series(SHAP[:, k]), method="spearman") for k, j in enumerate(conf)])
cv = [r for r in REC if r["kind"] == "cv"]
freq = pd.Series(0.0, index=GENES)
for r in cv:
    freq[r[f"perc{MAIN_PERC}"]["genes"]] += 1
freq /= len(cv)''')

md("## 3. Agreement with the core model, and single genes")

code(r'''CG = pd.read_csv(find_input("core_gene_importance.csv")).set_index("gene")
N = len(panel); top_core = CG.sort_values("rank").index[:N].tolist()
ovl = sorted(set(panel) & set(top_core))
p_hyper = float(hypergeom.sf(len(ovl) - 1, len(GENES), N, N)) if N else 1.0
AGREE = {"panel_size": N, "overlap_with_core_topN": len(ovl), "expected_by_chance": N * N / len(GENES),
         "hypergeometric_p": p_hyper, "overlap_genes": [sym(g) for g in ovl],
         "panel_median_core_rank": float(CG.loc[panel, "rank"].median()) if N else None,
         "panel_core_ranks": {sym(g): int(CG.loc[g, "rank"]) for g in panel}}
json.dump(AGREE, open(OUT / "panel_core_agreement.json", "w"), indent=1)
log(f"{len(ovl)} of {N} panel genes are in the core model's top {N} (chance {AGREE['expected_by_chance']:.2f}, "
    f"p = {p_hyper:.2g}); median core rank of panel genes {AGREE['panel_median_core_rank']}")

# each gene's own expression as the score; its direction is learned on the training fold
def single_gene_oof(j, reps=10):
    P = np.zeros(len(y))
    for r in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(X, STRATA):
            sgn = np.sign(X[tr][y[tr] == 1, j].mean() - X[tr][y[tr] == 0, j].mean()) or 1.0
            P[te] += sgn * X[te, j]
    return P / reps
rng = np.random.default_rng(SEED)
BOOT = [rng.integers(0, len(y), len(y)) for _ in range(200 if SMOKE else 2000)]
def auc_ci(p):
    a = roc_auc_score(y, p); flipped = a < 0.5
    if flipped:
        p = -p; a = 1 - a
    bs = [roc_auc_score(y[i], p[i]) for i in BOOT if len(set(y[i])) == 2]
    return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5), flipped, p
sg_rows, roc_rows = [], []
for j in conf:
    a, lo, hi, fl, p = auc_ci(single_gene_oof(j))
    g = GENES[j]
    sg_rows.append({"gene": g, "symbol": sym(g), "auc": a, "ci_lo": lo, "ci_hi": hi, "direction_flipped": fl})
    fpr, tpr, _ = roc_curve(y, p)
    roc_rows += [{"curve": g, "curve_type": "single_gene", "fpr": a_, "tpr": b_, "auc": a, "ci_lo": lo, "ci_hi": hi}
                 for a_, b_ in zip(fpr, tpr)]
SG = pd.DataFrame(sg_rows).sort_values("auc", ascending=False).reset_index(drop=True)
print(SG[["symbol", "auc", "ci_lo", "ci_hi"]].round(3).to_string(index=False))''')

md("## 4. Tables for the figures")

code(r'''NOMINAL_P = 0.01
de = pd.read_csv(find_input("03_de_results_full.csv")); de["is_deg"] = de["pvalue"] < NOMINAL_P
deg_set = set(de.loc[de.is_deg, "gene"]); bor_set = set(panel); overlap = bor_set & deg_set
imp = pd.DataFrame({"gene": panel, "symbol": [sym(g) for g in panel], "mean_abs_shap": mean_abs, "mean_shap": SHAP.mean(0),
                    "shap_std": SHAP.std(0), "shap_expression_spearman": rho,
                    "boruta_fold_frequency": freq[panel].to_numpy(), "core_model_rank": CG.loc[panel, "rank"].to_numpy()})
imp["direction_sign"] = np.sign(imp.shap_expression_spearman).astype(int)
imp["signed_importance"] = imp.direction_sign * imp.mean_abs_shap
imp["direction"] = np.where(imp.direction_sign > 0, "Higher expression raises PD probability",
                            "Higher expression lowers PD probability")
imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
imp["shap_rank"] = range(1, len(imp) + 1)
deL = de.set_index("gene")
ann = imp.merge(SG[["gene", "auc", "ci_lo", "ci_hi"]].rename(columns={"auc": "single_gene_auc", "ci_lo": "single_gene_ci_lo",
                                                                     "ci_hi": "single_gene_ci_hi"}), on="gene")
for c in ("log2FC", "pvalue", "padj", "hedges_g_meta", "I2", "same_direction_datasets"):
    ann[c] = ann.gene.map(deL[c])
ann["is_deg"] = ann.gene.isin(deg_set); ann["in_overlap_panel"] = ann.gene.isin(overlap)
ann["in_core_topN"] = ann.gene.isin(top_core)
members = sorted(bor_set | deg_set)
memb = pd.DataFrame({"gene": members, "symbol": [sym(g) for g in members]})
memb["in_boruta"] = memb.gene.isin(bor_set); memb["in_deg"] = memb.gene.isin(deg_set); memb["in_overlap"] = memb.gene.isin(overlap)
for c in ("log2FC", "pvalue", "padj"):
    memb[c] = memb.gene.map(deL[c])

W = {}
def save(df, name, desc):
    df.to_csv(OUT / name, index=False); W[name] = (len(df), desc)
save(de, "03_de_results_full.csv", "DE across 63 people (merged pipeline; nominal p<0.01 flagged)")
save(pd.DataFrame([("de_rule", "nominal"), ("de_label", f"nominal DE genes (p < {NOMINAL_P}, not FDR-corrected)"),
                   ("nominal_p", NOMINAL_P), ("n_de_set", len(deg_set))], columns=["statistic", "value"]),
     "03_de_settings.csv", "DE rule used")
save(pd.DataFrame({"gene": panel, "symbol": [sym(g) for g in panel], "boruta_rank": bor.ranking_[conf], "confirmed": True}),
     "04_boruta_selected_genes.csv", f"Boruta + Random Forest (perc {MAIN_PERC}), all 63 people")
save(pd.DataFrame([("importance_model", f"Random Forest, {B_TREES} trees, max_features=sqrt, balanced"),
                   ("boruta_perc", MAIN_PERC), ("n_confirmed", len(conf)), ("n_tentative", len(tent)),
                   ("n_confirmed_perc100", len(conf100)),
                   ("mean_genes_per_training_fold", float(np.mean([len(r[f"perc{MAIN_PERC}"]["genes"]) for r in cv]))),
                   ("n_folds", len(cv))], columns=["setting", "value"]), "04_boruta_settings.csv", "Boruta settings and stability")
save(imp, "05_boruta_shap_importance.csv", "TreeSHAP of the Random Forest on the Boruta genes")
save(pd.DataFrame([{"set": "Boruta only", "n": len(bor_set - deg_set)}, {"set": "DEG only", "n": len(deg_set - bor_set)},
                   {"set": "Overlap", "n": len(overlap)}, {"set": "Boruta total", "n": len(bor_set)},
                   {"set": "DEG total", "n": len(deg_set)}]), "06_overlap_analysis.csv", "Venn counts")
save(memb, "06_gene_set_membership.csv", "Boruta / DE membership")
save(SG, "07_single_gene_auc.csv", "Out-of-fold single-gene AUC (10 x 5 CV, 63 people)")
save(ann, "13_gene_annotation_master.csv", "Per-gene summary for the Boruta genes")
save(pd.DataFrame(roc_rows), "15_roc_curves.csv", "Out-of-fold single-gene ROC coordinates")
for f in ("15_gene_symbol_map.csv", "01_preprocessing_summary.csv", "01_cohort_by_dataset.csv"):
    save(pd.read_csv(find_input(f)), f, "copied from the core notebook")
pd.DataFrame([{"n": i + 1, "file": k, "rows": v[0], "description": v[1]} for i, (k, v) in enumerate(W.items())]
             ).to_csv(OUT / "00_manifest.csv", index=False)
print(imp[["shap_rank", "symbol", "mean_abs_shap", "direction_sign", "boruta_fold_frequency", "core_model_rank"]].round(3).to_string(index=False))
log("done")''')

write_nb(HERE / "PD_LCM_rf_boruta_panel.ipynb", CELLS, "bor")
