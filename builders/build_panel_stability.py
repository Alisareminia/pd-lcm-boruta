"""Builds PD_LCM_rf_panel_stability.ipynb - does a smaller, more stable panel transfer as well as all 30?

Boruta reselected some panel genes in only a minority of the training folds (MANF and L3MBTL1 in 16%,
LGMN in 20%, CACNA1A in 24%). A reader can fairly ask whether the external result rests on the genes that
Boruta chose every time, or needs the unstable tail as well. This notebook answers it directly: the same
frozen-forest procedure, with the panel restricted by fold frequency, tested in the same eight bulk cohorts.

  panels tested - all 30 genes (as reported), then the genes reselected in more than 50% and more than 70%
                  of the 25 training folds, and, as a control against cherry-picking a threshold, every
                  cut-off from 0 to 100% in steps of 4%;
  a null        - 200 random panels of the same size drawn from the 5,622 measured genes, so that "a smaller
                  panel still works" is read against what any panel of that size does;
  nothing new   - the fold frequencies were computed when the panel was selected, on discovery donors only,
                  and no external cohort takes part in choosing anything.
"""
import json, pathlib
import pandas as pd
from kaggle_nb import write_nb

HERE = pathlib.Path(__file__).parent
SRC = (HERE / "build_external_multi.py").read_text()
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))


def block(marker):
    i = SRC.index("code(r'''" + marker) + len("code(r'''")
    return SRC[i:SRC.index("''')", i)]


md("""# Does a smaller, more stable panel transfer as well?

Boruta reselected the 30 panel genes at very different rates across the 25 training folds: *RBM4B* in every
fold, *MANF* and *L3MBTL1* in four of twenty-five. The reported panel was then selected once on all 63
discovery donors and frozen. This notebook asks whether the external performance needs the whole list or only
its stable core.

**Panels.** All 30 genes as reported; the genes reselected in more than 50% of folds; the genes reselected in
more than 70%; and every threshold from 0 to 100% so that no single cut-off is doing the work.

**A null.** For each panel size, 200 random panels drawn from the 5,622 measured genes, scored the same way.
Without it, "fourteen genes also reach 0.8" says nothing.

**What is fixed.** Fold frequencies come from the discovery donors alone and were computed when the panel was
selected. Every forest here is trained on discovery donors only and applied once to each external cohort.""")

code(block("import os, io, re, gzip, glob, json, time, tarfile, urllib.re"))
code("DISC = json.loads(r'''" + json.dumps({}) + "''')   # the donor-overlap check is not repeated here")

md("## 1. Discovery data, the panel and how often Boruta kept each gene")

code(r'''cz = np.load(find_any("core_data.npz", "rf-core"), allow_pickle=True)
X, XR, y, DS = cz["X"], cz["XR"], cz["y"].astype(int), cz["ds"].astype(str)
GENES = [str(g) for g in cz["genes"]]
SYM = pd.read_csv(find_any("15_gene_symbol_map.csv", "rf-core")).set_index("gene")["symbol"].to_dict()
PANEL = pd.read_csv(find_any("04_boruta_selected_genes.csv", "boruta-panel")).gene.tolist()
SH = pd.read_csv(find_any("05_boruta_shap_importance.csv", "boruta-panel")).set_index("gene")
FREQ_COL = "boruta_fold_frequency"
FREQ = SH[FREQ_COL].astype(float).to_dict()
assert all(g in FREQ for g in PANEL), "a panel gene has no fold frequency"
PI = [GENES.index(g) for g in PANEL]
print(f"{len(PANEL)} panel genes; fold frequency column '{FREQ_COL}'")
print(pd.Series({SYM.get(g, g): FREQ[g] for g in PANEL}).sort_values(ascending=False).to_string())''')

code(r'''def oob_threshold(s, yt):
    u = np.unique(np.round(s, 6)); cuts = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])
    acc = [accuracy_score(yt, (s > c).astype(int)) for c in cuts]
    best = np.flatnonzero(np.isclose(acc, max(acc)))
    return float(cuts[best[len(best) // 2]])

def panel_forest(idx, seeds=5):
    """The reported panel procedure on any set of gene columns: five forests, averaged, threshold from OOB."""
    rfs = [RandomForestClassifier(n_estimators=1000, max_features="sqrt", class_weight="balanced", oob_score=True,
                                  random_state=42 + k, n_jobs=-1).fit(X[:, idx], y) for k in range(seeds)]
    thr = oob_threshold(np.mean([m.oob_decision_function_[:, 1] for m in rfs], axis=0), y)
    oof = np.mean([m.oob_decision_function_[:, 1] for m in rfs], axis=0)
    return {"rfs": rfs, "thr": thr, "idx": list(idx), "oob_auc": float(roc_auc_score(y, oof))}

CUTS = [round(c, 2) for c in np.arange(0.0, 1.01, 0.04)]
SUBSETS = {}
for c in CUTS:
    idx = [GENES.index(g) for g in PANEL if FREQ[g] > c]
    if len(idx) >= 3:
        SUBSETS[c] = idx
SUBSETS[-1.0] = PI                                   # the reported panel, all 30 genes
print("panel sizes by threshold:", {c: len(v) for c, v in sorted(SUBSETS.items())})
FOR = {c: panel_forest(idx) for c, idx in SUBSETS.items()}
print("discovery out-of-bag AUC:", {c: round(FOR[c]["oob_auc"], 3) for c in sorted(FOR)})''')

md("## 2. The eight bulk cohorts, parsed exactly as in the external validation")

code(block('def geo_url(acc, sub, f): return f"https://ftp.ncbi.nlm.nih.g'))
code(block("# ---------------- GSE7621 (HG-U133 Plus 2) - the same file a"))
code(block("# ---------------- GSE114517 (RNA-seq counts per sample; subs"))

md("## 3. Every panel in every cohort")

code(r'''from scipy.stats import rankdata
zrow = lambda D: ((D.T - D.T.mean()) / D.T.std(ddof=1).clip(lower=0.05)).T
rng = np.random.default_rng(42)

def boot_ci(yv, s, n=4000):
    bs = [roc_auc_score(yv[i], s[i]) for i in (rng.integers(0, len(yv), len(yv)) for _ in range(n)) if len(set(yv[i])) == 2]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

Z = {}
for name, c in COH.items():
    Z[name] = zrow(c["G"]).reindex(GENES).fillna(0.0).T.to_numpy()
    print(f"{name}: {len(c['y'])} donors, {int(c['G'].shape[0])} genes measured")

def score_panel(P, Zc):
    return np.mean([m.predict_proba(Zc[:, P["idx"]])[:, 1] for m in P["rfs"]], axis=0)

rows = []
for c, P in sorted(FOR.items()):
    for name, co in COH.items():
        yv = np.asarray(co["y"], int)
        if len(set(yv)) < 2:
            continue
        s = score_panel(P, Z[name])
        lo, hi = boot_ci(yv, s)
        measured = int(np.isin([GENES[i] for i in P["idx"]], co["G"].index).sum())
        rows.append(dict(threshold=c, n_genes=len(P["idx"]), cohort=name, n=len(yv),
                         genes_measured=measured, auc=float(roc_auc_score(yv, s)), ci_lo=lo, ci_hi=hi,
                         accuracy=float(accuracy_score(yv, (s > P["thr"]).astype(int)))))
BYCOH = pd.DataFrame(rows)
BYCOH.to_csv(OUT / "stability_auc_by_cohort.csv", index=False)
print(BYCOH[BYCOH.threshold.isin([-1.0, 0.48, 0.68])].round(3).to_string(index=False))''')

code(r'''# pooled over cohorts, each donor once, weighted by cohort size
def pooled(d):
    w = d.n / d.n.sum()
    return float((d.auc * w).sum())
POOL = BYCOH.groupby(["threshold", "n_genes"]).apply(pooled).rename("pooled_auc").reset_index()
POOL["genes"] = [", ".join(SYM.get(GENES[i], GENES[i]) for i in FOR[t]["idx"]) for t in POOL.threshold]
POOL["discovery_oob_auc"] = [FOR[t]["oob_auc"] for t in POOL.threshold]
POOL.to_csv(OUT / "stability_pooled_auc.csv", index=False)
print(POOL[["threshold", "n_genes", "discovery_oob_auc", "pooled_auc"]].round(3).to_string(index=False))''')

md("""## 4. Against random panels of the same size

A panel of fourteen genes that reaches 0.8 is only interesting if fourteen genes drawn at random do not.
Each random panel goes through the identical procedure: five forests on the discovery donors, then scored
once in every cohort and pooled the same way.""")

code(r'''# the null is expensive, so it is run for the three panels the paper reports: all 30, >50% and >70% of folds
SIZES = sorted({len(FOR[t]["idx"]) for t in (-1.0, 0.48, 0.68)})
print("random-panel null for sizes:", SIZES)
NRAND = 200
RAND = {}
for k in SIZES:
    aucs = []
    for b in range(NRAND):
        idx = list(rng.choice(len(GENES), size=k, replace=False))
        P = panel_forest(idx, seeds=1)
        d = pd.DataFrame([dict(n=len(COH[n]["y"]),
                               auc=roc_auc_score(np.asarray(COH[n]["y"], int), score_panel(P, Z[n])))
                          for n in COH if len(set(COH[n]["y"])) > 1])
        aucs.append(float((d.auc * d.n / d.n.sum()).sum()))
    RAND[k] = np.array(aucs)
    print(f"{k:3d} random genes: median pooled AUC {np.median(aucs):.3f}, 95th percentile {np.percentile(aucs, 95):.3f}")
pd.DataFrame({f"size_{k}": v for k, v in RAND.items()}).to_csv(OUT / "stability_random_panels.csv", index=False)''')

code(r'''rows = []
for t in sorted(FOR):
    k = len(FOR[t]["idx"])
    obs = float(POOL.loc[POOL.threshold == t, "pooled_auc"].iloc[0])
    if k not in RAND:                       # sizes outside the three reported panels carry no null
        rows.append(dict(threshold=t, n_genes=k, pooled_auc=obs, random_median=np.nan,
                         random_p95=np.nan, p_vs_random=np.nan))
        continue
    null = RAND[k]
    rows.append(dict(threshold=t, n_genes=k, pooled_auc=obs,
                     random_median=float(np.median(null)), random_p95=float(np.percentile(null, 95)),
                     p_vs_random=float((np.sum(null >= obs) + 1) / (len(null) + 1))))
VS = pd.DataFrame(rows)
VS.to_csv(OUT / "stability_vs_random.csv", index=False)
print(VS.round(3).to_string(index=False))

REPORTED = VS[VS.threshold == -1.0].iloc[0]
BEST = VS[(VS.threshold >= 0) & VS.p_vs_random.notna()].sort_values("pooled_auc", ascending=False).iloc[0]
SUMMARY = dict(reported=dict(n_genes=int(REPORTED.n_genes), pooled_auc=float(REPORTED.pooled_auc),
                             p_vs_random=float(REPORTED.p_vs_random)),
               best_stable=dict(threshold=float(BEST.threshold), n_genes=int(BEST.n_genes),
                                pooled_auc=float(BEST.pooled_auc), p_vs_random=float(BEST.p_vs_random),
                                genes=[SYM.get(GENES[i], GENES[i]) for i in FOR[BEST.threshold]["idx"]]),
               at_50=VS[VS.threshold == 0.48].to_dict("records"),
               at_70=VS[VS.threshold == 0.68].to_dict("records"),
               fold_frequency={SYM.get(g, g): FREQ[g] for g in PANEL})
json.dump(SUMMARY, open(OUT / "stability_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in SUMMARY.items() if k != "fold_frequency"}, indent=1))''')

md("""## 5. Reading this honestly

If the stable subset matches the full panel, the reported 30 genes are carrying about as much as their core
does, and the unstable tail is candidate biology rather than signal. If it does better, the tail is noise
worth dropping. If it does worse, the tail matters and the panel should stay whole. The random-panel null
decides how much of any of it is the panel and how much is simply fitting thirty numbers to 63 people.""")

write_nb(HERE / "PD_LCM_rf_panel_stability.ipynb", CELLS, "stb")
