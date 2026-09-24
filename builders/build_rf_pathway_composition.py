"""Builds PD_LCM_rf_pathway_composition.ipynb - g:Profiler pathways, and the neuron-mix (composition) test.

Reads pd-lcm-rf-core (data, folds, out-of-fold scores), pd-lcm-rf-boruta-panel (panel, DE table) and pd-lcm-rf-enrichment
(the Kamath et al. 2022 workbook it saved). Nothing from those notebooks is refitted except where this one says so.
  A. Pathways with g:Profiler itself: custom statistical domain = the 5,622 measured genes, significance threshold =
     Benjamini-Hochberg FDR, lists up (18) / down (12) / all (30). The same settings are run on 100 random gene lists
     of each size, so every result comes with how often chance alone produces it.
  B. Neuron mix: each donor's SOX6-vs-CALB1 composition score from Kamath subtype markers (panel genes excluded), then
     the core classifier (and the panel forest) re-run on the same folds with the composition regressed out of every
     gene. If they still separate PD from controls, the panel carries changes inside surviving neurons.
"""
import pathlib
from kaggle_nb import write_nb, SETUP, BLOCKS, LOAD_CORE
HERE = pathlib.Path(__file__).parent
FC = HERE / "figcode"
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md("""# Pathways (g:Profiler) and the neuron-mix test

**A. Pathways.** g:Profiler g:GOSt with a *custom statistical domain* (the 5,622 genes measured in the discovery data) and
*Benjamini-Hochberg FDR* as the significance threshold - the settings of the g:Profiler web tool - for the genes higher in PD
(18), lower in PD (12) and all 30. The identical request is sent for 100 random gene lists of each size: the share of random
lists that also return "significant" pathways shows how much each result is worth.

**B. Neuron mix.** PD kills SOX6/AGTR1 dopamine neurons and spares CALB1 neurons (Kamath et al. 2022), so a PD donor's
laser-captured neurons are a different *mix*. Each donor gets a composition score from Kamath subtype markers (the 30 panel
genes excluded, so the test is not circular). The composition is then regressed out of every gene and the core classifier is
re-run on the same folds. If it still separates PD from controls, the panel reflects changes *inside* surviving neurons as well
as which neurons survived.""")

code(SETUP)
code("N_TREES = 1000 if ON_KAGGLE else 40\nimport urllib.request, urllib.parse")
code(BLOCKS)
code(LOAD_CORE)
code((FC / "strict_xlsx.py").read_text())

code(r'''from scipy.stats import hypergeom, mannwhitneyu, spearmanr
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict, StratifiedKFold
rng = np.random.default_rng(42)
PANEL = pd.read_csv(find_input("04_boruta_selected_genes.csv")).gene.tolist()
DE = pd.read_csv(find_input("03_de_results_full.csv")).set_index("gene").reindex(GENES)
UP = [g for g in PANEL if DE.loc[g, "hedges_g_meta"] > 0]
DOWN = [g for g in PANEL if DE.loc[g, "hedges_g_meta"] < 0]
LISTS = {"up": UP, "down": DOWN, "all": PANEL}
PI = [GENES.index(g) for g in PANEL]
log(f"panel {len(PANEL)} ({len(UP)} up, {len(DOWN)} down); background {len(GENES):,} genes")''')

md("## A. g:Profiler with the 5,622-gene background and Benjamini-Hochberg FDR")

code(r'''GP_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
SOURCES = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC", "WP"]
def gost(query, all_results=True):
    body = {"organism": "hsapiens", "query": list(query), "sources": SOURCES, "user_threshold": 0.05, "all_results": all_results,
            "ordered": False, "no_evidences": not all_results, "combined": False, "measure_underrepresentation": False,
            "no_iea": False, "domain_scope": "custom", "significance_threshold_method": "fdr", "background": GENES}
    for k in range(6):
        try:
            req = urllib.request.Request(GP_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
                                                                                        "User-Agent": "pd-lcm-analysis"})
            with urllib.request.urlopen(req, timeout=300) as fh:
                return json.loads(fh.read())
        except Exception as exc:
            print("  g:Profiler retry", k + 1, type(exc).__name__); time.sleep(10)
    raise RuntimeError("g:Profiler unreachable")
def parse(r, name):
    ensgs = r["meta"]["genes_metadata"]["query"]["query_1"]["ensgs"]
    rows = []
    for t in r["result"]:
        members = [ensgs[i] for i, ev in enumerate(t.get("intersections", [])) if ev]
        rows.append({"list": name, "source": t["source"], "native": t["native"], "name": t["name"], "term_size": t["term_size"],
                     "query_size": t["query_size"], "hits": t["intersection_size"], "domain": t["effective_domain_size"],
                     "p_raw": hypergeom.sf(t["intersection_size"] - 1, t["effective_domain_size"], t["term_size"], t["query_size"]),
                     "p_adj": t["p_value"], "significant": bool(t["significant"]),
                     "genes": ", ".join(sym(g) for g in members), "genes_ensg": ", ".join(members)})
    return rows
if ON_KAGGLE:
    GPR = []
    for name, genes in LISTS.items():
        GPR += parse(gost(genes), name)
    GPR = pd.DataFrame(GPR).sort_values(["list", "p_adj"]).reset_index(drop=True)
else:                                                    # smoke run: invented results
    GPR = pd.DataFrame([{"list": l, "source": "GO:BP", "native": f"GO:{i:07d}", "name": f"term {i}", "term_size": 40, "query_size": 10,
                         "hits": 2, "domain": 5000, "p_raw": 1e-3 * (i + 1), "p_adj": 0.02 * (i + 1), "significant": i < 2,
                         "genes": ", ".join(sym(g) for g in LISTS[l][:2]), "genes_ensg": ""} for l in LISTS for i in range(6)])
GPR.to_csv(OUT / "gprofiler_results.csv", index=False)
for name in LISTS:
    s = GPR[GPR.list == name]
    log(f"{name}: {len(s)} terms touched, {int(s.significant.sum())} significant (BH FDR < 0.05)")
    print(s.head(10)[["source", "name", "term_size", "hits", "genes", "p_raw", "p_adj", "significant"]].to_string(index=False))''')

code(r'''# the same request for random gene lists of the same sizes
B_CAL = 100 if ON_KAGGLE else 5
CALR = []
for name, genes in LISTS.items():
    n = len(genes)
    for b in range(B_CAL):
        q = list(rng.choice(GENES, n, replace=False))
        if ON_KAGGLE:
            r = gost(q, all_results=False); res = r["result"]; time.sleep(0.3)
            CALR.append({"list": name, "draw": b, "n_significant": len(res), "best_p_adj": min([t["p_value"] for t in res], default=1.0)})
        else:
            CALR.append({"list": name, "draw": b, "n_significant": int(np.random.default_rng(b).integers(0, 3)), "best_p_adj": 0.5})
    log(f"calibration {name}: {B_CAL} random lists of {n}")
CALR = pd.DataFrame(CALR); CALR.to_csv(OUT / "gprofiler_random_lists.csv", index=False)
CAL = {}
for name in LISTS:
    c = CALR[CALR.list == name]; s = GPR[GPR.list == name]
    best = s.p_adj.min() if len(s) else 1.0
    CAL[name] = {"n": len(LISTS[name]), "observed_significant": int(s.significant.sum()), "observed_best_p_adj": float(best),
                 "random_with_any_significant": float((c.n_significant > 0).mean()),
                 "random_mean_significant": float(c.n_significant.mean()),
                 "random_at_least_observed": float((c.n_significant >= s.significant.sum()).mean()),
                 "calibrated_p_best": float((np.sum(c.best_p_adj <= best) + 1) / (len(c) + 1))}
GPR["calibrated_p"] = [float((np.sum(CALR[CALR.list == r.list].best_p_adj <= r.p_adj) + 1) / (B_CAL + 1)) for r in GPR.itertuples()]
GPR.to_csv(OUT / "gprofiler_results.csv", index=False)
print(json.dumps(CAL, indent=1))''')

md("""## B. The neuron-mix test

**Composition score** of a donor = mean z-score of the 50 strongest CALB1-lineage markers minus the mean of the 50 strongest
SOX6-lineage markers (Kamath et al. 2022, Supplementary Table 8; lineage = mean marker z over CALB1 subtypes minus mean over
SOX6 subtypes), computed within each study, **panel genes excluded**. It uses no labels. Sensitivity: 25, 100 and 200 markers;
the SOX6_AGTR1 markers alone; composition plus dopamine-neuron purity.""")

code(r'''T8 = read_strict_xlsx(find_input("kamath_2022_supplementary.xlsx"))["Supplementary_Table_8"]
T8.columns = [str(c) if c is not None else f"c{i}" for i, c in enumerate(T8.iloc[0])]; T8 = T8.iloc[1:]
T8["z"] = pd.to_numeric(T8["z"], errors="coerce"); T8["symbol"] = T8.primerid.astype(str).str.upper()
ZT = T8.pivot_table(index="symbol", columns="DA_subtype", values="z", aggfunc="first").fillna(0.0)
SOXS = [c for c in ZT.columns if c.startswith("SOX6")]; CALS = [c for c in ZT.columns if c.startswith("CALB1")]
LIN_SYM = (ZT[CALS].mean(1) - ZT[SOXS].mean(1)).to_dict(); AGT_SYM = ZT["SOX6_AGTR1"].to_dict()
SU = [sym(g).upper() for g in GENES]
LIN = np.array([LIN_SYM.get(s, 0.0) for s in SU]); AGT = np.array([AGT_SYM.get(s, 0.0) for s in SU])
def zds(M):
    out = np.empty_like(M, dtype=float)
    for d in set(DS):
        m = DS == d; sd = M[m].std(0, ddof=1); out[m] = (M[m] - M[m].mean(0)) / np.where(sd > 1e-9, sd, 1)
    return out
XZ = zds(X.astype(float))
NOTP = np.array([g not in set(PANEL) for g in GENES])
def markers(v, n, sign):
    idx = np.flatnonzero(NOTP & (np.sign(v) == sign) & (v != 0))
    return idx[np.argsort(-np.abs(v[idx]))][:n]
def comp_score(n):
    cal, sox = markers(LIN, n, +1), markers(LIN, n, -1)
    return XZ[:, cal].mean(1) - XZ[:, sox].mean(1), cal, sox
CS, CAL50, SOX50 = comp_score(50)
CAND = [s for s in ["TH", "DDC", "SLC18A2", "SLC6A3", "NR4A2", "EN1", "FOXA2", "LMX1B", "PITX3", "DLK1", "CHRNA4", "RET"]
        if s in SU and GENES[SU.index(s)] not in set(PANEL)]
print("pan-dopamine candidates and their lineage scores:", {s: round(LIN_SYM.get(s, 0.0), 1) for s in CAND})
PANDA = [s for s in CAND if abs(LIN_SYM.get(s, 0.0)) < 5.0] or CAND      # keep the ones that do not mark a lineage
PURITY = XZ[:, [SU.index(s) for s in PANDA]].mean(1)
AGT50 = markers(AGT, 50, +1); AGTS = XZ[:, AGT50].mean(1)
print(f"lineage markers used: {len(CAL50)} CALB1 ({', '.join(sym(GENES[j]) for j in CAL50[:8])}, ...) and {len(SOX50)} SOX6 "
      f"({', '.join(sym(GENES[j]) for j in SOX50[:8])}, ...); pan-dopamine purity genes: {PANDA}")
def auc_within(score):
    """AUC of a score for PD vs control, averaged over studies with weights = PD-control pairs."""
    a, w = [], []
    for d in sorted(set(DS)):
        m = DS == d
        if len(set(y[m])) == 2:
            a.append(roc_auc_score(y[m], score[m])); w.append(y[m].sum() * (1 - y[m]).sum())
    return float(np.average(a, weights=w))
COMPO = pd.DataFrame({"person": PERSON, "dataset": DS, "y": y, "composition_score": CS, "agtr1_score": AGTS, "purity": PURITY})
OOF = pd.read_csv(find_input("core_oof_scores.csv")).set_index("person").reindex(PERSON)
COMPO["core_oof_score"] = OOF.oof_score.to_numpy()
COMPO.to_csv(OUT / "composition_scores.csv", index=False)
C_SUM = {"auc_composition_pd_more_CALB1_like": auc_within(CS), "composition_mwu_p": mannwhitneyu(CS[y == 1], CS[y == 0]).pvalue,
         "auc_AGTR1_markers_pd_lower": auc_within(-AGTS), "auc_dopamine_purity_pd_lower": auc_within(-PURITY),
         "rho_oof_composition": spearmanr(COMPO.core_oof_score, CS)[0],
         "rho_oof_composition_controls": spearmanr(COMPO.core_oof_score[y == 0], CS[y == 0])[0],
         "rho_oof_composition_pd": spearmanr(COMPO.core_oof_score[y == 1], CS[y == 1])[0]}
print(json.dumps(C_SUM, indent=1, default=float))
print(COMPO.groupby(["dataset", "y"]).composition_score.mean().unstack().round(2))''')

code(r'''# the classifier with the composition removed from every gene, same folds, same model
def regress_out(covs):
    out = np.empty_like(XZ)
    for d in set(DS):
        m = DS == d; Cd = np.column_stack([np.ones(m.sum())] + [c[m] for c in covs])
        B, *_ = np.linalg.lstsq(Cd, XZ[m], rcond=None); out[m] = XZ[m] - Cd @ B
    return zds(out)
VARIANTS = {"original": X.astype(float), "composition (50 markers)": regress_out([CS]),
            "composition (25 markers)": regress_out([comp_score(25)[0]]), "composition (100 markers)": regress_out([comp_score(100)[0]]),
            "composition (200 markers)": regress_out([comp_score(200)[0]]), "SOX6_AGTR1 markers only": regress_out([AGTS]),
            "composition + dopamine purity": regress_out([CS, PURITY])}
def one_fold(Xm, XRm, f):
    tr, te = np.array(f["train"]), np.array(f["test"])
    p = PCA(30, random_state=SEED).fit(XRm[tr])
    m = rf(n_jobs=1, **HEAD[2]).fit(p.transform(XRm[tr]), y[tr])
    m2 = rf(n_jobs=1).fit(Xm[tr][:, PI], y[tr])
    return {"kind": f["kind"], "rep": f["rep"], "tag": f["tag"], "test": te,
            "core": {"score": m.predict_proba(p.transform(XRm[te]))[:, 1], "thr": oob_threshold(m, y[tr])},
            "panel": {"score": m2.predict_proba(Xm[te][:, PI])[:, 1], "thr": oob_threshold(m2, y[tr])}}
CV, OOFV = [], {}
for name, Xm in VARIANTS.items():
    XRm = np.apply_along_axis(rankdata, 1, Xm) / Xm.shape[1]
    recs = Parallel(n_jobs=N_CPU)(delayed(one_fold)(Xm, XRm, f) for f in FOLDS)
    for key, lab in (("core", "core classifier"), ("panel", "30-gene forest (genes fixed on all donors; optimistic)")):
        row = summarise(lab, recs, key); row["variant"] = name; CV.append(row)
        oof = np.zeros(len(y)); cnt = np.zeros(len(y))
        for r in recs:
            if r["kind"] == "cv":
                oof[r["test"]] += r[key]["score"]; cnt[r["test"]] += 1
        OOFV[(name, key)] = oof / np.maximum(cnt, 1)
    log(f"{name}: core AUC {CV[-2]['cv_auc']:.3f} +/- {CV[-2]['cv_auc_sd']:.3f}; panel forest {CV[-1]['cv_auc']:.3f}")
CV = pd.DataFrame(CV); CV.to_csv(OUT / "composition_cv.csv", index=False)
pd.DataFrame({f"{v} | {k}": s for (v, k), s in OOFV.items()}).assign(person=PERSON, dataset=DS, y=y).to_csv(OUT / "composition_oof_scores.csv", index=False)
print(CV[["variant", "model", "cv_auc", "cv_auc_sd", "cv_accuracy", "cv_sensitivity", "cv_specificity", "lodo_auc"]].round(3).to_string(index=False))''')

code(r'''# is the adjusted classifier better than chance? bootstrap CI and label shuffles on its out-of-fold scores
rng2 = np.random.default_rng(7)
def boot_auc(s, n=4000):
    bs = [roc_auc_score(y[i], s[i]) for i in (rng2.integers(0, len(y), len(y)) for _ in range(n)) if len(set(y[i])) == 2]
    return np.percentile(bs, [2.5, 97.5])
def shuffle_p(s, n=10000):
    a = roc_auc_score(y, s); ys = y.copy(); null = []
    for _ in range(n):
        for d in set(DS):
            m = np.flatnonzero(DS == d); ys[m] = rng2.permutation(y[m])
        null.append(roc_auc_score(ys, s))
    return (np.sum(np.array(null) >= a) + 1) / (n + 1)
ADJ = []
for (v, k), s in OOFV.items():
    lo, hi = boot_auc(s)
    ADJ.append({"variant": v, "model": k, "oof_auc": roc_auc_score(y, s), "ci_lo": lo, "ci_hi": hi, "shuffle_p": shuffle_p(s)})
ADJ = pd.DataFrame(ADJ); ADJ.to_csv(OUT / "composition_oof_auc.csv", index=False)
print(ADJ.round(4).to_string(index=False))
# does the original classifier score add to composition? logistic regression with study as a covariate
D = pd.get_dummies(pd.Series(DS), drop_first=True).astype(float).to_numpy()
LR = {}
try:
    import statsmodels.api as sm
    from scipy.stats import chi2
    Xa = sm.add_constant(np.column_stack([D, CS])); Xb = sm.add_constant(np.column_stack([D, CS, COMPO.core_oof_score.to_numpy()]))
    fa, fb = sm.Logit(y, Xa).fit(disp=0), sm.Logit(y, Xb).fit(disp=0)
    LR = {"lr_stat": 2 * (fb.llf - fa.llf), "lr_p": chi2.sf(2 * (fb.llf - fa.llf), 1),
          "score_coef_z": float(fb.tvalues[-1]), "score_coef_p": float(fb.pvalues[-1])}
except Exception as exc:
    print("statsmodels unavailable:", exc)
LR.update({"loo_auc_composition": roc_auc_score(y, cross_val_predict(LogisticRegression(max_iter=1000), np.column_stack([D, CS]), y, cv=LeaveOneOut(), method="predict_proba")[:, 1]),
      "loo_auc_composition_plus_score": roc_auc_score(y, cross_val_predict(LogisticRegression(max_iter=1000), np.column_stack([D, CS, COMPO.core_oof_score]), y, cv=LeaveOneOut(), method="predict_proba")[:, 1])})
print(json.dumps(LR, indent=1, default=float))''')

code(r'''# gene by gene: which panel genes keep their PD effect once the composition is removed?
def pooled_g(M):
    num, den = np.zeros(M.shape[1]), 0.0
    for d in sorted(set(DS)):
        m = DS == d; a, b = M[m & (y == 1)], M[m & (y == 0)]; n1, n0 = len(a), len(b)
        sp = np.sqrt(((n1 - 1) * a.var(0, ddof=1) + (n0 - 1) * b.var(0, ddof=1)) / (n1 + n0 - 2))
        J = 1 - 3 / (4 * (n1 + n0) - 9); w = n1 * n0 / (n1 + n0)
        num += w * J * (a.mean(0) - b.mean(0)) / np.where(sp > 1e-9, sp, np.inf); den += w
    return num / den
G0, G1 = pooled_g(XZ), pooled_g(VARIANTS["composition (50 markers)"])
GE = pd.DataFrame({"gene": PANEL, "symbol": [sym(g) for g in PANEL], "direction": ["up" if g in UP else "down" for g in PANEL],
                   "lineage": LIN[PI], "g_original": G0[PI], "g_composition_removed": G1[PI]})
GE["retained"] = GE.g_composition_removed / GE.g_original
GE.sort_values("retained").to_csv(OUT / "composition_gene_effects.csv", index=False)
ALLG = pd.DataFrame({"gene": GENES, "g_original": G0, "g_composition_removed": G1, "lineage": LIN, "panel": ~NOTP})
ALLG.to_csv(OUT / "composition_all_gene_effects.csv", index=False)
print(GE.sort_values("retained").round(2).to_string(index=False))
print(f"panel genes keeping >= half of their PD effect: {int((GE.retained >= 0.5).sum())}/30; "
      f"median retained {GE.retained.median():.2f}; all genes median |g| {np.median(np.abs(G0)):.3f} -> {np.median(np.abs(G1)):.3f}")
json.dump({"gprofiler": CAL, "composition": C_SUM, "logistic": LR, "markers": {"calb1": [sym(GENES[j]) for j in CAL50],
           "sox6": [sym(GENES[j]) for j in SOX50], "purity": PANDA},
           "cv": CV.to_dict("records"), "oof_auc": ADJ.to_dict("records"),
           "genes_retaining_half": int((GE.retained >= 0.5).sum())}, open(OUT / "pathway_composition_summary.json", "w"), indent=1, default=float)
log("analysis done")''')

write_nb(HERE / "PD_LCM_rf_pathway_composition.ipynb", CELLS, "pwc")
