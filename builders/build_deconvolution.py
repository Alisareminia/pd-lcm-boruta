"""Builds PD_LCM_rf_deconvolution.ipynb - reference-based deconvolution of the external bulk cohorts.

The paper estimates how much dopamine-neuron material a bulk sample holds from an average of eight marker
z scores, and the composition adjustment rests on that estimate. A reviewer can reasonably ask whether the
conclusion depends on our own scoring choice. This notebook answers that with established deconvolution
against a single-nucleus reference, and reports the agreement:

  1. Reference - GSE157783, 41,435 single nuclei from human midbrain (Smajic et al. 2022, Brain), with the
     authors' own cell-type labels and donor identifiers: 6 controls and 5 PD, twelve cell types including
     dopaminergic neurons.
  2. Signature matrix - per donor per cell type pseudobulk in counts per million, averaged across donors,
     with markers chosen the way CIBERSORT builds a signature: genes that separate one cell type from the
     rest, capped per type, and condition number checked.
  3. Three estimators, all standard: non-negative least squares; nu support-vector regression, which is the
     CIBERSORT algorithm (Newman et al. 2015); and cross-subject weighted NNLS, which is the idea behind
     MuSiC (Wang et al. 2019) - genes whose expression is consistent across reference donors count more.
  4. The comparison - deconvolved dopamine-neuron proportion against the eight-marker score in every
     external cohort, and the external AUCs recomputed with the deconvolved proportion regressed out
     instead of the marker score.

Nothing here refits the classifier or the panel. Both stay frozen exactly as they were.
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
    """One code block of build_external_multi, verbatim, so the cohorts are parsed exactly as before."""
    i = SRC.index("code(r'''" + marker) + len("code(r'''")
    return SRC[i:SRC.index("''')", i)]


md("""# Deconvolution of the external cohorts against a single-nucleus reference

The composition argument in this work rests on an estimate of how much dopamine-neuron material each bulk
sample contains. In the paper that estimate is the average z score of eight canonical markers
(*TH*, *SLC6A3*, *SLC18A2*, *DDC*, *KCNJ6*, *ALDH1A1*, *NR4A2*, *EN1*). This notebook asks whether an
established deconvolution framework, with a real single-nucleus reference behind it, says the same thing.

**Reference.** GSE157783, 41,435 nuclei of human midbrain from 6 controls and 5 people with Parkinson's
disease, with the authors' cell-type labels (Smajic et al. 2022, *Brain* 145:964-978).

**Estimators.** Non-negative least squares; nu-SVR, the algorithm behind CIBERSORT; and a cross-subject
weighted NNLS in the spirit of MuSiC. All three read the same signature matrix.

**What is fixed.** The classifier, the panel, the thresholds and the marker score are all unchanged. The only
new quantity is the deconvolved proportion, and it is used to repeat an adjustment already reported.""")

code(block("import os, io, re, gzip, glob, json, time, tarfile, urllib.re"))
code("DISC = json.loads(r'''" + json.dumps({}) + "''')   # the overlap check is not repeated here")

md("""## 1. The frozen model, the panel and the discovery gene space

Only what this notebook needs: the locked core classifier as it was saved, the 30-gene panel and the five
panel forests. Nothing is re-selected and nothing is retrained on external data.""")
code(r'''cz = np.load(find_any("core_data.npz", "rf-core"), allow_pickle=True)
X, XR, y, DS = cz["X"], cz["XR"], cz["y"].astype(int), cz["ds"].astype(str)
GENES = [str(g) for g in cz["genes"]]
CORE = joblib.load(find_any("core_model.joblib", "rf-core"))
SYM = pd.read_csv(find_any("15_gene_symbol_map.csv", "rf-core")).set_index("gene")["symbol"].to_dict()
PANEL = pd.read_csv(find_any("04_boruta_selected_genes.csv", "boruta-panel")).gene.tolist()
PI = [GENES.index(g) for g in PANEL]
PANEL_RF = [RandomForestClassifier(n_estimators=1000, max_features="sqrt", class_weight="balanced", oob_score=True,
                                   random_state=42 + k, n_jobs=-1).fit(X[:, PI], y) for k in range(5)]
MODELS = ["frozen", "panel"]
log(f"discovery {len(y)} people x {len(GENES):,} genes; panel {len(PANEL)} genes")''')

md("## 2. The single-nucleus reference (GSE157783)")
code(r'''REF_ACC = "GSE157783"
PRIOR = Path("/kaggle/input/pd-lcm-rf-deconvolution")      # version 1 of this notebook, for its downloads
def ref_file(name):
    cached = PRIOR / name
    if cached.exists():
        print("reusing", cached)
        return cached
    dest = Path(f"/kaggle/working/{name}")
    if not dest.exists():
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{REF_ACC[:-3]}nnn/{REF_ACC}/suppl/{REF_ACC}_{name}"
        for k in range(6):
            try:
                urllib.request.urlretrieve(url, dest); break
            except Exception as exc:
                print("  retry", k + 1, type(exc).__name__); time.sleep(10)
    return dest

def one_member(tar_path):
    with tarfile.open(tar_path) as tf:
        m = [x for x in tf.getmembers() if x.isfile()][0]
        return tf.extractfile(m).read()

CELLS_TSV = pd.read_csv(io.BytesIO(one_member(ref_file("IPDCO_hg_midbrain_cell.tar.gz"))), sep="\t")
GENES_TSV = pd.read_csv(io.BytesIO(one_member(ref_file("IPDCO_hg_midbrain_genes.tar.gz"))), sep="\t", header=None)
print("reference cells:", CELLS_TSV.shape, "| columns:", list(CELLS_TSV.columns))
print(CELLS_TSV.cell_ontology.value_counts().to_string())
print("reference donors:", sorted(CELLS_TSV.patient.unique()))
print("reference genes file:", GENES_TSV.shape, GENES_TSV.head(3).to_dict("list"))''')

code(r'''# the UMI matrix: genes x cells, sparse triplet or dense text depending on the release
UMI_PATH = ref_file("IPDCO_hg_midbrain_UMI.tar.gz")
with tarfile.open(UMI_PATH) as tf:
    members = [m.name for m in tf.getmembers() if m.isfile()]
print("UMI archive holds:", members)
raw = one_member(UMI_PATH)
head = raw[:400].decode("utf8", "replace")
print("first bytes:\n", head[:400])''')

code(r'''# build the counts matrix, whichever layout the archive uses
import scipy.sparse as sp

def load_umi(raw, genes_tsv, cells_tsv):
    txt = io.BytesIO(raw)
    first = raw[:200].decode("utf8", "replace").splitlines()[0]
    if first.startswith("%%MatrixMarket"):
        from scipy.io import mmread
        M = sp.csr_matrix(mmread(txt))
        rows = genes_tsv.iloc[:, -1].astype(str).to_numpy()
        cols = cells_tsv.barcode.to_numpy()
    else:
        D = pd.read_csv(txt, sep="\t", index_col=0)
        M = sp.csr_matrix(D.to_numpy(dtype=np.float32))
        print("   row labels as the matrix carries them:", list(D.index.astype(str)[:4]),
              "->", D.index.nunique(), "distinct for", len(D.index), "rows")
        rows, cols = D.index.astype(str).to_numpy(), D.columns.astype(str).to_numpy()
        # the matrix ships without usable gene names, so take them from the genes table, which is row-ordered
        gt = genes_tsv.copy()
        gt.columns = [str(c) for c in gt.iloc[0]]
        gt = gt.iloc[1:]
        gt["row"] = gt["row"].astype(int)
        names = gt.sort_values("row")["gene"].astype(str).to_numpy()
        if len(names) == M.shape[0]:
            rows = names
            print("   gene names taken from the genes table:", list(rows[:3]), "...", len(set(rows)), "distinct")
        else:
            print("   genes table does not match the matrix:", len(names), "vs", M.shape[0])
    if M.shape[0] != len(rows) and M.shape[1] == len(rows):      # cells x genes
        M, rows, cols = M.T.tocsr(), rows, cols
    print("counts:", M.shape, "| genes:", len(rows), "| cells:", len(cols))
    return M, rows, cols

CNT, RGENE, RCELL = load_umi(raw, GENES_TSV, CELLS_TSV)
LAB = CELLS_TSV.set_index("barcode").reindex(RCELL)
print("labelled cells:", int(LAB.cell_ontology.notna().sum()), "of", len(RCELL))''')

md("""## 3. Pseudobulk per donor per cell type, and the signature matrix

Each cell type is summarised once per reference donor, in counts per million, so that a donor with more
nuclei does not dominate. The signature keeps genes that separate one cell type from the rest, the way a
CIBERSORT signature is built, and the condition number of the resulting matrix is reported.""")

code(r'''keep = LAB.cell_ontology.notna().to_numpy()
CT = LAB.cell_ontology.to_numpy()[keep]
DN = LAB.patient.to_numpy()[keep]
XREF = CNT[:, keep]
TYPES = [t for t, n in pd.Series(CT).value_counts().items() if n >= 40]
print("cell types kept (>= 40 nuclei):", TYPES)

rows = {}
for t in TYPES:
    for d in sorted(set(DN)):
        m = (CT == t) & (DN == d)
        if m.sum() < 10:
            continue
        v = np.asarray(XREF[:, m].sum(1)).ravel()
        if v.sum() == 0:
            continue
        rows[(t, d)] = v / v.sum() * 1e6                      # counts per million of that donor's cells
PB = pd.DataFrame(rows, index=RGENE)
PB = PB.groupby(level=0).sum()                                 # duplicate gene symbols summed
print("pseudobulk:", PB.shape, "| donors per type:", pd.Series([t for t, _ in PB.columns]).value_counts().to_dict())

REF = PB.T.groupby(level=0).mean().T[TYPES]                    # mean profile per cell type
VAR = PB.T.groupby(level=0).var().T[TYPES].fillna(0.0)         # across-donor variance, for the MuSiC weights
print("reference profiles:", REF.shape)''')

code(r'''def signature(REF, per_type=150, min_cpm=5.0):
    """CIBERSORT-style marker choice: per cell type, the genes with the largest fold change over the rest."""
    L = np.log2(REF + 1.0)
    picked = []
    for t in REF.columns:
        rest = L.drop(columns=[t]).max(1)
        fc = L[t] - rest
        ok = (REF[t] >= min_cpm) & (fc > 0.5)
        picked += list(fc[ok].sort_values(ascending=False).head(per_type).index)
    genes = sorted(set(picked))
    S = REF.loc[genes]
    return S, np.linalg.cond(S.to_numpy())

SIG, COND = signature(REF)
assert SIG.shape[0] >= 200, f"signature too small ({SIG.shape[0]} genes) - the reference did not load correctly"
print(f"signature matrix: {SIG.shape[0]} genes x {SIG.shape[1]} cell types, condition number {COND:.1f}")
print("markers per type:", {t: int(((np.log2(SIG + 1).T.idxmax()) == t).sum()) for t in SIG.columns})
SIG.to_csv("/kaggle/working/deconv_signature_matrix.csv")
REF.to_csv("/kaggle/working/deconv_reference_profiles.csv")''')

md("""## 4. The bulk cohorts, on a linear scale

The same eight cohorts, parsed by the same code as `pd-lcm-rf-external-multi`, but kept on the expression
scale the deconvolution needs rather than z scored. The probe-to-gene mapping is widened to cover the
signature genes as well as the 5,622 discovery genes.""")

code(block('def geo_url(acc, sub, f): return f"https://ftp.ncbi.nlm.nih.g').replace(
    "    bg, bm = best(GENES), best(MARKERS + SEXG)",
    "    bg, bm = best(sorted(set(GENES) | set(SIG.index))), best(MARKERS + SEXG)"))

code(block("# ---------------- GSE7621 (HG-U133 Plus 2) - the same file a"))
rna = block("# ---------------- GSE114517 (RNA-seq counts per sample; subs")
rna = rna.replace(
    'SYM2E = {s: sorted(v) for s, v in gconvert(MARKERS + SEXG, "ENSG").items()}',
    'SYM2E = {s: sorted(v) for s, v in gconvert(MARKERS + SEXG, "ENSG").items()}\n'
    'SIG2E = {s: sorted(v) for s, v in gconvert(list(SIG.index), "ENSG").items()}\n'
    'def sig_rows(LG):\n'
    '    """Signature genes as symbol-level rows, so the deconvolution can find them."""\n'
    '    return pd.DataFrame({s: LG.loc[[e for e in es if e in LG.index]].sum(0).to_numpy(float)\n'
    '                         for s, es in SIG2E.items() if any(e in LG.index for e in es)}, index=LG.columns).T\n'
    'def with_sig(LG, base):\n'
    '    D = pd.concat([base, sig_rows(LG)])\n'
    '    return D[~D.index.duplicated()]')
rna = rna.replace('add("GSE114517", LC.reindex([g for g in GENES if g in LC.index]), rna_markers(LC),',
                  'add("GSE114517", with_sig(LC, LC.reindex([g for g in GENES if g in LC.index])), rna_markers(LC),')
rna = rna.replace('GT = np.log2(tx_to_gene(GENES) + 1)',
                  'GTT = pd.concat([tx_to_gene(GENES), tx_to_gene(list(SIG.index))])\n'
                  'GT = np.log2(GTT[~GTT.index.duplicated()] + 1)')
code(rna)

md("""## 5. Scoring every cohort with the frozen models

Exactly the steps of `pd-lcm-rf-external-multi`: genes z scored within the cohort, then ranked within each
person, then the frozen PCA and forest. No cohort is fitted on.""")

code(r'''from scipy.stats import rankdata
zrow = lambda D: ((D.T - D.T.mean()) / D.T.std(ddof=1).clip(lower=0.05)).T
SC = {}
for name, c in COH.items():
    Z = zrow(c["G"]).reindex(GENES).fillna(0.0).T.to_numpy()
    ZR = np.apply_along_axis(rankdata, 1, Z) / Z.shape[1]
    SC[name] = {"frozen": CORE["rf"].predict_proba(CORE["pca"].transform(ZR))[:, 1],
                "panel": np.mean([m.predict_proba(Z[:, PI])[:, 1] for m in PANEL_RF], axis=0)}
    log(f"{name}: {len(c['y'])} donors scored")''')

md("## 6. Deconvolution: NNLS, nu-SVR (CIBERSORT) and cross-subject weighted NNLS (MuSiC)")

code(r'''from scipy.optimize import nnls
from sklearn.svm import NuSVR

def to_linear(G):
    """The cohort matrices are log2; deconvolution needs the linear scale, in counts per million."""
    L = np.power(2.0, G.astype(float)) - 1.0
    L = L.clip(lower=0.0)
    return L / L.sum(0).replace(0, np.nan) * 1e6

def fit_nnls(S, y, w=None):
    A, b = S.to_numpy(float), y.to_numpy(float)
    if w is not None:
        A, b = A * w[:, None], b * w
    x, _ = nnls(A, b)
    return x / x.sum() if x.sum() > 0 else x

def fit_nusvr(S, y):
    """The CIBERSORT estimator: nu-SVR over three nu, the fit with the lowest RMSE, weights clipped at zero."""
    A = ((S - S.values.mean()) / S.values.std()).to_numpy(float)
    b = ((y - y.mean()) / (y.std() if y.std() else 1.0)).to_numpy(float)
    best, berr = None, np.inf
    for nu in (0.25, 0.5, 0.75):
        m = NuSVR(nu=nu, C=1.0, kernel="linear").fit(A, b)
        c = m.coef_.ravel()
        err = float(np.sqrt(np.mean((A @ c - b) ** 2)))
        if err < berr:
            best, berr = c, err
    c = np.clip(best, 0, None)
    return (c / c.sum() if c.sum() > 0 else c), berr

def deconvolve(G, SIG, VARm):
    """Every donor of one cohort, by all three estimators. Returns three frames of proportions."""
    lin = to_linear(G)
    genes = [g for g in SIG.index if g in lin.index]
    S = SIG.loc[genes]
    V = VARm.loc[genes]
    w = 1.0 / np.sqrt(V.mean(1).to_numpy() + 1.0)              # MuSiC: steady genes weigh more
    out = {}
    for name in ("nnls", "nusvr", "music"):
        out[name] = pd.DataFrame(index=lin.columns, columns=S.columns, dtype=float)
    rmse = {}
    for s in lin.columns:
        y = lin[s].reindex(genes).fillna(0.0)
        out["nnls"].loc[s] = fit_nnls(S, y)
        out["music"].loc[s] = fit_nnls(S, y, w=w)
        c, e = fit_nusvr(S, y)
        out["nusvr"].loc[s] = c
        rmse[s] = e
    return out, len(genes), rmse

DEC, SIGCOV, RMSE = {}, {}, {}
for name, c in COH.items():
    DEC[name], SIGCOV[name], RMSE[name] = deconvolve(c["G"], SIG, VAR)
    p = DEC[name]["nusvr"]
    da = p["DaNs"] if "DaNs" in p.columns else pd.Series(np.nan, index=p.index)
    print(f"{name}: {SIGCOV[name]} signature genes measured; median DaN fraction {100 * da.median():.2f}%")''')

md("## 7. Does the deconvolution agree with the eight-marker score?")

code(r'''from scipy.stats import spearmanr, pearsonr
rows = []
for name, c in COH.items():
    mk = zrow(c["Mk"].loc[[m for m in MARKERS if m in c["Mk"].index]]).mean(0)
    for meth in ("nnls", "nusvr", "music"):
        p = DEC[name][meth]
        if "DaNs" not in p.columns:
            continue
        da = p["DaNs"].astype(float).to_numpy()
        rho, pv = spearmanr(da, mk.to_numpy())
        neuronal = p[[t for t in ("DaNs", "Excitatory", "Inhibitory", "GABA", "CADPS2+ neurons") if t in p.columns]].sum(1)
        rho_n, pv_n = spearmanr(neuronal.to_numpy(), mk.to_numpy())
        rows.append(dict(cohort=name, method=meth, n=len(da), signature_genes=SIGCOV[name],
                         da_fraction_median=float(np.median(da)), rho_marker_vs_DaN=rho, p_DaN=pv,
                         rho_marker_vs_all_neurons=rho_n, p_all_neurons=pv_n,
                         oligodendrocyte=float(p["Oligodendrocytes"].median()) if "Oligodendrocytes" in p else np.nan,
                         astrocyte=float(p["Astrocytes"].median()) if "Astrocytes" in p else np.nan,
                         microglia=float(p["Microglia"].median()) if "Microglia" in p else np.nan))
AGREE = pd.DataFrame(rows)
AGREE.to_csv("/kaggle/working/deconv_agreement_by_cohort.csv", index=False)
print(AGREE.round(3).to_string(index=False))''')

code(r'''# pooled across cohorts, each donor once, marker score standardised within cohort
pool = []
for name, c in COH.items():
    mk = zrow(c["Mk"].loc[[m for m in MARKERS if m in c["Mk"].index]]).mean(0)
    for meth in ("nnls", "nusvr", "music"):
        p = DEC[name][meth]
        if "DaNs" not in p.columns:
            continue
        z = lambda v: (v - np.mean(v)) / (np.std(v) if np.std(v) else 1.0)
        pool.append(pd.DataFrame(dict(cohort=name, method=meth, donor=p.index,
                                      marker_z=z(mk.to_numpy()), da_z=z(p["DaNs"].astype(float).to_numpy()),
                                      y=c["y"])))
POOL = pd.concat(pool, ignore_index=True)
POOL.to_csv("/kaggle/working/deconv_pooled_donor_estimates.csv", index=False)
SUM = {}
for meth, d in POOL.groupby("method"):
    rho, pv = spearmanr(d.marker_z, d.da_z)
    r, rp = pearsonr(d.marker_z, d.da_z)
    SUM[meth] = dict(n=int(len(d)), spearman=float(rho), spearman_p=float(pv), pearson=float(r), pearson_p=float(rp))
    print(f"{meth:6s} n={len(d):4d}  Spearman rho={rho:+.3f} (P={pv:.2g})  Pearson r={r:+.3f}")''')

md("""## 8. PD against control on the deconvolved proportions, and the adjustment repeated

Two questions. Does the deconvolution see the neuron loss that the marker score sees? And if the external
scores are residualised on the deconvolved dopamine-neuron proportion instead of on the marker score, do the
AUCs land in the same place?""")

code(r'''from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
rows = []
for name, c in COH.items():
    yv = np.asarray(c["y"], int)
    for meth in ("nnls", "nusvr", "music"):
        p = DEC[name][meth]
        for t in p.columns:
            v = p[t].astype(float).to_numpy()
            if np.allclose(v, v[0]) or len(set(yv)) < 2:
                continue
            rows.append(dict(cohort=name, method=meth, cell_type=t, n=len(yv),
                             median_control=float(np.median(v[yv == 0])), median_PD=float(np.median(v[yv == 1])),
                             auc=float(roc_auc_score(yv, v)),
                             p=float(mannwhitneyu(v[yv == 1], v[yv == 0], alternative="two-sided").pvalue)))
CTPD = pd.DataFrame(rows)
CTPD.to_csv("/kaggle/working/deconv_celltype_pd_vs_control.csv", index=False)
print(CTPD[CTPD.method == "nusvr"].pivot_table(index="cell_type", values="auc", aggfunc="median").round(3).to_string())''')

code(r'''def resid(v, x):
    """v with x projected out, within one cohort."""
    X = np.c_[np.ones(len(x)), np.asarray(x, float)]
    b, *_ = np.linalg.lstsq(X, np.asarray(v, float), rcond=None)
    return np.asarray(v, float) - X @ b

rows = []
for name, c in COH.items():
    yv = np.asarray(c["y"], int)
    if len(set(yv)) < 2:
        continue
    mk = zrow(c["Mk"].loc[[m for m in MARKERS if m in c["Mk"].index]]).mean(0).to_numpy()
    da = DEC[name]["nusvr"]["DaNs"].astype(float).to_numpy() if "DaNs" in DEC[name]["nusvr"] else None
    neu = DEC[name]["nusvr"][[t for t in ("DaNs", "Excitatory", "Inhibitory", "GABA", "CADPS2+ neurons")
                              if t in DEC[name]["nusvr"].columns]].sum(1).to_numpy()
    for model in MODELS:
        s = np.asarray(SC[name][model], float)
        rows.append(dict(cohort=name, model=model, n=len(yv),
                         auc=roc_auc_score(yv, s),
                         auc_marker_removed=roc_auc_score(yv, resid(s, mk)),
                         auc_DaN_removed=roc_auc_score(yv, resid(s, da)) if da is not None else np.nan,
                         auc_all_neurons_removed=roc_auc_score(yv, resid(s, neu))))
ADJ = pd.DataFrame(rows)
ADJ.to_csv("/kaggle/working/deconv_adjusted_auc_by_cohort.csv", index=False)
print(ADJ.round(3).to_string(index=False))''')

code(r'''# pooled, weighted by cohort size, the three adjustments side by side
out = {}
for model, d in ADJ.groupby("model"):
    w = d.n / d.n.sum()
    out[model] = {c: float((d[c] * w).sum()) for c in
                  ("auc", "auc_marker_removed", "auc_DaN_removed", "auc_all_neurons_removed")}
POOLED_ADJ = pd.DataFrame(out).T
POOLED_ADJ.to_csv("/kaggle/working/deconv_adjusted_auc_pooled.csv")
print(POOLED_ADJ.round(3).to_string())

SUMMARY = dict(reference=dict(accession=REF_ACC, nuclei=int(len(CT)), donors=int(len(set(DN))),
                              cell_types=list(SIG.columns), signature_genes=int(SIG.shape[0]),
                              condition_number=float(COND)),
               agreement=SUM,
               signature_genes_measured={k: int(v) for k, v in SIGCOV.items()},
               pooled_adjusted_auc=POOLED_ADJ.to_dict())
json.dump(SUMMARY, open("/kaggle/working/deconv_summary.json", "w"), indent=1)
print(json.dumps(SUMMARY["agreement"], indent=1))''')

md("""## 9. What this notebook is allowed to conclude

The reference holds few dopaminergic nuclei (74 of 41,435), which is what human midbrain single-nucleus data
looks like; a proportion estimated from so small a profile is noisy, and the honest reading is the agreement
between methods and with the marker score, not the absolute percentage. Everything written to
`/kaggle/working` is a table the manuscript can cite directly.""")

write_nb(HERE / "PD_LCM_rf_deconvolution.ipynb", CELLS, "dec")
