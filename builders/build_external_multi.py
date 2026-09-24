"""Builds PD_LCM_rf_external_multi.ipynb - external validation across several independent bulk nigra cohorts.

A companion to pd-lcm-rf-external (GSE7621 only), which stays as it is. Adds what a reviewer asks for:
  1. more independent cohorts - seven public bulk substantia nigra cohorts (5 arrays, 2 RNA-seq), plus
     early-stage (incidental Lewy body) donors;
  2. a donor-overlap check - public donor IDs matched against the discovery studies (and checked for
     diagnosis and sex), shared donors removed; and every cohort also scored by a strictly independent
     model, retrained without any discovery study that could come from the same brain bank;
  3. calls, not only ranking - accuracy, sensitivity and specificity at thresholds fixed on the
     discovery people before any external cohort is seen.
Downloads happen on Kaggle, from the GEO FTP site. Nothing is fitted on an external cohort.
"""
import json, pathlib
import pandas as pd
from kaggle_nb import write_nb
HERE = pathlib.Path(__file__).parent
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

# discovery donors (ID -> diagnosis, sex called from expression), for the ID-based overlap check
FP = pd.read_csv(HERE.parent / "data/lcm/harmonised/fingerprints.csv")
DISC = {d: {str(r.person): [int(r.y), None if pd.isna(r.female) else int(r.female)] for r in FP[FP.dataset == d].itertuples()}
        for d in ("GSE20141", "GSE182622")}

md("""# External validation across independent bulk substantia nigra cohorts

A companion to `pd-lcm-rf-external` (GSE7621 only), which is left as it is. This notebook adds:

1. **More cohorts** - GSE7621, GSE20292, GSE20163, GSE20164, GSE8397, GSE49036 (arrays) and GSE114517, GSE168496
   (RNA-seq); GSE49036 also gives early-stage donors (incidental Lewy body disease, Braak 1-4).
2. **Donor overlap** - where donor IDs are public they are matched against the discovery studies (and the matches
   checked for diagnosis and sex); shared donors are removed. Every cohort is also scored by a **strictly
   independent model** - the same locked pipeline, retrained without any discovery study that could come from the
   same brain bank.
3. **Calls, not only ranking** - accuracy, sensitivity and specificity at two thresholds fixed on the discovery
   people before any external cohort is seen: 0.5, and the out-of-bag optimum of each forest.

**The model was locked before any of these cohorts was examined.** The core classifier (within-person ranks ->
PCA 30 -> Random Forest) was chosen by a sweep that used only the 63 discovery people (`pd-lcm-rf-confirm`). Nothing
is fitted on an external cohort: each goes through the same label-free steps as discovery (genes z-scored within the
cohort, then ranked within each person) and is scored once.""")

code(r'''import os, io, re, gzip, glob, json, time, tarfile, urllib.request, warnings
from pathlib import Path
import numpy as np, pandas as pd
import joblib
from scipy.stats import rankdata, spearmanr, binomtest
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, balanced_accuracy_score
warnings.filterwarnings("ignore")
OUT = Path("/kaggle/working"); GEO = OUT / "geo"; GEO.mkdir(exist_ok=True)
def find_any(pattern, key):
    hits = sorted((h for h in glob.glob(f"/kaggle/input/**/{pattern}", recursive=True) if key in h), key=len)
    if not hits:
        raise FileNotFoundError(f"{pattern} ({key})")
    return hits[0]
t0 = time.time()
def log(m): print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)
pd.set_option("display.width", 230); pd.set_option("display.max_colwidth", 70); pd.set_option("display.max_columns", 40)''')

code("DISC = json.loads(r'''" + json.dumps(DISC) + "''')   # discovery donor -> [PD, female (from expression)]")

md("## 1. Discovery: the frozen core model, its pre-specified thresholds, the strictly independent models")

code(r'''cz = np.load(find_any("core_data.npz", "rf-core"), allow_pickle=True)
X, XR, y, DS = cz["X"], cz["XR"], cz["y"].astype(int), cz["ds"].astype(str)
GENES = [str(g) for g in cz["genes"]]
CORE = joblib.load(find_any("core_model.joblib", "rf-core"))
SYM = pd.read_csv(find_any("15_gene_symbol_map.csv", "rf-core")).set_index("gene")["symbol"].to_dict()
PANEL = pd.read_csv(find_any("04_boruta_selected_genes.csv", "boruta-panel")).gene.tolist()
DE = pd.read_csv(find_any("03_de_results_full.csv", "boruta-panel")).set_index("gene")
PI = [GENES.index(g) for g in PANEL]

def oob_threshold(s, yt):
    """The accuracy cut-off on out-of-bag scores of the training people (the rule used in every discovery fold)."""
    u = np.unique(np.round(s, 6)); cuts = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])
    acc = [accuracy_score(yt, (s > c).astype(int)) for c in cuts]
    best = np.flatnonzero(np.isclose(acc, max(acc)))
    return float(cuts[best[len(best) // 2]])
def fit_core(mask):
    """The locked pipeline (within-person ranks -> PCA 30 -> Random Forest) on a subset of the discovery people."""
    p = PCA(min(30, int(mask.sum()) - 1), random_state=42).fit(XR[mask])        # 30 components, fewer only if too few people
    rf = RandomForestClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1, class_weight="balanced",
                                oob_score=True, random_state=42, n_jobs=-1).fit(p.transform(XR[mask]), y[mask])
    return {"pca": p, "rf": rf, "thr": oob_threshold(rf.oob_decision_function_[:, 1], y[mask]), "n": int(mask.sum()),
            "components": p.n_components_}
CORE["thr"], CORE["n"] = oob_threshold(CORE["rf"].oob_decision_function_[:, 1], y), len(y)
# discovery studies that could share a brain bank with an external cohort
RELATED = {"harvard": ["GSE20141", "GSE24378"], "nbb": ["GSE182622"]}
STRICT = {k: fit_core(~np.isin(DS, v)) for k, v in RELATED.items()}
STRICT["none"] = CORE
PANEL_RF = [RandomForestClassifier(n_estimators=1000, max_features="sqrt", class_weight="balanced", oob_score=True,
                                   random_state=42 + k, n_jobs=-1).fit(X[:, PI], y) for k in range(5)]
PANEL_THR = oob_threshold(np.mean([m.oob_decision_function_[:, 1] for m in PANEL_RF], axis=0), y)
log(f"discovery {len(y)} people x {len(GENES):,} genes")
log("strictly independent models: " + ", ".join(f"without {'+'.join(v)} ({STRICT[k]['n']} people, {STRICT[k]['components']} components)" for k, v in RELATED.items()))

# the panel, made strictly independent the same way: Boruta (the reported settings) re-run without the related studies
for _a, _t in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _a):
        setattr(np, _a, _t)
try:
    from boruta import BorutaPy
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "Boruta==0.4.3"], check=True)
    from boruta import BorutaPy
def fit_panel(mask):
    """Boruta (500 trees, perc 99, 100 iterations) on a subset of the discovery people, then five forests on its genes."""
    idx = np.flatnonzero(mask)
    b = BorutaPy(RandomForestClassifier(max_features="sqrt", class_weight="balanced", n_jobs=-1), n_estimators=500, perc=99,
                 alpha=0.05, two_step=True, max_iter=100, random_state=42, verbose=0).fit(X[idx], y[idx])
    genes = list(np.flatnonzero(b.support_)) or list(np.flatnonzero(b.support_ | b.support_weak_))
    rfs = [RandomForestClassifier(n_estimators=1000, max_features="sqrt", class_weight="balanced", oob_score=True,
                                  random_state=42 + k, n_jobs=-1).fit(X[idx][:, genes], y[idx]) for k in range(5)]
    return {"genes": genes, "rfs": rfs, "n": len(idx),
            "thr": oob_threshold(np.mean([m.oob_decision_function_[:, 1] for m in rfs], axis=0), y[idx])}
PANELS = {"none": {"genes": PI, "rfs": PANEL_RF, "n": len(y), "thr": PANEL_THR}}
for k, v in RELATED.items():
    PANELS[k] = fit_panel(~np.isin(DS, v))
    log(f"strict panel without {'+'.join(v)}: {len(PANELS[k]['genes'])} genes, {len(set(PANELS[k]['genes']) & set(PI))} of them "
        f"in the reported 30-gene panel: " + ", ".join(SYM.get(GENES[j], GENES[j]) for j in PANELS[k]["genes"]))
THRESHOLDS = {"core": CORE["thr"], **{f"core_strict_{k}": STRICT[k]["thr"] for k in RELATED},
              "panel": PANEL_THR, **{f"panel_strict_{k}": PANELS[k]["thr"] for k in RELATED}}
log("pre-specified out-of-bag thresholds: " + ", ".join(f"{k} {v:.3f}" for k, v in THRESHOLDS.items()))''')

md("## 2. Download and parse the cohorts (on Kaggle, from GEO)")

code(r'''def geo_url(acc, sub, f): return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{acc[:-3]}nnn/{acc}/{sub}/{f}"
def fetch(acc, sub, f):
    dest = GEO / f
    for k in range(6):
        try:
            if not dest.exists():
                urllib.request.urlretrieve(geo_url(acc, sub, f), dest)
            return dest
        except Exception as exc:
            print("  retry", k + 1, type(exc).__name__); dest.unlink(missing_ok=True); time.sleep(10)
    raise RuntimeError(f"could not download {f}")
def read_matrix(path):
    op = gzip.open if str(path).endswith(".gz") else open
    lines = op(path, "rt", encoding="utf-8", errors="replace").read().split("\n")
    meta = {}
    for l in lines:
        if l.startswith("!Sample_"):
            k = l.split("\t")[0]; meta.setdefault(k, []).append([v.strip().strip('"') for v in l.split("\t")[1:]])
    b0 = next((i for i, l in enumerate(lines) if "!series_matrix_table_begin" in l), None)
    b1 = next((i for i, l in enumerate(lines) if "!series_matrix_table_end" in l), None)
    M = None
    if b0 is not None and b1 - b0 > 2:
        M = pd.read_csv(io.StringIO("\n".join(lines[b0 + 1:b1])), sep="\t", index_col=0).apply(pd.to_numeric, errors="coerce")
    return meta, M
def series_matrix(acc, fname=None): return read_matrix(fetch(acc, "matrix", fname or f"{acc}_series_matrix.txt.gz"))
def chars(meta, *keys):
    """One characteristic across samples, whichever characteristics row holds it (first key found)."""
    out = [None] * len(meta["!Sample_geo_accession"][0])
    for row in meta.get("!Sample_characteristics_ch1", []):
        for i, v in enumerate(row):
            for key in keys:
                if out[i] is None and v.lower().startswith(key.lower() + ":"):
                    out[i] = v.split(":", 1)[1].strip()
    return out
def gconvert(ids, target, batch=2500):
    out = {}; ids = list(ids)
    for s in range(0, len(ids), batch):
        body = {"organism": "hsapiens", "target": target, "query": ids[s:s + batch]}
        for k in range(6):
            try:
                req = urllib.request.Request("https://biit.cs.ut.ee/gprofiler/api/convert/convert/", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=300) as fh:
                    for rec in json.loads(fh.read())["result"]:
                        c = rec.get("converted")
                        if c and c not in ("None", "N/A"):
                            out.setdefault(rec["incoming"], set()).add(c)
                break
            except Exception as exc:
                print("  g:Convert retry", k + 1, type(exc).__name__); time.sleep(10)
    return out
MARKERS = ["TH", "SLC6A3", "SLC18A2", "DDC", "KCNJ6", "ALDH1A1", "NR4A2", "EN1"]        # dopamine-neuron content
SEXG = ["XIST", "RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY"]                             # sex, from expression
def array_genes(M, namespace):
    """Probe matrix -> gene matrix (each gene's highest-mean probe), log2 unless already logged.
    HG-U133A probe sets all exist, under the same IDs, on HG-U133 Plus 2, so both use the Plus 2 mapping."""
    L = np.log2(M.clip(lower=1)) if np.nanmax(M.values) > 50 else M.copy()
    pm = L.mean(1)
    def best(ids):
        conv = gconvert(ids, namespace)
        return {g: max((p for p in ps if p in L.index), key=lambda p: pm[p]) for g, ps in conv.items() if any(p in L.index for p in ps)}
    bg, bm = best(GENES), best(MARKERS + SEXG)
    G = pd.DataFrame({g: L.loc[p].to_numpy(float) for g, p in bg.items()}, index=L.columns).T
    Mk = pd.DataFrame({s: L.loc[p].to_numpy(float) for s, p in bm.items()}, index=L.columns).T
    return G, Mk
def sex_from_expression(Mk):
    """Female = low Y-chromosome genes and high XIST; split at the widest gap of the combined score."""
    z = lambda D: ((D.T - D.T.mean()) / D.T.std(ddof=1).clip(lower=0.05)).T
    yg = [g for g in SEXG[1:] if g in Mk.index]
    s = z(Mk.loc[yg]).mean(0).to_numpy() - (z(Mk.loc[["XIST"]]).mean(0).to_numpy() if "XIST" in Mk.index else 0)
    v = np.sort(s); i = int(np.argmax(np.diff(v)))
    return (s < (v[i] + v[i + 1]) / 2).astype(int)
def meta_sex(vals):
    return [None if v is None else (1 if v.strip().lower().startswith("f") else 0) for v in vals]
COH = {}
def add(name, G, Mk, yv, donors, note, sex=None, stage=None):
    yv, donors = np.asarray(yv, int), np.asarray(donors, str)
    COH[name] = dict(G=G, Mk=Mk, y=yv, donor=donors, note=note, sex_meta=sex if sex is not None else [None] * len(yv),
                     sex_expr=sex_from_expression(Mk), stage=stage if stage is not None else ["PD" if v else "control" for v in yv])
    log(f"{name}: {len(yv)} people ({int((yv == 0).sum())} control, {int(yv.sum())} PD); {G.shape[0]:,}/{len(GENES):,} genes; "
        f"{sum(m in Mk.index for m in MARKERS)}/8 neuron markers  [{note}]")''')

code(r'''# ---------------- GSE7621 (HG-U133 Plus 2) - the same file as pd-lcm-rf-external ----------------
meta, M = read_matrix(find_any("GSE7621_series_matrix.txt*", "externalvalidation2"))
tit = meta["!Sample_title"][0]
G, Mk = array_genes(M, "AFFY_HG_U133_PLUS_2")
add("GSE7621", G, Mk, [0 if "normal" in t.lower() else 1 for t in tit], tit, "Plus 2", sex=meta_sex(chars(meta, "gender", "sex")))

# ---------------- GSE20292 (HG-U133A); donor numbers shared with discovery GSE20141 ----------------
meta, M = series_matrix("GSE20292")
tit = meta["!Sample_title"][0]; dx = chars(meta, "disease state")
G, Mk = array_genes(M, "AFFY_HG_U133_PLUS_2")
add("GSE20292", G, Mk, [0 if d.lower().startswith("control") else 1 for d in dx], [t.split()[0] for t in tit], "U133A",
    sex=meta_sex(chars(meta, "gender")))

# ---------------- GSE20163 (HG-U133A) ----------------
meta, M = series_matrix("GSE20163")
tit = meta["!Sample_title"][0]; src = meta["!Sample_source_name_ch1"][0]
G, Mk = array_genes(M, "AFFY_HG_U133_PLUS_2")
add("GSE20163", G, Mk, [0 if "control" in s.lower() else 1 for s in src], [t.split("_")[0] for t in tit], "U133A")

# ---------------- GSE20164 (HG-U133A) ----------------
meta, M = series_matrix("GSE20164")
tit = meta["!Sample_title"][0]; src = meta["!Sample_source_name_ch1"][0]
G, Mk = array_genes(M, "AFFY_HG_U133_PLUS_2")
add("GSE20164", G, Mk, [0 if "control" in s.lower() else 1 for s in src], tit, "U133A", sex=meta_sex(chars(meta, "gender")))

# ---------------- GSE8397 (HG-U133A chip): lateral and medial nigra averaged per case; frontal cortex left out ----------------
meta, M = series_matrix("GSE8397", "GSE8397-GPL96_series_matrix.txt.gz")
tit = meta["!Sample_title"][0]; ag = chars(meta, "age")
keep = [i for i, t in enumerate(tit) if "substantia nigra" in t.lower()]
def case_no(t):
    m = re.search(r"case\s*(\d+)", t, re.I) or re.search(r"(\d+)[^\d]*-\s*[AB]\s*chip", t, re.I) or re.search(r"(\d+)", t)
    return int(m.group(1))
print("   GSE8397 nigra titles:", [tit[i] for i in keep])
case = [("PD" if "parkinson" in tit[i].lower() else "C") + "-" + str(case_no(tit[i])) for i in keep]
sexc = {c: (1 if "gender: f" in (ag[i] or "").lower() else 0) for c, i in zip(case, keep)}
L = M.iloc[:, keep].copy(); L.columns = case
L = L.T.groupby(level=0).mean().T                                          # one profile per person
print("   GSE8397 nigra samples per case:", pd.Series(case).value_counts().value_counts().to_dict())
G, Mk = array_genes(L, "AFFY_HG_U133_PLUS_2")
add("GSE8397", G, Mk, [1 if c.startswith("PD") else 0 for c in L.columns], list(L.columns), "U133A, lateral+medial SN averaged",
    sex=[sexc[c] for c in L.columns])

# ---------------- GSE49036 (Plus 2), Netherlands Brain Bank: control vs PD; ILBD kept for the early-stage test ----------------
meta, M = series_matrix("GSE49036")
tit = meta["!Sample_title"][0]; dx = chars(meta, "disease state"); braak = chars(meta, "braak stage")
G, Mk = array_genes(M, "AFFY_HG_U133_PLUS_2")
lab = np.array([0 if d.lower() == "control" else (1 if "parkinson" in d.lower() else 2) for d in dx])   # 2 = incidental Lewy body
print("   GSE49036 diagnosis x Braak:", pd.crosstab(np.array(dx), np.array(braak)).to_dict())
m = lab < 2
add("GSE49036", G.loc[:, m], Mk.loc[:, m], lab[m], np.array(tit)[m], "Plus 2, NBB")
EARLY_RAW = dict(G=G, Mk=Mk, lab=lab, braak=np.array(braak), donor=np.array(tit))''')

code(r'''# ---------------- GSE114517 (RNA-seq counts per sample; substantia nigra only; PD with dementia) ----------------
meta, _ = series_matrix("GSE114517")
gsm = meta["!Sample_geo_accession"][0]; tit = meta["!Sample_title"][0]
tissue = chars(meta, "tissue"); st = chars(meta, "subject status", "disease state"); sx = chars(meta, "gender", "sex")
sn = [i for i, t in enumerate(tissue) if t and "substantia" in t.lower()]
cols = {}
with tarfile.open(fetch("GSE114517", "suppl", "GSE114517_RAW.tar")) as tf:
    for mem in tf.getmembers():
        g = mem.name.split("_")[0]
        if g in {gsm[i] for i in sn}:
            raw = tf.extractfile(mem).read()
            txt = (gzip.decompress(raw) if mem.name.endswith(".gz") else raw).decode()
            s = pd.read_csv(io.StringIO(txt), sep="\t", header=None, index_col=0)[1]
            s.index = s.index.astype(str).str.split(".").str[0]
            cols[g] = s.groupby(level=0).sum()
C = pd.DataFrame(cols)[[gsm[i] for i in sn]].fillna(0)
C = C[~C.index.str.startswith("__")]
print(f"   GSE114517: {C.shape[1]} nigra libraries, median {np.median(C.sum(0)) / 1e6:.1f} M counted reads")
LC = np.log2(C / C.sum(0) * 1e6 + 1)
SYM2E = {s: sorted(v) for s, v in gconvert(MARKERS + SEXG, "ENSG").items()}
def rna_markers(LG):
    return pd.DataFrame({s: LG.loc[[e for e in es if e in LG.index]].sum(0).to_numpy(float) for s, es in SYM2E.items()
                         if any(e in LG.index for e in es)}, index=LG.columns).T
add("GSE114517", LC.reindex([g for g in GENES if g in LC.index]), rna_markers(LC),
    [0 if "control" in st[i].lower() else 1 for i in sn], [re.search(r"\[(.*?)\]", tit[i]).group(1) for i in sn],
    "RNA-seq, PD with dementia", sex=meta_sex([sx[i] for i in sn]))

# ---------------- GSE168496 (RNA-seq, transcript level; Netherlands Brain Bank IDs) ----------------
meta, _ = series_matrix("GSE168496")
tit = meta["!Sample_title"][0]; dx = chars(meta, "disease state")
T = pd.read_csv(fetch("GSE168496", "suppl", "GSE168496_all_samples_preprocessed_data.tsv.gz"), sep="\t", index_col=0)
T.index = T.index.astype(str).str.split(".").str[0]
T = T.groupby(level=0).sum()[tit]
def tx_to_gene(ids):
    conv = gconvert(ids, "ENST", batch=1500)
    return pd.DataFrame({g: T.loc[[t for t in ts if t in T.index]].sum(0).to_numpy(float) for g, ts in conv.items()
                         if any(t in T.index for t in ts)}, index=T.columns).T
GT = np.log2(tx_to_gene(GENES) + 1)
MT = np.log2(tx_to_gene(MARKERS + SEXG) + 1)
print(f"   GSE168496: {T.shape[0]:,} transcripts, {T.shape[1]} people; column sums {T.sum(0).min():,.0f}-{T.sum(0).max():,.0f}")
add("GSE168496", GT, MT, [0 if "control" in d.lower() else 1 for d in dx], tit, "RNA-seq, NBB", sex=meta_sex(chars(meta, "gender")))
COV = pd.DataFrame([{"cohort": n, "platform": c["note"], "people": len(c["y"]), "control": int((c["y"] == 0).sum()), "PD": int(c["y"].sum()),
                     "discovery_genes_measured": c["G"].shape[0], "share": c["G"].shape[0] / len(GENES),
                     "panel_genes_measured": int(np.isin(PANEL, c["G"].index).sum()),
                     "neuron_markers": int(np.isin(MARKERS, c["Mk"].index).sum())} for n, c in COH.items()])
COV.to_csv(OUT / "cohorts.csv", index=False); print(COV.round(3).to_string(index=False))''')

md("## 3. Donor overlap with the discovery studies")

code(r'''def nbb_key(s):
    """Netherlands Brain Bank IDs written 'PD-03-43' or '2003-043' -> (year, number)."""
    m = re.search(r"(\d{2,4})-(\d{1,3})$", s)
    return None if not m else (int(m.group(1)) % 100, int(m.group(2)))
HARV = {re.sub(r"\D", "", k): (k, v) for k, v in DISC["GSE20141"].items()}
NBB = {nbb_key(k): (k, v) for k, v in DISC["GSE182622"].items()}
FAMILY = {"GSE20292": "harvard", "GSE20163": "harvard", "GSE20164": "harvard", "GSE49036": "nbb", "GSE168496": "nbb"}
DETAIL, OVER = [], []
for name, c in COH.items():
    if name in ("GSE20292", "GSE20163"):
        match = [HARV.get(re.sub(r"\D", "", d)) for d in c["donor"]]; basis = "donor number = GSE20141"
    elif name == "GSE168496":
        match = [NBB.get(nbb_key(d)) for d in c["donor"]]; basis = "brain-bank ID = GSE182622"
    else:
        match = [None] * len(c["donor"])
        basis = "no donor IDs in a comparable format" if name in FAMILY else "different brain bank"
    c["overlap"] = np.array([m is not None for m in match])
    c["strict"] = FAMILY.get(name, "none")
    for i, m in enumerate(match):
        if m is not None:
            DETAIL.append({"cohort": name, "external_id": c["donor"][i], "discovery_id": m[0],
                           "diagnosis_agrees": int(c["y"][i]) == m[1][0], "sex_metadata": c["sex_meta"][i],
                           "sex_expression": int(c["sex_expr"][i]), "sex_discovery_expression": m[1][1]})
    OVER.append({"cohort": name, "people": len(c["donor"]), "shared_with_discovery": int(c["overlap"].sum()), "basis": basis,
                 "strict_model_trained_without": " + ".join(RELATED.get(c["strict"], [])) or "(same as frozen)"})
# donors that appear in two external cohorts (GSE20163 and GSE20292) are counted once when cohorts are pooled
seen = {re.sub(r"\D", "", d) for d, o in zip(COH["GSE20292"]["donor"], COH["GSE20292"]["overlap"]) if not o}
for name, c in COH.items():
    c["pool"] = ~c["overlap"]
    if name == "GSE20163":
        c["pool"] &= np.array([re.sub(r"\D", "", d) not in seen for d in c["donor"]])
OV = pd.DataFrame(OVER); OV["also_in_another_external_cohort"] = [int((~c["pool"] & ~c["overlap"]).sum()) for c in COH.values()]
DET = pd.DataFrame(DETAIL)
OV.to_csv(OUT / "donor_overlap.csv", index=False); DET.to_csv(OUT / "donor_overlap_matches.csv", index=False)
print(OV.to_string(index=False)); print(); print(DET.to_string(index=False))''')

md("## 4. Every cohort: frozen and strictly independent models, neuron content, calls at the fixed thresholds")

code(r'''zrow = lambda D: ((D.T - D.T.mean()) / D.T.std(ddof=1).clip(lower=0.05)).T
rng = np.random.default_rng(42)
def boot_ci(yv, s, n=4000):
    bs = [roc_auc_score(yv[i], s[i]) for i in (rng.integers(0, len(yv), len(yv)) for _ in range(n)) if len(set(yv[i])) == 2]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)
def perm_p(yv, s, n=10000):
    a = roc_auc_score(yv, s); null = np.array([roc_auc_score(rng.permutation(yv), s) for _ in range(n)])
    return (np.sum(null >= a) + 1) / (n + 1)
def prepare(G, Mk):
    Z = zrow(G).reindex(GENES).fillna(0.0).T.to_numpy()
    return Z, np.apply_along_axis(rankdata, 1, Z) / Z.shape[1], zrow(Mk.loc[[m for m in MARKERS if m in Mk.index]]).mean(0).to_numpy()
def score(model, ZR): return model["rf"].predict_proba(model["pca"].transform(ZR))[:, 1]
loo = lambda F, yv: roc_auc_score(yv, cross_val_predict(LogisticRegression(), F, yv, cv=LeaveOneOut(), method="predict_proba")[:, 1])
def calls(yv, s, t):
    yh = (s > t).astype(int)
    return {"accuracy": accuracy_score(yv, yh), "sensitivity": recall_score(yv, yh), "specificity": recall_score(1 - yv, 1 - yh),
            "balanced_accuracy": balanced_accuracy_score(yv, yh)}
def pscore(P, Z): return np.mean([m.predict_proba(Z[:, P["genes"]])[:, 1] for m in P["rfs"]], axis=0)
MODELS = ["frozen", "strict", "panel", "panel_strict"]       # core classifier as reported / strictly independent; panel forest likewise
ROWS, PEOPLE, SC = [], [], {}
for name, c in COH.items():
    k = ~c["overlap"]; fam = c["strict"]
    Z, ZR, NEU = prepare(c["G"].loc[:, k], c["Mk"].loc[:, k]); yv = c["y"][k]
    S = {"frozen": (score(CORE, ZR), CORE["thr"]), "strict": (score(STRICT[fam], ZR), STRICT[fam]["thr"]),
         "panel": (pscore(PANELS["none"], Z), PANEL_THR), "panel_strict": (pscore(PANELS[fam], Z), PANELS[fam]["thr"])}
    SC[name] = dict(y=yv, neuron=NEU, Z=Z, pool=c["pool"][k], scores={m: s for m, (s, _) in S.items()}, thr={m: t for m, (_, t) in S.items()})
    for m, (s, t) in S.items():
        lo, hi = boot_ci(yv, s)
        resid = s - np.polyval(np.polyfit(NEU, s, 1), NEU)
        r = {"cohort": name, "model": m, "n": len(yv), "control": int((yv == 0).sum()), "PD": int(yv.sum()),
             "auc": roc_auc_score(yv, s), "ci_lo": lo, "ci_hi": hi, "perm_p": perm_p(yv, s),
             "auc_neuron_markers": roc_auc_score(yv, -NEU), "auc_neuron_removed": roc_auc_score(yv, resid),
             "rho_with_neuron": spearmanr(s, NEU)[0], "loo_neuron_only": loo(NEU[:, None], yv),
             "loo_neuron_plus_score": loo(np.c_[NEU, s], yv), "threshold_oob": t}
        for tn, tv in (("0.5", 0.5), ("oob", t)):
            r.update({f"{kk}@{tn}": v for kk, v in calls(yv, s, tv).items()})
        ROWS.append(r)
    PEOPLE.append(pd.DataFrame({"cohort": name, "donor": c["donor"][k], "y": yv, "stage": np.array(c["stage"])[k],
                                **{m: S[m][0] for m in MODELS}, "neuron_score": NEU,
                                "counted_in_pool": c["pool"][k]}))
RES = pd.DataFrame(ROWS); RES.to_csv(OUT / "external_multi_results.csv", index=False)
pd.concat(PEOPLE).to_csv(OUT / "external_multi_scores.csv", index=False)
print(RES[["cohort", "model", "n", "control", "PD", "auc", "ci_lo", "ci_hi", "perm_p", "auc_neuron_markers", "auc_neuron_removed",
           "loo_neuron_only", "loo_neuron_plus_score"]].round(3).to_string(index=False))
print(); print(RES[["cohort", "model"] + [c for c in RES.columns if "@" in c]].round(2).to_string(index=False))''')

code(r'''# ---------------- cohorts together (each donor once) ----------------
def fast_auc(yv, s):
    r = rankdata(s); n1 = int(yv.sum()); n0 = len(yv) - n1
    return (r[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
def resid(s, neu): return s - np.polyval(np.polyfit(neu, s, 1), neu)
def pool_set(names, n_boot=4000):
    # cohort AUCs averaged with weights = PD-control pairs; 95% CI by resampling people within each cohort
    P = {n: (SC[n]["y"][SC[n]["pool"]], SC[n]["neuron"][SC[n]["pool"]], {m: SC[n]["scores"][m][SC[n]["pool"]] for m in MODELS}) for n in names}
    Y = [P[n][0] for n in names]; W = np.array([v.sum() * (1 - v).sum() for v in Y], float)
    get = {"neuron": [-P[n][1] for n in names]}
    for m in MODELS:
        get[m] = [P[n][2][m] for n in names]
        get[m + "|neuron_removed"] = [resid(P[n][2][m], P[n][1]) for n in names]
    full = [np.arange(len(v)) for v in Y]
    boots = [[rng.integers(0, len(v), len(v)) for v in Y] for _ in range(n_boot)]
    def est(S_, I):
        a = np.array([fast_auc(Y[j][i], S_[j][i]) for j, i in enumerate(I)]); ok = ~np.isnan(a)
        return np.sum(W[ok] * a[ok]) / W[ok].sum()
    out = {"cohorts": names, "people": int(sum(len(v) for v in Y)), "control": int(sum((v == 0).sum() for v in Y)),
           "PD": int(sum(v.sum() for v in Y)), "auc": {}}
    for key, S_ in get.items():
        bs = [est(S_, I) for I in boots]
        out["auc"][key] = {"auc": est(S_, full), "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5)}
    yy = np.concatenate(Y)
    for tn in ("0.5", "oob"):
        out[f"calls@{tn}"] = {m: calls(yy, np.concatenate([(P[n][2][m] > (0.5 if tn == "0.5" else SC[n]["thr"][m])).astype(float)
                                                           for n in names]), 0.5) for m in MODELS}
    return out
NAMES = list(SC)
UNSEEN = [n for n in NAMES if n != "GSE7621"]         # GSE7621 was looked at in earlier versions of the project
POOL = {"unseen": pool_set(UNSEEN), "all": pool_set(NAMES)}
for k, v in POOL.items():
    print(f"== {k}: {len(v['cohorts'])} cohorts, {v['people']} people ({v['control']} control, {v['PD']} PD)")
    for key, a in v["auc"].items():
        print(f"   {key:30s} AUC {a['auc']:.3f} ({a['ci_lo']:.2f}-{a['ci_hi']:.2f})")
    print(pd.DataFrame(v["calls@oob"]).T.round(3).to_string())''')

md("## 5. Gene by gene: the Boruta panel across the external cohorts (fixed-effect inverse-variance mean)")

code(r'''def hedges(v, yv):
    a, b = v[yv == 1], v[yv == 0]; n1, n0 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n0 - 1) * b.var(ddof=1)) / (n1 + n0 - 2)) + 1e-9
    g = (1 - 3 / (4 * (n1 + n0) - 9)) * (a.mean() - b.mean()) / sp
    return g, (n1 + n0) / (n1 * n0) + g ** 2 / (2 * (n1 + n0))
EFF = []
for name in NAMES:
    S = SC[name]; p = S["pool"]; Z, yv, NEU = S["Z"][p], S["y"][p], S["neuron"][p]
    nc = NEU - NEU.mean(); ZA = Z - np.outer(nc, (Z * nc[:, None]).sum(0) / (nc ** 2).sum())
    for g in PANEL:
        if g in COH[name]["G"].index:
            j = GENES.index(g); gr, vr = hedges(Z[:, j], yv); ga, va = hedges(ZA[:, j], yv)
            EFF.append({"cohort": name, "gene": g, "symbol": SYM.get(g, g), "g": gr, "v": vr, "g_adj": ga, "v_adj": va})
EFF = pd.DataFrame(EFF); EFF.to_csv(OUT / "external_multi_gene_by_cohort.csv", index=False)
def ivw(col, vcol):
    return EFF.groupby("gene").apply(lambda d: pd.Series({"g": np.sum(d[col] / d[vcol]) / np.sum(1 / d[vcol]),
                                                        "se": np.sqrt(1 / np.sum(1 / d[vcol])), "k": len(d)}))
MR, MA = ivw("g", "v").reindex(PANEL), ivw("g_adj", "v_adj").reindex(PANEL)
GM = pd.DataFrame({"symbol": [SYM.get(g, g) for g in PANEL], "g_lcm": DE.loc[PANEL, "hedges_g_meta"].to_numpy(),
                   "deg": DE.loc[PANEL, "is_deg"].to_numpy(), "g_external": MR["g"].to_numpy(), "se_external": MR["se"].to_numpy(),
                   "g_external_neuron_adj": MA["g"].to_numpy(), "se_external_neuron_adj": MA["se"].to_numpy(),
                   "k_cohorts": MR["k"].to_numpy()}, index=pd.Index(PANEL, name="gene"))
GM.to_csv(OUT / "external_multi_gene_meta.csv")
AGREE = {}
for col in ("g_external", "g_external_neuron_adj"):
    d = GM.dropna(subset=[col]); k = int((np.sign(d[col]) == np.sign(d.g_lcm)).sum())
    AGREE[col] = {"same_sign": k, "n": len(d), "binom_p": binomtest(k, len(d), 0.5, alternative="greater").pvalue,
                  "spearman": spearmanr(d.g_lcm, d[col])[0]}
print(GM.round(2).to_string()); print(json.dumps(AGREE, indent=1, default=float))''')

md("## 6. Early stage: incidental Lewy body disease (Braak 1-4) against controls, GSE49036")

code(r'''e = EARLY_RAW; m = e["lab"] != 1
Z, ZR, NEU = prepare(e["G"].loc[:, m], e["Mk"].loc[:, m]); ye = (e["lab"][m] == 2).astype(int)
EARLY = {"control": int((ye == 0).sum()), "ilbd": int(ye.sum())}
for key, mod in (("frozen", CORE), ("strict", STRICT["nbb"]), ("panel", PANELS["none"]), ("panel_strict", PANELS["nbb"])):
    s = score(mod, ZR) if "pca" in mod else pscore(mod, Z); lo, hi = boot_ci(ye, s)
    EARLY[key] = {"auc": roc_auc_score(ye, s), "ci_lo": lo, "ci_hi": hi, "perm_p": perm_p(ye, s),
                  "auc_neuron_removed": roc_auc_score(ye, s - np.polyval(np.polyfit(NEU, s, 1), NEU)),
                  **{f"{k}@oob": v for k, v in calls(ye, s, mod["thr"]).items()}}
EARLY["auc_neuron_markers"] = roc_auc_score(ye, -NEU)
# score along the whole Braak sequence (all 28 GSE49036 donors, one within-cohort standardisation)
Z, ZR, NEU = prepare(e["G"], e["Mk"])
ST = pd.DataFrame({"donor": e["donor"], "braak": e["braak"], "frozen": score(CORE, ZR), "strict": score(STRICT["nbb"], ZR), "neuron_score": NEU})
ST["stage_rank"] = ST.braak.map({"CTRL": 0, "BR12": 1, "BR34": 2, "BR56": 3}); ST.to_csv(OUT / "gse49036_braak_scores.csv", index=False)
EARLY["spearman_strict_with_braak"] = spearmanr(ST.stage_rank, ST.strict)[0]
EARLY["spearman_neuron_with_braak"] = spearmanr(ST.stage_rank, ST.neuron_score)[0]
print(json.dumps(EARLY, indent=1, default=float)); print(ST.groupby("braak")[["frozen", "strict", "neuron_score"]].mean().round(3))
json.dump({"thresholds": THRESHOLDS, "strict_models": {k: {"trained_without": v, "people": STRICT[k]["n"], "components": STRICT[k]["components"],
                                                          "panel_genes": [SYM.get(GENES[j], GENES[j]) for j in PANELS[k]["genes"]],
                                                          "panel_genes_in_reported_panel": len(set(PANELS[k]["genes"]) & set(PI))}
                             for k, v in RELATED.items()},
           "pooled": POOL, "gene_agreement": AGREE, "early_stage_GSE49036": EARLY, "donor_overlap": OV.to_dict("records"),
           "donor_matches": DET.to_dict("records")}, open(OUT / "external_multi_summary.json", "w"), indent=1, default=float)
import shutil; shutil.rmtree(GEO, ignore_errors=True)
log("analysis done")''')

md("## 7. The figure (print size, 183 x 150 mm)")
code((HERE / "figcode" / "fig_ext_multi.py").read_text())

md("## 8. Ready-to-paste text: legend, methods, numbers")
code((HERE / "figcode" / "ext_multi_text.py").read_text())

write_nb(HERE / "PD_LCM_rf_external_multi.ipynb", CELLS, "exm")
