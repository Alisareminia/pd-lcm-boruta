"""Builds PD_LCM_rf_enrichment.ipynb - what the 30 Boruta genes are about.

Reads the finished outputs of pd-lcm-rf-core (expression, labels), pd-lcm-rf-boruta-panel (panel, DE table, gene
annotation) and pd-lcm-rf-external-multi (external replication); downloads gene-set libraries (Enrichr), the
Kamath et al. 2022 dopamine-neuron subtype markers (Europe PMC), gene descriptions (mygene.info) and Parkinson's
disease association scores (Open Targets) on Kaggle. Four analyses:
  1. over-representation, up (18) / down (12) / all (30), background = the 5,622 genes measured, BH over the whole
     family, calibrated against random and DE-matched gene sets of the same size;
  2. dopamine-neuron subtypes: do the down genes mark the vulnerable SOX6 lineage and the up genes the resilient
     CALB1 lineage (Kamath 2022, Supplementary Table 8)? Panel-level and transcriptome-wide, label-shuffle nulls;
  3. pathway neighbours: for every pathway holding a panel gene, do its other members shift the same way?
     Label-shuffle null, max-statistic FWER and empirical FDR;
  4. a per-gene annotation table.
"""
import pathlib
from kaggle_nb import write_nb
HERE = pathlib.Path(__file__).parent
FC = HERE / "figcode"
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md("""# Enrichment and pathway analysis of the 30 Boruta genes

Reads `pd-lcm-rf-core`, `pd-lcm-rf-boruta-panel` and `pd-lcm-rf-external-multi`; nothing there is refitted.

1. **Over-representation** - up-regulated (18), down-regulated (12) and all 30 genes, analysed separately as the primary
   analysis (up- and down-regulated genes usually reflect different biology) and together as a secondary one.
   Background: the 5,622 genes actually measured. Terms of 10-500 genes, BH over **every** tested term, and each result
   calibrated against random gene sets of the same size and against random sets matched on differential-expression strength.
2. **Dopamine-neuron subtypes** - Kamath et al. 2022 (Nat Neurosci) found that the SOX6_AGTR1 dopamine neurons degenerate
   in PD while CALB1 neurons are relatively spared. Do the panel's down genes mark the vulnerable SOX6 lineage and its up
   genes the resilient CALB1 lineage?
3. **Pathway neighbours** - for each pathway that contains a panel gene, do the pathway's *other* genes move the same way in
   the discovery donors? Label shuffles within study; whole-search FWER and empirical FDR.
4. **Per-gene table** - function, known PD association (Open Targets), subtype lineage, replication in the 8 bulk cohorts.""")

code(r'''import os, io, re, json, time, glob, zipfile, urllib.request, urllib.parse, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import hypergeom, mannwhitneyu, spearmanr, rankdata
from scipy import sparse
warnings.filterwarnings("ignore")
ON_KAGGLE = Path("/kaggle/input").exists()
OUT = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("SMOKE_OUT", "smoke_enr"))
OUT.mkdir(parents=True, exist_ok=True)
B_RAND = 2000 if ON_KAGGLE else 50          # random / matched gene sets per list
B_PERM = 5000 if ON_KAGGLE else 60          # label shuffles
rng = np.random.default_rng(42)
t0 = time.time()
def log(m): print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)
def find_any(pattern, key):
    root = "/kaggle/input" if ON_KAGGLE else os.environ["LOCAL_" + key.upper().replace("-", "_")]
    hits = sorted((h for h in glob.glob(f"{root}/**/{pattern}", recursive=True) if (key in h or not ON_KAGGLE)), key=len)
    if not hits:
        raise FileNotFoundError(f"{pattern} ({key})")
    return hits[0]
def get(url, data=None, headers=None, timeout=300, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers or {"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                return fh.read()
        except Exception as exc:
            print("  retry", k + 1, type(exc).__name__); time.sleep(8)
    raise RuntimeError(url)
pd.set_option("display.width", 230); pd.set_option("display.max_colwidth", 60)''')

md("## 0. The discovery data, the panel and the background")

code(r'''cz = np.load(find_any("core_data.npz", "rf-core"), allow_pickle=True)
X, y, DS = cz["X"].astype(float), cz["y"].astype(int), cz["ds"].astype(str)
GENES = [str(g) for g in cz["genes"]]
SYM = pd.read_csv(find_any("15_gene_symbol_map.csv", "rf-core")).set_index("gene")["symbol"].to_dict()
DE = pd.read_csv(find_any("03_de_results_full.csv", "boruta-panel")).set_index("gene").reindex(GENES)
ANN = pd.read_csv(find_any("13_gene_annotation_master.csv", "boruta-panel")).set_index("gene")
PANEL = pd.read_csv(find_any("04_boruta_selected_genes.csv", "boruta-panel")).gene.tolist()
EXT = pd.read_csv(find_any("external_multi_gene_meta.csv", "external-multi")).set_index("gene")
sym = [SYM.get(g) if isinstance(SYM.get(g), str) else None for g in GENES]
BG = pd.DataFrame({"gene": GENES, "symbol": [s.upper() if s else None for s in sym], "g": DE.hedges_g_meta.to_numpy(),
                   "p": DE.pvalue.to_numpy()})
BG = BG[BG.symbol.notna() & ~BG.symbol.duplicated()].reset_index(drop=True)
IDX = {s: i for i, s in enumerate(BG.symbol)}
GI = {g: i for i, g in enumerate(BG.gene)}
N = len(BG)
UP = [g for g in PANEL if DE.loc[g, "hedges_g_meta"] > 0]
DOWN = [g for g in PANEL if DE.loc[g, "hedges_g_meta"] < 0]
LISTS = {"up": UP, "down": DOWN, "all": PANEL}
S = lambda gs: [SYM.get(g, g) for g in gs]
log(f"{len(y)} donors, background {N:,} genes with symbols; panel {len(PANEL)}: {len(UP)} up, {len(DOWN)} down")
print("up:  ", ", ".join(S(UP))); print("down:", ", ".join(S(DOWN)))
PANEL_I = np.array([GI[g] for g in PANEL])
# matching for the DE-matched null: 10 bins of |g| x sign, panel genes excluded from the pools
BG["bin"] = pd.qcut(BG.g.abs().rank(method="first"), 10, labels=False).astype(int) * 2 + (BG.g > 0).astype(int)
POOLS = {b: np.setdiff1d(np.flatnonzero(BG.bin.to_numpy() == b), PANEL_I) for b in BG.bin.unique()}
def matched_draw(idx):
    out = []
    for b, k in pd.Series(BG.bin.to_numpy()[idx]).value_counts().items():
        out.extend(rng.choice(POOLS[b], k, replace=False))
    return np.array(out)''')

md("## 1. Gene-set libraries")

code(r'''LIBS = ["GO_Biological_Process_2023", "GO_Cellular_Component_2023", "GO_Molecular_Function_2023", "Reactome_2022",
        "KEGG_2021_Human", "WikiPathway_2023_Human"]
SHORT = {"GO_Biological_Process_2023": "GO:BP", "GO_Cellular_Component_2023": "GO:CC", "GO_Molecular_Function_2023": "GO:MF",
         "Reactome_2022": "Reactome", "KEGG_2021_Human": "KEGG", "WikiPathway_2023_Human": "WikiPathways"}
def enrichr_lib(name):
    txt = get(f"https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={name}").decode()
    out = {}
    for line in txt.splitlines():
        parts = line.split("\t")
        if len(parts) > 2:
            out[parts[0].strip()] = {g.split(",")[0].strip().upper() for g in parts[2:] if g.strip()}
    return out
if ON_KAGGLE:
    RAW = {lib: enrichr_lib(lib) for lib in LIBS}
else:                                            # smoke run: random sets plus a few holding panel genes
    syms = BG.symbol.to_numpy(); RAW = {}
    for lib in LIBS:
        RAW[lib] = {f"{lib[:6]} term {i}": set(rng.choice(syms, rng.integers(12, 300), replace=False)) for i in range(150)}
        for i, g in enumerate(S(PANEL)[:8]):
            RAW[lib][f"{lib[:6]} anchored {i}"] = set(rng.choice(syms, 40, replace=False)) | {g.upper()}
TERMS, seen = [], set()
for lib in LIBS:
    for term, genes in RAW[lib].items():
        idx = tuple(sorted(IDX[g] for g in genes if g in IDX))
        if len(idx) < 10 or idx in seen:                              # identical gene sets across libraries kept once
            continue
        seen.add(idx)
        TERMS.append({"library": SHORT[lib], "term": re.sub(r"\s*\((GO:\d+)\)$", r" (\1)", term), "idx": np.array(idx),
                      "size": len(idx)})
TERMS = pd.DataFrame(TERMS)
def member_matrix(T):
    rows = np.repeat(np.arange(len(T)), T["size"].to_numpy())
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, np.concatenate(T.idx.to_numpy()))), shape=(len(T), N))
log(f"{len(TERMS):,} distinct gene sets with >= 10 background genes: " + ", ".join(f"{k} {v}" for k, v in TERMS.library.value_counts().items()))''')

md("""## 2. Over-representation: up, down, all

Hypergeometric test against the 5,622 measured genes, terms of 10-500 background genes, BH over **all** terms (not only
those the list touches - selecting the family on the outcome is what makes small queries return false pathways).
Calibration: the same procedure on 2,000 random gene sets of the same size, and on 2,000 sets matched gene by gene on
differential-expression strength and direction.""")

code(r'''ORA_T = TERMS[TERMS["size"] <= 500].reset_index(drop=True)
MB = member_matrix(ORA_T)
K = ORA_T["size"].to_numpy()
def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out
def ora_table(n):
    """P value lookup by (hits, term) for a list of n genes: hypergeometric upper tail."""
    return np.vstack([hypergeom.sf(h - 1, N, K, n) for h in range(n + 1)])
def hits_of(sets):
    R = np.zeros((N, len(sets)), dtype=np.int32)
    for j, s in enumerate(sets):
        R[s, j] = 1
    return np.asarray(MB @ R)
ORA, CAL = [], {}
for name, genes in LISTS.items():
    idx = np.array([GI[g] for g in genes]); n = len(idx); P = ora_table(n)
    h = hits_of([idx])[:, 0]; p = P[h, np.arange(len(K))]; q = bh(p)
    rnd = [rng.choice(N, n, replace=False) for _ in range(B_RAND)]
    mat = [matched_draw(idx) for _ in range(B_RAND)]
    hr, hm = hits_of(rnd), hits_of(mat)                                # terms x draws
    pr, pm = P[hr, np.arange(len(K))[:, None]], P[hm, np.arange(len(K))[:, None]]
    minr = np.where(hr >= 2, pr, 1).min(0); minm = np.where(hm >= 2, pm, 1).min(0)
    fp_any = np.mean([(bh(pr[:, j]) < 0.05)[hr[:, j] >= 2].any() for j in range(B_RAND)])
    CAL[name] = {"n": n, "terms": len(K), "random_sets_with_any_BH_hit": float(fp_any),
                 "p_fwer05_random": float(np.quantile(minr, 0.05)), "p_fwer05_matched": float(np.quantile(minm, 0.05))}
    keep = np.flatnonzero(h >= 2)
    for t in keep:
        members = set(ORA_T.idx[t]); gl = [BG.symbol[i] for i in idx if i in members]
        ORA.append({"list": name, "library": ORA_T.library[t], "term": ORA_T.term[t], "size": int(K[t]), "hits": int(h[t]),
                    "genes": ", ".join(sorted(gl)), "p": p[t], "q_BH": q[t],
                    "fwer_random": (np.sum(minr <= p[t]) + 1) / (B_RAND + 1),
                    "p_vs_matched": (np.sum(pm[t] <= p[t]) + 1) / (B_RAND + 1),
                    "fwer_matched": (np.sum(minm <= p[t]) + 1) / (B_RAND + 1)})
    log(f"{name}: {n} genes, {len(K):,} terms, {len(keep)} with >= 2 genes; best p = {p[keep].min() if len(keep) else 1:.2e}; "
        f"BH < 0.05: {int((q[keep] < 0.05).sum())}; random sets with any BH hit: {100 * fp_any:.1f}%")
ORA = pd.DataFrame(ORA).sort_values(["list", "p"]).reset_index(drop=True)
ORA.to_csv(OUT / "enrich_ora.csv", index=False)
for name in LISTS:
    print(f"\n==== {name}"); print(ORA[ORA.list == name].head(8)[["library", "term", "size", "hits", "genes", "p", "q_BH", "fwer_random",
                                                                    "p_vs_matched"]].to_string(index=False))''')

md("""## 3. Per-gene PD effect in the discovery donors, observed and under label shuffles

The same statistic for every gene - a Hedges' g within each study, averaged across the four studies with weights
n1*n0/(n1+n0) - computed on the real labels and on labels shuffled within each study. Sections 4 and 5 use it.""")

code(r'''XB = X[:, [GENES.index(g) for g in BG.gene]]
STUDIES = sorted(set(DS))
def effect(Ys):
    """Ys: (b x donors) 0/1 labels -> (b x genes) pooled within-study Hedges' g."""
    num = np.zeros((Ys.shape[0], XB.shape[1]), np.float32); den = 0.0
    for d in STUDIES:
        m = DS == d; Xk = XB[m].astype(np.float32); Yk = Ys[:, m].astype(np.float32)
        n1 = Yk.sum(1, keepdims=True); n0 = m.sum() - n1
        S1, Q1 = Yk @ Xk, Yk @ (Xk ** 2); S, Q = Xk.sum(0), (Xk ** 2).sum(0)
        m1, m0 = S1 / n1, (S - S1) / n0
        v1 = (Q1 - n1 * m1 ** 2) / (n1 - 1); v0 = (Q - Q1 - n0 * m0 ** 2) / (n0 - 1)
        sp = np.sqrt(np.clip(((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2), 1e-6, None))
        J = 1 - 3 / (4 * (n1 + n0) - 9); w = float((n1 * n0 / (n1 + n0))[0, 0])
        num += w * J * (m1 - m0) / sp; den += w
    return num / den
G_OBS = effect(y[None, :])[0]
def shuffled(b):
    Ys = np.tile(y, (b, 1))
    for d in STUDIES:
        m = np.flatnonzero(DS == d)
        Ys[:, m] = np.array([rng.permutation(y[m]) for _ in range(b)])
    return Ys
G_PERM = np.vstack([effect(shuffled(min(500, B_PERM - s))) for s in range(0, B_PERM, 500)])
log(f"per-gene effect: observed vs the DE table's meta g, Spearman {spearmanr(G_OBS, BG.g)[0]:.3f}; {G_PERM.shape[0]:,} label shuffles")''')

md("""## 4. Dopamine-neuron subtypes (Kamath et al. 2022)

Supplementary Table 8 of Kamath et al. gives, for each of ten human dopamine-neuron subtypes, the genes that mark it against
the other dopamine neurons (MAST z). **Lineage score** of a gene = its mean z across the six CALB1 subtypes minus its mean z
across the four SOX6 subtypes: positive marks the resilient CALB1 lineage, negative the vulnerable SOX6 lineage (SOX6_AGTR1 is
the population lost in PD).""")

code((FC / "strict_xlsx.py").read_text())

code(r'''if ON_KAGGLE:
    j = json.loads(get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&query=" + urllib.parse.quote('DOI:"10.1038/s41593-022-01061-1"')))
    pmcid = j["resultList"]["result"][0]["pmcid"]
    zf = zipfile.ZipFile(io.BytesIO(get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles", timeout=600)))
    name = [n for n in zf.namelist() if n.endswith("MOESM3_ESM.xlsx")][0]
    (OUT / "kamath_2022_supplementary.xlsx").write_bytes(zf.read(name))
    T8 = read_strict_xlsx(OUT / "kamath_2022_supplementary.xlsx")["Supplementary_Table_8"]
    T8.columns = [str(c) if c is not None else f"c{i}" for i, c in enumerate(T8.iloc[0])]; T8 = T8.iloc[1:]
    T8 = T8.rename(columns={"primerid": "symbol"})[["DA_subtype", "symbol", "coef", "z", "fdr"]]
else:                                                  # smoke run: invented markers
    subs = ["SOX6_AGTR1", "SOX6_PART1", "SOX6_DDT", "SOX6_GFRA2", "CALB1_CALCR", "CALB1_CRYM_CCDC68", "CALB1_GEM", "CALB1_PPP1R17",
            "CALB1_RBP4", "CALB1_TRHR"]
    T8 = pd.DataFrame([{"DA_subtype": s, "symbol": g, "coef": rng.normal(), "z": rng.normal(0, 4), "fdr": rng.uniform(0, 0.05)}
                       for s in subs for g in rng.choice(BG.symbol, 1500, replace=False)])
for c in ("coef", "z", "fdr"):
    T8[c] = pd.to_numeric(T8[c], errors="coerce")
T8["symbol"] = T8.symbol.astype(str).str.upper()
print(f"Kamath Table 8: {len(T8):,} rows, {T8.symbol.nunique():,} genes, subtypes {sorted(T8.DA_subtype.unique())}; "
      f"FDR range {T8.fdr.min():.1e}-{T8.fdr.max():.2f}; coef<0 rows {int((T8.coef < 0).sum()):,}")
ZT = T8.pivot_table(index="symbol", columns="DA_subtype", values="z", aggfunc="first").reindex(BG.symbol).fillna(0.0)
SOX_SUB = [c for c in ZT.columns if c.startswith("SOX6")]; CALB_SUB = [c for c in ZT.columns if c.startswith("CALB1")]
BG["lineage"] = (ZT[CALB_SUB].mean(1) - ZT[SOX_SUB].mean(1)).to_numpy()
BG["agtr1"] = ZT["SOX6_AGTR1"].to_numpy()
cover = float((ZT.abs().sum(1) > 0).mean())
lin = BG.lineage.to_numpy()
SUB = {"coverage": cover}
up_i, dn_i = np.array([GI[g] for g in UP]), np.array([GI[g] for g in DOWN])
# two-sided throughout: the direction of the split is a finding, not an assumption
SUB["up_vs_down"] = mannwhitneyu(lin[up_i], lin[dn_i], alternative="two-sided").pvalue
SUB["up_vs_background"] = mannwhitneyu(lin[up_i], np.delete(lin, PANEL_I), alternative="two-sided").pvalue
SUB["down_vs_background"] = mannwhitneyu(lin[dn_i], np.delete(lin, PANEL_I), alternative="two-sided").pvalue
# DE-matched null: is the separation larger than for genes equally strongly changed in PD?
obs = lin[up_i].mean() - lin[dn_i].mean()
null = np.array([lin[matched_draw(up_i)].mean() - lin[matched_draw(dn_i)].mean() for _ in range(B_RAND)])
SUB["separation"] = float(obs); SUB["p_vs_DE_matched"] = float((np.sum(null >= obs) + 1) / (B_RAND + 1))
# transcriptome-wide: do genes of the CALB1 lineage rise and genes of the SOX6 lineage fall in PD neurons?
rho = spearmanr(G_OBS, lin)[0]
rho_null = np.array([spearmanr(G_PERM[b], lin)[0] for b in range(min(G_PERM.shape[0], 2000))])
SUB["rho_transcriptome"] = float(rho); SUB["rho_p_label_shuffle"] = float((np.sum(np.abs(rho_null) >= abs(rho)) + 1) / (len(rho_null) + 1))
# each subtype's top-200 markers, against the up and the down list
SUBTAB = []
for st in ZT.columns:
    mk = T8[(T8.DA_subtype == st) & (T8.coef > 0) & (T8.fdr < 0.05) & T8.symbol.isin(IDX)].nlargest(200, "z").symbol
    mk_i = set(IDX[s] for s in mk)
    for name, gi in (("up", up_i), ("down", dn_i)):
        h = int(sum(i in mk_i for i in gi))
        SUBTAB.append({"subtype": st, "lineage": st.split("_")[0], "list": name, "markers": len(mk_i), "hits": h,
                       "genes": ", ".join(sorted(BG.symbol[i] for i in gi if i in mk_i)),
                       "p": hypergeom.sf(h - 1, N, len(mk_i), len(gi))})
SUBTAB = pd.DataFrame(SUBTAB); SUBTAB["q_BH"] = bh(SUBTAB.p)
SUBTAB.to_csv(OUT / "enrich_subtypes.csv", index=False)
PG = BG.loc[PANEL_I, ["gene", "symbol", "g", "lineage", "agtr1"]].assign(direction=lambda d: np.where(d.g > 0, "up", "down"))
PG.to_csv(OUT / "enrich_panel_lineage.csv", index=False)
BG[["gene", "symbol", "g", "lineage", "agtr1"]].to_csv(OUT / "enrich_background_lineage.csv", index=False)
SUB["background_p05"], SUB["background_p95"] = float(np.percentile(lin, 5)), float(np.percentile(lin, 95))
print(json.dumps(SUB, indent=1, default=float))
print(PG.sort_values("lineage").round(2).to_string(index=False))
print(SUBTAB.sort_values("p").head(12).round(4).to_string(index=False))''')

md("""## 5. Pathway neighbours

Every term of 10-150 background genes that holds at least one panel gene. The anchor's direction is the sign of its panel
genes' summed effect; the statistic is the mean effect of the term's **other** genes (all panel genes removed), signed by the
anchor's direction. Null: the same statistic on 5,000 within-study label shuffles; whole-search FWER from the minimum p
across terms in each shuffle, and empirical FDR.""")

code(r'''PANEL_SET = set(PANEL_I)
CT = []
for t in TERMS[TERMS["size"] <= 150].itertuples():
    anchors = [i for i in t.idx if i in PANEL_SET]
    nb = [i for i in t.idx if i not in PANEL_SET]
    if anchors and len(nb) >= 5:
        d = np.sign(sum(G_OBS[i] for i in anchors))
        if d != 0:
            CT.append({"library": t.library, "term": t.term, "size": t.size, "anchors": ", ".join(BG.symbol[i] for i in anchors),
                       "direction": "up" if d > 0 else "down", "d": d, "nb": np.array(nb)})
CT = pd.DataFrame(CT)
NBM = sparse.csr_matrix((np.ones(sum(len(v) for v in CT.nb)), (np.repeat(np.arange(len(CT)), [len(v) for v in CT.nb]), np.concatenate(CT.nb))),
                        shape=(len(CT), N))
nn = np.array([len(v) for v in CT.nb]); dd = CT.d.to_numpy()
obs = dd * (NBM @ G_OBS) / nn
null = (np.asarray(NBM @ G_PERM.T.astype(float)).T / nn) * dd                 # shuffles x terms
p_obs = (np.sum(null >= obs, axis=0) + 1) / (null.shape[0] + 1)
p_null = (null.shape[0] - rankdata(null, axis=0, method="min") + 1) / null.shape[0]          # each shuffle's own p, per term
minp = p_null.min(1)
CT["neighbours"] = nn; CT["shift"] = obs; CT["p"] = p_obs
CT["fwer"] = [(np.sum(minp <= p) + 1) / (len(minp) + 1) for p in p_obs]
order = np.argsort(p_obs); fdr = np.empty(len(p_obs))
for rank, t in enumerate(order, 1):
    fdr[t] = min(1.0, np.mean((p_null <= p_obs[t]).sum(1)) / rank)
CT["fdr"] = np.minimum.accumulate(fdr[order][::-1])[::-1][np.argsort(order)]
pct = rankdata(G_OBS) / N
CT["neighbour_median_percentile"] = [float(np.median(pct[v] if d > 0 else 1 - pct[v])) for v, d in zip(CT.nb, CT.d)]
CT["top_neighbours"] = [", ".join(BG.symbol[i] for i in v[np.argsort(-d * G_OBS[v])][:6]) for v, d in zip(CT.nb, CT.d)]
CT = CT.drop(columns=["d", "nb"]).sort_values("p").reset_index(drop=True)
CT.to_csv(OUT / "enrich_pathway_neighbours.csv", index=False)
log(f"pathway neighbours: {len(CT)} terms hold a panel gene; best p {CT.p.min():.1e}; FWER < 0.05: {int((CT.fwer < 0.05).sum())}; "
    f"empirical FDR < 0.05: {int((CT.fdr < 0.05).sum())}, < 0.10: {int((CT.fdr < 0.10).sum())}")
print(CT.head(15)[["library", "term", "size", "anchors", "direction", "neighbours", "shift", "p", "fwer", "fdr", "neighbour_median_percentile",
                   "top_neighbours"]].round(4).to_string(index=False))''')

md("## 6. Per-gene table")

code(r'''ids = list(PANEL)
if ON_KAGGLE:
    mg = json.loads(get("https://mygene.info/v3/query", data=urllib.parse.urlencode(
        {"q": ",".join(ids), "scopes": "ensembl.gene", "fields": "symbol,name,summary", "species": "human"}).encode()))
    MG = {x["query"]: x for x in mg if not x.get("notfound")}
    q = """query($ids:[String!]!){ disease(efoId:"MONDO_0005180"){ associatedTargets(Bs:$ids, page:{index:0,size:100}){ rows{
          target{ id } score datatypeScores{ id score } } } } }"""
    ot = json.loads(get("https://api.platform.opentargets.org/api/v4/graphql", data=json.dumps({"query": q, "variables": {"ids": ids}}).encode(),
                        headers={"Content-Type": "application/json"}))
    OT = {r["target"]["id"]: {"ot_score": r["score"], **{f"ot_{d['id']}": d["score"] for d in r["datatypeScores"]}}
          for r in ot["data"]["disease"]["associatedTargets"]["rows"]}
else:
    MG, OT = {}, {}
first = lambda s: (re.split(r"(?<=[.])\s", s)[0] if isinstance(s, str) else "")
best_nb = {}
for r in CT.itertuples():
    for a in r.anchors.split(", "):
        best_nb.setdefault(a, r)
rows = []
for g in PANEL:
    a = ANN.loc[g] if g in ANN.index else pd.Series(dtype=float)
    e = EXT.loc[g] if g in EXT.index else pd.Series(dtype=float)
    s_ = BG.symbol[GI[g]]; b = BG.loc[GI[g]]; nbr = best_nb.get(s_)
    rows.append({"gene": g, "symbol": SYM.get(g, g), "name": MG.get(g, {}).get("name", ""),
                 "direction": "up" if DE.loc[g, "hedges_g_meta"] > 0 else "down", "hedges_g_discovery": DE.loc[g, "hedges_g_meta"],
                 "p_discovery": DE.loc[g, "pvalue"], "deg": bool(DE.loc[g, "is_deg"]),
                 "boruta_fold_frequency": a.get("boruta_fold_frequency"), "shap_rank": a.get("shap_rank"), "single_gene_auc": a.get("single_gene_auc"),
                 "g_external_neuron_adjusted": e.get("g_external_neuron_adj"),
                 "replicates_in_bulk": bool(np.sign(e.get("g_external_neuron_adj", np.nan)) == np.sign(DE.loc[g, "hedges_g_meta"])),
                 "dopamine_lineage_score": b.lineage,
                 "top_subtype_marker": (ZT.loc[s_].idxmax() if ZT.loc[s_].max() > 0 else ""),
                 "pd_association_open_targets": OT.get(g, {}).get("ot_score", 0.0),
                 "pd_genetic_association": OT.get(g, {}).get("ot_genetic_association", 0.0),
                 "pd_literature": OT.get(g, {}).get("ot_literature", 0.0),
                 "best_pathway_neighbours": f"{nbr.term} ({nbr.library}; p={nbr.p:.1e}, FDR={nbr.fdr:.2f})" if nbr is not None else "",
                 "summary": first(MG.get(g, {}).get("summary", ""))})
GT = pd.DataFrame(rows).sort_values(["direction", "hedges_g_discovery"], ascending=[False, False])
GT.to_csv(OUT / "enrich_gene_table.csv", index=False)
try:
    GT.to_excel(OUT / "enrich_gene_table.xlsx", index=False)
except Exception as exc:
    print("xlsx not written:", exc)
print(GT.drop(columns=["gene", "summary"]).round(3).to_string(index=False))
json.dump({"lists": {k: S(v) for k, v in LISTS.items()}, "ora_calibration": CAL, "subtypes": SUB,
           "neighbours": {"terms": int(len(CT)), "fwer_lt_0.05": int((CT.fwer < 0.05).sum()), "fdr_lt_0.05": int((CT.fdr < 0.05).sum()),
                          "fdr_lt_0.10": int((CT.fdr < 0.10).sum())}},
          open(OUT / "enrich_summary.json", "w"), indent=1, default=float)
log("done")''')

md("## 7. Figure (print size, 183 x 128 mm)")
code((FC / "ext_unified_style.py").read_text())
code((FC / "fig_enrich.py").read_text())

md("## 8. Ready-to-paste legend, methods and numbers")
code((FC / "enrich_text.py").read_text())

write_nb(HERE / "PD_LCM_rf_enrichment.ipynb", CELLS, "enr")
