"""Builds PD_LCM_rf_gsea.ipynb - pathway analysis with the power of the whole transcriptome.

The 30-gene over-representation test has almost no power (see pd-lcm-rf-pathway-composition); these two analyses use every
measured gene in every donor instead, so their results reach FDR.
  A. GSEA on the ranking of all 5,622 genes by their PD effect (pooled within-study Hedges' g), with the diagnosis labels
     shuffled within each study - the strict null, not gene shuffling. Reports NES, permutation P, BH FDR, the leading-edge
     genes and which panel genes are in them.
  B. Pathway scores per donor (ssGSEA), tested PD vs control like a gene: pooled within-study effect, the same label
     shuffles, BH FDR, and each pathway's AUC.
Gene sets: MSigDB Hallmark, Reactome, KEGG and GO Biological Process from Enrichr, restricted to the measured genes and to
15-500 genes so every result can be named; identical sets are kept once.
"""
import pathlib
from kaggle_nb import write_nb, SETUP, BLOCKS, LOAD_CORE
HERE = pathlib.Path(__file__).parent
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md("""# Pathways with the power of the whole transcriptome

The over-representation test on 30 genes has almost no power, and pathway-level claims should not rest on it. These two
analyses ask a better-powered question of the same donors.

**A. GSEA over the whole ranking.** All 5,622 measured genes are ranked by their PD effect (Hedges' g pooled within study) and
every pathway is tested for sitting at the top or the bottom of that ranking. The null comes from shuffling the diagnosis
labels **within each study** (not from shuffling genes), which is the strict version of the test.

**B. Pathway scores per donor.** Every pathway is scored in every donor (single-sample GSEA) and then tested PD versus control
exactly like a gene, with the same label shuffles. This also gives each pathway an AUC.

Panel genes are marked throughout: for each pathway we report which of the 30 are in it and which are in its leading edge.""")

code(SETUP)
code("N_TREES = 1000\nimport urllib.request")
code(BLOCKS)
code(LOAD_CORE)

code(r'''from scipy import sparse
from scipy.stats import rankdata as _rd
B_PERM = 1000 if ON_KAGGLE else 40
PANEL = pd.read_csv(find_input("04_boruta_selected_genes.csv")).gene.tolist()
PSET = set(PANEL)
DE = pd.read_csv(find_input("03_de_results_full.csv")).set_index("gene").reindex(GENES)
UPP = {g for g in PANEL if DE.loc[g, "hedges_g_meta"] > 0}
SYMU = [sym(g).upper() for g in GENES]
IDX = {}
for i, s_ in enumerate(SYMU):
    IDX.setdefault(s_, i)                                          # first index per symbol
G = len(GENES)
def zds(M):
    out = np.empty_like(M, dtype=float)
    for d in set(DS):
        m = DS == d; sd = M[m].std(0, ddof=1); out[m] = (M[m] - M[m].mean(0)) / np.where(sd > 1e-9, sd, 1)
    return out
XZ = zds(X.astype(float))
log(f"{len(y)} donors, {G:,} genes; panel {len(PANEL)}")''')

md("## The gene sets")

code(r'''import re
LIBS = ["MSigDB_Hallmark_2020", "Reactome_2022", "KEGG_2021_Human", "GO_Biological_Process_2023"]
SHORT = {"MSigDB_Hallmark_2020": "Hallmark", "Reactome_2022": "Reactome", "KEGG_2021_Human": "KEGG",
         "GO_Biological_Process_2023": "GO BP"}
def enrichr_lib(name):
    for k in range(6):
        try:
            txt = urllib.request.urlopen(f"https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={name}",
                                         timeout=300).read().decode()
            out = {}
            for line in txt.splitlines():
                p = line.split("\t")
                if len(p) > 2:
                    out[p[0].strip()] = {g.split(",")[0].strip().upper() for g in p[2:] if g.strip()}
            return out
        except Exception as exc:
            print("  Enrichr retry", k + 1, type(exc).__name__); time.sleep(8)
    raise RuntimeError(name)
if ON_KAGGLE:
    RAW = {lib: enrichr_lib(lib) for lib in LIBS}
else:
    rs = np.random.default_rng(0)
    RAW = {lib: {f"{SHORT[lib]} set {i}": set(rs.choice(list(IDX), rs.integers(18, 200), replace=False)) for i in range(120)} for lib in LIBS}
SETS, seen = [], set()
for lib in LIBS:
    for name, genes in RAW[lib].items():
        idx = tuple(sorted({IDX[g] for g in genes if g in IDX}))
        if 15 <= len(idx) <= 500 and idx not in seen:
            seen.add(idx)
            nm = re.sub(r"\s*\(GO:\d+\)$", "", name)
            SETS.append({"source": SHORT[lib], "set": re.sub(r"\s+R-HSA-\d+$", "", nm), "idx": np.array(idx), "size": len(idx)})
SETS = pd.DataFrame(SETS)
SIZES = SETS["size"].to_numpy()
rows = np.repeat(np.arange(len(SETS)), SIZES)
MB = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, np.concatenate(SETS.idx.to_numpy()))), shape=(len(SETS), G))
PANEL_IN = [", ".join(sorted(sym(GENES[j]) for j in s_ if GENES[j] in PSET)) for s_ in SETS.idx]
SETS["panel_genes"] = PANEL_IN
SETS["n_panel"] = [0 if not p else len(p.split(", ")) for p in PANEL_IN]
log(f"{len(SETS):,} distinct gene sets of 15-500 measured genes: " + ", ".join(f"{k} {v}" for k, v in SETS.source.value_counts().items())
    + f"; {int((SETS.n_panel > 0).sum())} hold at least one panel gene")''')

md("## A. GSEA on the ranking of all genes, labels shuffled within study")

code(r'''def effect(Ys):
    """Pooled within-study Hedges' g per gene for each row of labels."""
    num = np.zeros((Ys.shape[0], G), np.float32); den = 0.0
    for d in sorted(set(DS)):
        m = DS == d; Xk = XZ[m].astype(np.float32); Yk = Ys[:, m].astype(np.float32)
        n1 = Yk.sum(1, keepdims=True); n0 = m.sum() - n1
        S1, Q1 = Yk @ Xk, Yk @ (Xk ** 2); S, Q = Xk.sum(0), (Xk ** 2).sum(0)
        m1, m0 = S1 / n1, (S - S1) / n0
        v1 = (Q1 - n1 * m1 ** 2) / (n1 - 1); v0 = (Q - Q1 - n0 * m0 ** 2) / (n0 - 1)
        sp = np.sqrt(np.clip(((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2), 1e-6, None))
        J = 1 - 3 / (4 * (n1 + n0) - 9); w = float((n1 * n0 / (n1 + n0))[0, 0])
        num += w * J * (m1 - m0) / sp; den += w
    return num / den
def shuffled(b, rs):
    Ys = np.tile(y, (b, 1))
    for d in set(DS):
        m = np.flatnonzero(DS == d)
        Ys[:, m] = np.array([rs.permutation(y[m]) for _ in range(b)])
    return Ys
METRIC = effect(y[None, :])[0]
GRANK = pd.DataFrame({"gene": GENES, "symbol": [sym(g) for g in GENES], "metric": METRIC, "panel": [g in PSET for g in GENES]})
GRANK.sort_values("metric", ascending=False).to_csv(OUT / "gsea_gene_ranking.csv", index=False)
def es_all(metric, want_edge=False):
    """Weighted Kolmogorov-Smirnov enrichment score of every gene set, in one pass."""
    order = np.argsort(-metric, kind="stable")
    a = np.abs(metric[order]).astype(np.float32)
    Msort = MB[:, order].toarray()
    W = Msort * a
    hit = np.cumsum(W, axis=1); tot = hit[:, -1:]
    hit /= np.where(tot > 0, tot, 1)
    miss = (np.arange(1, G + 1, dtype=np.float32) - np.cumsum(Msort, axis=1)) / (G - SIZES)[:, None]
    dev = hit - miss
    j = np.argmax(np.abs(dev), axis=1)
    es = dev[np.arange(len(SETS)), j]
    if not want_edge:
        return es
    edge = []
    for i in range(len(SETS)):
        lo, hi = (0, j[i] + 1) if es[i] > 0 else (j[i], G)
        members = order[lo:hi][Msort[i, lo:hi] > 0]
        members = members[np.argsort(-np.abs(metric[members]))]
        edge.append(members)
    return es, edge
ES, EDGE = es_all(METRIC, want_edge=True)
rs = np.random.default_rng(SEED)
NULL = np.empty((B_PERM, len(SETS)), np.float32)
for b in range(0, B_PERM, 50):
    ys = shuffled(min(50, B_PERM - b), rs)
    Gp = effect(ys)
    for k in range(Gp.shape[0]):
        NULL[b + k] = es_all(Gp[k])
    log(f"GSEA permutations {b + Gp.shape[0]}/{B_PERM}")
pos = NULL > 0
nes, pval = np.zeros(len(SETS)), np.zeros(len(SETS))
for i in range(len(SETS)):
    same = NULL[:, i][pos[:, i]] if ES[i] > 0 else NULL[:, i][~pos[:, i]]
    mu = np.abs(same).mean() if len(same) else np.nan
    nes[i] = ES[i] / mu if mu and np.isfinite(mu) else np.nan
    pval[i] = (np.sum(np.abs(same) >= abs(ES[i])) + 1) / (len(same) + 1)
def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out
# classic GSEA FDR (Subramanian et al.): the null NES of every set in every shuffle, pooled by sign
NES_NULL = np.empty_like(NULL)
for i in range(len(SETS)):
    col = NULL[:, i]; mp = np.abs(col[col > 0]).mean() if (col > 0).any() else np.nan
    mn = np.abs(col[col < 0]).mean() if (col < 0).any() else np.nan
    NES_NULL[:, i] = np.where(col > 0, col / (mp if np.isfinite(mp) else 1), col / (mn if np.isfinite(mn) else 1))
def gsea_fdr(nes_obs, nes_null):
    q = np.ones(len(nes_obs))
    npos_obs, nneg_obs = max((nes_obs > 0).sum(), 1), max((nes_obs < 0).sum(), 1)
    npos_null, nneg_null = max((nes_null > 0).sum(), 1), max((nes_null < 0).sum(), 1)
    for i, v in enumerate(nes_obs):
        if not np.isfinite(v):
            continue
        if v >= 0:
            q[i] = ((nes_null >= v).sum() / npos_null) / max((nes_obs >= v).sum() / npos_obs, 1e-9)
        else:
            q[i] = ((nes_null <= v).sum() / nneg_null) / max((nes_obs <= v).sum() / nneg_obs, 1e-9)
    return np.minimum(q, 1)
Q_GSEA = gsea_fdr(nes, NES_NULL)
# the lenient, widely used alternative: shuffle the genes instead of the donors (competitive null)
B_GENE = 1000 if ON_KAGGLE else 20
rsg = np.random.default_rng(SEED + 2)
GNULL = np.empty((B_GENE, len(SETS)), np.float32)
for b in range(B_GENE):
    GNULL[b] = es_all(METRIC[rsg.permutation(G)])
    if (b + 1) % 200 == 0:
        log(f"gene shuffles {b + 1}/{B_GENE}")
nes_g, p_g = np.zeros(len(SETS)), np.zeros(len(SETS))
for i in range(len(SETS)):
    col = GNULL[:, i]; same = col[col > 0] if ES[i] > 0 else col[col < 0]
    mu = np.abs(same).mean() if len(same) else np.nan
    nes_g[i] = ES[i] / mu if mu and np.isfinite(mu) else np.nan
    p_g[i] = (np.sum(np.abs(same) >= abs(ES[i])) + 1) / (len(same) + 1)
NESG_NULL = np.empty_like(GNULL)
for i in range(len(SETS)):
    col = GNULL[:, i]; mp = np.abs(col[col > 0]).mean() if (col > 0).any() else np.nan
    mn = np.abs(col[col < 0]).mean() if (col < 0).any() else np.nan
    NESG_NULL[:, i] = np.where(col > 0, col / (mp if np.isfinite(mp) else 1), col / (mn if np.isfinite(mn) else 1))
Q_GENE = gsea_fdr(nes_g, NESG_NULL)
GS = SETS.drop(columns=["idx"]).copy()
GS["direction"] = np.where(ES > 0, "up in PD", "down in PD")
GS["ES"], GS["NES"], GS["p_label_shuffle"], GS["q_BH_label_shuffle"] = ES, nes, pval, bh(pval)
GS["q_gsea_label_shuffle"] = Q_GSEA                       # the FDR the GSEA software reports, donors shuffled
GS["NES_gene_shuffle"], GS["p_gene_shuffle"] = nes_g, p_g
GS["q_gsea_gene_shuffle"] = Q_GENE                        # the usual, more lenient published version
GS["q_BH_gene_shuffle"] = bh(p_g)
GS["leading_edge_size"] = [len(e) for e in EDGE]
GS["leading_edge"] = [", ".join(sym(GENES[j]) for j in e[:12]) for e in EDGE]
GS["panel_in_leading_edge"] = [", ".join(sym(GENES[j]) for j in e if GENES[j] in PSET) for e in EDGE]
GS = GS.sort_values("p_label_shuffle").reset_index(drop=True)
GS.to_csv(OUT / "gsea_results.csv", index=False)
for lab, col, thr in (("strict (donors shuffled, BH)", "q_BH_label_shuffle", 0.05),
                      ("GSEA's own FDR, donors shuffled", "q_gsea_label_shuffle", 0.25),
                      ("GSEA's own FDR, genes shuffled", "q_gsea_gene_shuffle", 0.25),
                      ("genes shuffled, BH", "q_BH_gene_shuffle", 0.05)):
    n = int((GS[col] < thr).sum())
    log(f"GSEA, {lab}: {n} gene sets below {thr}")
SHOWCOL = ["source", "set", "size", "direction", "NES", "p_label_shuffle", "q_gsea_label_shuffle", "NES_gene_shuffle",
           "q_gsea_gene_shuffle", "n_panel", "panel_in_leading_edge"]
print(GS.head(25)[SHOWCOL].round(4).to_string(index=False))
sig = GS[GS.q_gsea_gene_shuffle < 0.25]''')

md("## B. A pathway score for every donor (ssGSEA), tested like a gene")

code(r'''def ssgsea(alpha=0.75):
    """Barbie-style single-sample enrichment score of every set in every donor."""
    out = np.zeros((len(y), len(SETS)), np.float32)
    w = (np.arange(G, 0, -1, dtype=np.float32)) ** alpha
    for i in range(len(y)):
        order = np.argsort(-XZ[i], kind="stable")
        M = MB[:, order].toarray()
        Wt = M * w
        hit = np.cumsum(Wt, axis=1); tot = hit[:, -1:]
        hit /= np.where(tot > 0, tot, 1)
        miss = (np.arange(1, G + 1, dtype=np.float32) - np.cumsum(M, axis=1)) / (G - SIZES)[:, None]
        out[i] = (hit - miss).sum(1)
    return out
SS = ssgsea()
SSZ = zds(SS.astype(float))                                   # comparable across studies
def path_effect(Ys, P):
    num = np.zeros((Ys.shape[0], P.shape[1]), np.float32); den = 0.0
    for d in sorted(set(DS)):
        m = DS == d; Pk = P[m].astype(np.float32); Yk = Ys[:, m].astype(np.float32)
        n1 = Yk.sum(1, keepdims=True); n0 = m.sum() - n1
        S1, Q1 = Yk @ Pk, Yk @ (Pk ** 2); S, Q = Pk.sum(0), (Pk ** 2).sum(0)
        m1, m0 = S1 / n1, (S - S1) / n0
        v1 = (Q1 - n1 * m1 ** 2) / (n1 - 1); v0 = (Q - Q1 - n0 * m0 ** 2) / (n0 - 1)
        sp = np.sqrt(np.clip(((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2), 1e-6, None))
        J = 1 - 3 / (4 * (n1 + n0) - 9); w_ = float((n1 * n0 / (n1 + n0))[0, 0])
        num += w_ * J * (m1 - m0) / sp; den += w_
    return num / den
g_obs = path_effect(y[None, :], SSZ)[0]
rs2 = np.random.default_rng(SEED + 1)
nullp = np.vstack([path_effect(shuffled(min(200, 2 * B_PERM - b), rs2), SSZ) for b in range(0, 2 * B_PERM, 200)])
p_ss = (np.sum(np.abs(nullp) >= np.abs(g_obs), axis=0) + 1) / (nullp.shape[0] + 1)
def auc_within(score):
    a, w = [], []
    for d in sorted(set(DS)):
        m = DS == d
        if len(set(y[m])) == 2:
            a.append(roc_auc_score(y[m], score[m])); w.append(y[m].sum() * (1 - y[m]).sum())
    return float(np.average(a, weights=w))
SSR = SETS.drop(columns=["idx"]).copy()
SSR["direction"] = np.where(g_obs > 0, "up in PD", "down in PD")
SSR["g"] = g_obs; SSR["p"] = p_ss; SSR["q_BH"] = bh(p_ss)
SSR["auc"] = [auc_within(SSZ[:, i] if g_obs[i] > 0 else -SSZ[:, i]) for i in range(len(SETS))]
SSR = SSR.sort_values("p").reset_index(drop=True)
SSR.to_csv(OUT / "ssgsea_results.csv", index=False)
pd.DataFrame(SSZ, columns=SETS["set"]).assign(person=PERSON, dataset=DS, y=y).to_csv(OUT / "ssgsea_scores.csv", index=False)
sig2 = SSR[SSR.q_BH < 0.05]
log(f"ssGSEA: {len(sig2)} pathways differ at FDR < 0.05 ({int((sig2.direction == 'up in PD').sum())} up, "
    f"{int((sig2.direction == 'down in PD').sum())} down); best AUC {sig2.auc.max() if len(sig2) else float('nan'):.3f}")
print(sig2.head(25)[["source", "set", "size", "direction", "g", "auc", "p", "q_BH", "n_panel", "panel_genes"]].round(4).to_string(index=False))
both = set(GS[GS.q_gsea_gene_shuffle < 0.25]["set"]) & set(SSR[SSR.p < 0.05]["set"])
json.dump({"sets_tested": int(len(SETS)),
           "gsea_strict_BH_lt_0.05": int((GS.q_BH_label_shuffle < 0.05).sum()),
           "gsea_own_fdr_donors_lt_0.25": int((GS.q_gsea_label_shuffle < 0.25).sum()),
           "gsea_own_fdr_genes_lt_0.25": int((GS.q_gsea_gene_shuffle < 0.25).sum()),
           "gsea_genes_BH_lt_0.05": int((GS.q_BH_gene_shuffle < 0.05).sum()),
           "gsea_significant_with_panel_gene": int(((GS.q_gsea_gene_shuffle < 0.25) & (GS.n_panel > 0)).sum()),
           "ssgsea_BH_lt_0.05": int((SSR.q_BH < 0.05).sum()), "ssgsea_p_lt_0.05": int((SSR.p < 0.05).sum()), "in_both": len(both),
           "permutations": int(B_PERM), "sources": SETS.source.value_counts().to_dict()},
          open(OUT / "gsea_summary.json", "w"), indent=1, default=float)
log(f"{len(both)} pathways significant in both analyses")
print(sorted(both)[:30])''')

write_nb(HERE / "PD_LCM_rf_gsea.ipynb", CELLS, "gse")
