import os, re, json, glob, time, textwrap, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
ON_KAGGLE = Path("/kaggle/input").exists()
OUT = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("FIG_OUT", "."))
FIG_IN = Path(os.environ["GSEA_IN"]) if not ON_KAGGLE else Path(sorted(
    {str(Path(h).parent) for h in glob.glob("/kaggle/input/**/gsea_results.csv", recursive=True)}, key=len)[0])
GS = pd.read_csv(FIG_IN / "gsea_results.csv")
SSR = pd.read_csv(FIG_IN / "ssgsea_results.csv")
SSC = pd.read_csv(FIG_IN / "ssgsea_scores.csv")
RANK = pd.read_csv(FIG_IN / "gsea_gene_ranking.csv").sort_values("metric", ascending=False).reset_index(drop=True)
SUM = json.load(open(FIG_IN / "gsea_summary.json"))
Q_THR = 0.25
TOP = GS[GS.q_gsea_gene_shuffle < Q_THR].sort_values("q_gsea_gene_shuffle").reset_index(drop=True)

# gene-set membership, needed only for the running-score curves
LIBS = {"MSigDB_Hallmark_2020": "Hallmark", "Reactome_2022": "Reactome", "KEGG_2021_Human": "KEGG",
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
                    nm = re.sub(r"\s*\(GO:\d+\)$", "", p[0].strip()); nm = re.sub(r"\s+R-HSA-\d+$", "", nm)
                    out[nm] = {g.split(",")[0].strip().upper() for g in p[2:] if g.strip()}
            return out
        except Exception as exc:
            print("  Enrichr retry", k + 1, type(exc).__name__); time.sleep(8)
    raise RuntimeError(name)
MEMBERS = {}
if ON_KAGGLE:
    for lib in LIBS:
        for nm, genes in enrichr_lib(lib).items():
            MEMBERS.setdefault(nm, genes)
else:
    rs = np.random.default_rng(0)
    for nm in TOP.set:
        MEMBERS[nm] = set(rs.choice(RANK.symbol.dropna().str.upper().to_numpy(), 40, replace=False))
SYMS = RANK.symbol.astype(str).str.upper().to_numpy()
METRIC = RANK.metric.to_numpy()
G = len(RANK)
def running(nm):
    """Running enrichment score along the ranked gene list, and the positions of the set's genes."""
    mem = np.isin(SYMS, list(MEMBERS.get(nm, set()))).astype(float)
    if mem.sum() < 2:
        return None, None
    w = np.abs(METRIC) * mem
    hit = np.cumsum(w) / w.sum()
    miss = np.cumsum(1 - mem) / (G - mem.sum())
    return hit - miss, np.flatnonzero(mem)
wrap = lambda t, w: textwrap.wrap(t, w)
print(f"{len(TOP)} gene sets at GSEA FDR < {Q_THR}; membership resolved for "
      f"{sum(1 for n in TOP.set if n in MEMBERS)}/{len(TOP)}")
