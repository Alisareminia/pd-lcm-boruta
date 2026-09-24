import os, re, json, glob, sys, subprocess, textwrap
from pathlib import Path
import numpy as np, pandas as pd
ON_KAGGLE = Path("/kaggle/input").exists()
OUT = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("FIG_OUT", "."))
def fin(name):
    if not ON_KAGGLE:
        return Path(os.environ["ENR_IN"]) / name
    hits = sorted((h for h in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if "rf-enrichment" in h), key=len)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]
try:
    from adjustText import adjust_text
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "adjustText"], check=True)
    from adjustText import adjust_text

BGL = pd.read_csv(fin("enrich_background_lineage.csv"))
PLN = pd.read_csv(fin("enrich_panel_lineage.csv"))
SBT = pd.read_csv(fin("enrich_subtypes.csv"))
ORT = pd.read_csv(fin("enrich_ora.csv"))
NBT = pd.read_csv(fin("enrich_pathway_neighbours.csv"))
GTB = pd.read_csv(fin("enrich_gene_table.csv"))
SMY = json.load(open(fin("enrich_summary.json")))
SU = SMY["subtypes"]
NICE = dict(zip(GTB.symbol.str.upper(), GTB.symbol))                     # upper-case symbol -> official capitalisation
nice = lambda s: NICE.get(str(s).upper(), s)
PLN["name"] = PLN.symbol.map(nice)
PLN = PLN.merge(GTB[["gene", "shap_rank"]], on="gene", how="left")
DOWN = PLN[PLN.direction == "down"].sort_values("lineage").name.tolist()
UP = PLN[PLN.direction == "up"].sort_values("lineage").name.tolist()
DIRN = dict(zip(PLN.name, PLN.direction))

# Kamath et al. 2022, Supplementary Table 8: marker z of every panel gene in every dopamine-neuron subtype
T8 = read_strict_xlsx(fin("kamath_2022_supplementary.xlsx"))["Supplementary_Table_8"]
T8.columns = [str(c) if c is not None else f"c{i}" for i, c in enumerate(T8.iloc[0])]; T8 = T8.iloc[1:]
T8["z"] = pd.to_numeric(T8["z"], errors="coerce"); T8["symbol"] = T8.primerid.astype(str).str.upper()
SUBTYPES = ["SOX6_AGTR1", "SOX6_PART1", "SOX6_DDT", "SOX6_GFRA2", "CALB1_CALCR", "CALB1_CRYM_CCDC68", "CALB1_GEM",
            "CALB1_PPP1R17", "CALB1_RBP4", "CALB1_TRHR"]
ZP = (T8.pivot_table(index="symbol", columns="DA_subtype", values="z", aggfunc="first")
        .reindex(index=[g.upper() for g in DOWN + UP], columns=SUBTYPES).fillna(0.0))
ZP.index = DOWN + UP

LIBTAG = {"GO:BP": "GO BP", "GO:CC": "GO CC", "GO:MF": "GO MF", "Reactome": "Reactome", "KEGG": "KEGG", "WikiPathways": "WikiPW"}
def term_name(t):
    """Full pathway name, identifiers removed, sentence case with acronyms kept."""
    t = re.sub(r"\s*\((GO:\d+)\)$", "", t); t = re.sub(r"\s+R-HSA-\d+$", "", t); t = re.sub(r"\s+WP\d+$", "", t)
    words = t.split(" ")
    out = [words[0]] + [w if (w.isupper() and len(w) > 1) or re.search(r"\d", w) else w.lower() for w in words[1:]]
    t = " ".join(out)
    for a in ("dna", "rna", "gpcrs", "gpcr", "hsf1", "h4/h2a", "h2a", "h4"):
        t = re.sub(rf"(?i)\b{re.escape(a)}\b", a.upper() if a not in ("gpcrs",) else "GPCRs", t)
    return t
wrap = lambda t, w: textwrap.wrap(t, w)
print(f"panel: {len(DOWN)} down, {len(UP)} up; Kamath markers for the panel: {int((ZP > 0).sum().sum())} gene-subtype pairs")
