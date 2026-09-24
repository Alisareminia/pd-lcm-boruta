"""Collects everything the 30-gene table needs: our own results, gene annotation, and the
literature. Writes gene_master.csv (one row per gene) and gene_evidence.bib.

Annotation comes from MyGene.info, the PD literature from Europe PMC; Open Targets scores were
already computed by the enrichment notebook. Nothing here refits a model.
"""
import json, re, time, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "classifier" / "outputs_rf"
HERE = Path(__file__).resolve().parent


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return json.loads(r.read().decode())
        except Exception as exc:
            print("   retry", k + 1, type(exc).__name__)
            time.sleep(4)
    return None


# ---------------------------------------------------------------- our own results
ENR = pd.read_csv(OUT / "enrichment" / "enrich_gene_table.csv")
CMP = pd.read_csv(OUT / "pathcomp" / "composition_gene_effects.csv")[
    ["gene", "lineage", "g_original", "g_composition_removed", "retained"]]
ANN = pd.read_csv(OUT / "panel" / "13_gene_annotation_master.csv")[
    ["gene", "log2FC", "pvalue", "I2", "same_direction_datasets", "boruta_fold_frequency", "shap_rank",
     "single_gene_auc", "is_deg"]]
G = ENR.merge(CMP, on="gene", how="left").merge(ANN, on="gene", how="left", suffixes=("", "_ann"))

# which significant pathway each gene carries
GPR = pd.read_csv(OUT / "pathcomp" / "gprofiler_results.csv")
SIG = GPR[(GPR.p_adj < 0.05) & (GPR.list == "all")]
paths = {}
for r in SIG.itertuples():
    for s in str(r.genes).split(", "):
        paths.setdefault(s, []).append(r.name)
G["pathways"] = [", ".join(paths.get(s, [])) for s in G.symbol]

# ---------------------------------------------------------------- annotation
loc, pos, full = {}, {}, {}
for s in G.symbol:
    q = get(f"https://mygene.info/v3/query?q=symbol:{s}&species=human&fields=map_location,name,genomic_pos")
    hit = (q or {}).get("hits", [{}])[0] if (q or {}).get("hits") else {}
    if not hit:      # withdrawn or renamed symbol: ask HGNC, which keeps the previous names
        h = get(f"https://rest.genenames.org/search/prev_symbol/{s}") or {}
        docs = h.get("response", {}).get("docs", [])
        if docs:
            f2 = get(f"https://rest.genenames.org/fetch/hgnc_id/{docs[0]['hgnc_id'].split(':')[-1]}") or {}
            hit = {"map_location": (f2.get("response", {}).get("docs", [{}])[0]).get("location", ""),
                   "name": (f2.get("response", {}).get("docs", [{}])[0]).get("name", "")}
    loc[s] = hit.get("map_location", "")
    gp = hit.get("genomic_pos")
    gp = gp[0] if isinstance(gp, list) else (gp or {})
    pos[s] = f"chr{gp.get('chr', '?')}:{gp.get('start', 0) / 1e6:.2f}~Mb" if gp else ""
    full[s] = hit.get("name", "")
    print("annotated", s, loc[s])
G["locus"], G["position"], G["full_name"] = G.symbol.map(loc), G.symbol.map(pos), G.symbol.map(full)

# ---------------------------------------------------------------- the PD literature
KEY = {}
rows = []
for s in G.symbol:
    qs = urllib.parse.quote(f'"{s}" AND "Parkinson"')
    res = get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={qs}&format=json&pageSize=6&sort=CITED%20desc")
    hits = ((res or {}).get("resultList") or {}).get("result", [])
    n = (res or {}).get("hitCount", 0)
    pick = None
    for h in hits:                                   # the most cited paper that really is about the gene
        title = h.get("title", "")
        if h.get("doi") and h.get("authorString") and h.get("journalTitle"):
            pick = h
            if s.lower() in title.lower():
                break
    if pick:
        first = pick["authorString"].split(",")[0].split()[0]
        key = re.sub(r"[^a-z]", "", f"{first}{pick.get('pubYear', '')}".lower()) or f"{s.lower()}ref"
        while key in KEY and KEY[key] != pick.get("doi"):
            key += "a"
        KEY[key] = pick.get("doi")
        rows.append(dict(symbol=s, n_pd_papers=n, key=key, title=pick.get("title", "").rstrip("."),
                         authors=pick.get("authorString", "").rstrip("."), journal=pick.get("journalTitle", ""),
                         year=pick.get("pubYear", ""), doi=pick.get("doi", ""), volume=pick.get("journalVolume", ""),
                         pages=pick.get("pageInfo", "")))
    else:
        rows.append(dict(symbol=s, n_pd_papers=n, key="", title="", authors="", journal="", year="", doi="",
                         volume="", pages=""))
    print("literature", s, n, rows[-1]["key"])
LIT = pd.DataFrame(rows)
G = G.merge(LIT, on="symbol", how="left")
G.to_csv(HERE.parent / "gene_master.csv", index=False)

with open(HERE.parent / "gene_evidence.bib", "w") as f:
    seen = set()
    for r in LIT.itertuples():
        if not r.key or r.key in seen:
            continue
        seen.add(r.key)
        au = " and ".join(a.strip() for a in str(r.authors).split(",")[:8] if a.strip())
        f.write(f"@article{{{r.key}, title={{{{{r.title}}}}}, author={{{au}}}, journal={{{r.journal}}}, "
                f"year={{{r.year}}}, volume={{{r.volume}}}, pages={{{r.pages}}}, "
                f"doi={{{r.doi}}}, url={{https://doi.org/{r.doi}}} }}\n\n")
print(f"\n{len(G)} genes; {LIT.key.ne('').sum()} with a citable PD paper")
