"""Builds PD_LCM_enr_probe.ipynb - a short look at the external resources the enrichment notebook needs.

Prints, without analysing anything: the supplementary files of Kamath et al. 2022 (human dopamine-neuron subtypes)
through Europe PMC, the Enrichr gene-set libraries that exist, and whether Open Targets and mygene.info answer.
"""
import pathlib
from kaggle_nb import write_nb
HERE = pathlib.Path(__file__).parent
CELLS = []
def code(s): CELLS.append(("code", s))
CELLS.append(("markdown", "# Probe: resources for the enrichment notebook"))

code(r'''import json, io, re, zipfile, urllib.request, urllib.parse
from pathlib import Path
import pandas as pd
D = Path("/kaggle/working/probe"); D.mkdir(parents=True, exist_ok=True)
def get(url, data=None, headers=None, timeout=180):
    req = urllib.request.Request(url, data=data, headers=headers or {"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return fh.read()''')

code(r'''# ---- Kamath et al. 2022 (Nat Neurosci 25:588), via Europe PMC ----
for doi in ("10.1038/s41593-022-01061-1",):
    try:
        j = json.loads(get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&resultType=core&query="
                           + urllib.parse.quote(f'DOI:"{doi}"')))
        hits = j["resultList"]["result"]
        for h in hits:
            print(doi, "->", h.get("pmcid"), h.get("pmid"), h.get("title", "")[:120])
        pmcid = hits[0].get("pmcid")
    except Exception as exc:
        print("search failed", type(exc).__name__, exc); pmcid = None
if pmcid:
    try:
        z = get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles", timeout=600)
        print("supplementary zip:", len(z) / 1e6, "MB")
        zf = zipfile.ZipFile(io.BytesIO(z))
        for n in zf.namelist():
            print("  ", n, zf.getinfo(n).file_size)
        for n in zf.namelist():
            if n.lower().endswith((".xlsx", ".xls")):
                (D / Path(n).name).write_bytes(zf.read(n))
                try:
                    xl = pd.ExcelFile(D / Path(n).name)
                    for sh in xl.sheet_names:
                        df = xl.parse(sh, header=None, nrows=8)
                        print(f"== {n} | sheet '{sh}' | shape preview {df.shape}")
                        print(df.iloc[:8, :10].to_string()[:1500])
                except Exception as exc:
                    print("  cannot read", n, type(exc).__name__, exc)
            elif n.lower().endswith((".csv", ".tsv", ".txt")):
                print(f"== {n}"); print(zf.read(n)[:800].decode("utf-8", "replace"))
    except Exception as exc:
        print("supplementary download failed", type(exc).__name__, exc)''')

code((HERE / "figcode" / "strict_xlsx.py").read_text())
code(r'''SH = read_strict_xlsx(D / "41593_2022_1061_MOESM3_ESM.xlsx")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
for name, df in SH.items():
    print(f"==== {name}: {df.shape}")
    print(df.iloc[:7, :12].to_string()[:2500])
    if df.shape[1] > 0:
        for j in range(min(df.shape[1], 12)):
            vals = df.iloc[1:, j].dropna().astype(str)
            if 0 < vals.nunique() <= 40:
                print(f"   col {j} values: {sorted(vals.unique())[:40]}")''')

write_nb(HERE / "PD_LCM_enr_probe.ipynb", CELLS, "enp")
