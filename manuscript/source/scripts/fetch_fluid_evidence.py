"""Asks, for each panel gene, whether anyone has measured it in blood or cerebrospinal fluid in
Parkinson's disease. A hit counts only when the gene, a fluid and the disease appear together in the
title or abstract; everything else is recorded as nothing found, which is the answer for most genes."""
import json, re, time, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
M = pd.read_csv(HERE / "gene_master.csv")
FLUID = re.compile(r"\bplasma\b|\bserum\b|cerebrospinal fluid|\bCSF\b|\bblood\b|peripheral blood", re.I)
PD = re.compile(r"parkinson", re.I)


def search(q, page=25):
    u = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=" + urllib.parse.quote(q)
         + f"&format=json&pageSize={page}&resultType=core&sort=CITED%20desc")
    for k in range(4):
        try:
            with urllib.request.urlopen(u, timeout=60) as r:
                return (json.loads(r.read().decode()).get("resultList") or {}).get("result", [])
        except Exception as exc:
            print("   retry", k + 1, type(exc).__name__); time.sleep(5)
    return []


def journal_of(h):
    return h.get("journalTitle") or (((h.get("journalInfo") or {}).get("journal") or {}).get("title", ""))


rows = []
for r in M.itertuples():
    s = r.symbol
    best = None
    for h in search(f'"{s}" AND ("plasma" OR "serum" OR "cerebrospinal fluid" OR "CSF") AND "Parkinson"'):
        text = (h.get("title", "") or "") + " " + (h.get("abstractText") or "")
        named = re.search(rf"(?<![A-Za-z0-9]){re.escape(s)}(?![A-Za-z0-9])", text, re.I)
        if named and FLUID.search(text) and PD.search(text) and h.get("doi") and h.get("authorString"):
            best = h
            break
    if best is None:
        rows.append(dict(symbol=s, key="", title="", doi="", year="", journal="", authors="", snippet=""))
        print(f"{s:9s} -")
        continue
    ab = re.sub(r"<[^>]+>", "", best.get("abstractText") or "")
    keep = [t.strip() for t in re.split(r"(?<=[.!?])\s+", ab)
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(s)}(?![A-Za-z0-9])", t, re.I) and (FLUID.search(t) or PD.search(t))]
    first = re.sub(r"[^a-z]", "", best["authorString"].split(",")[0].split()[0].lower()) or "anon"
    rows.append(dict(symbol=s, key=f"{first}{best['pubYear']}", title=re.sub(r"<[^>]+>", "", best["title"]).rstrip("."),
                     doi=best["doi"], year=best["pubYear"], journal=journal_of(best),
                     authors=best["authorString"].rstrip("."), snippet=" ".join(keep[:2])[:400]))
    print(f"{s:9s} [{rows[-1]['key']}] {rows[-1]['title'][:80]}")
    print(f"           {rows[-1]['snippet'][:220]}")

F = pd.DataFrame(rows)
F.to_csv(HERE / "fluid_evidence_raw.csv", index=False)
print(f"\n{F.doi.ne('').sum()} of {len(F)} genes have a fluid measurement in Parkinson's disease")
