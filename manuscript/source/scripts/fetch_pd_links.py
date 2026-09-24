"""Finds, for each panel gene, the best paper that actually ties it to Parkinson's disease biology.

Only papers that name the gene and a PD term in the title or abstract count. The script prints what it
found so the sentences in the table can be written from the abstracts rather than invented; genes with no
such paper get nothing, which is the honest answer for most of them.
"""
import json, re, time, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
M = pd.read_csv(HERE / "gene_master.csv")
PD = re.compile(r"parkinson|dopaminergic neuron|substantia nigra|α-synuclein|alpha-synuclein|synuclein|"
                r"nigrostriatal|MPTP|6-OHDA|Lewy", re.I)


def search(q, page=25):
    u = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=" + urllib.parse.quote(q)
         + f"&format=json&pageSize={page}&resultType=core&sort=CITED%20desc")
    for k in range(4):
        try:
            with urllib.request.urlopen(u, timeout=60) as r:
                d = json.loads(r.read().decode())
            return (d.get("resultList") or {}).get("result", [])
        except Exception as exc:
            print("   retry", k + 1, type(exc).__name__); time.sleep(5)
    return []


def journal_of(h):
    return h.get("journalTitle") or (((h.get("journalInfo") or {}).get("journal") or {}).get("title", ""))


def named(h, sym):
    text = (h.get("title", "") or "") + " " + (h.get("abstractText") or "")
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(sym)}(?![A-Za-z0-9])", text, re.I) is not None


rows = []
for r in M.itertuples():
    s = r.symbol
    hits = []
    for q in (f'TITLE:"{s}" AND (TITLE:"Parkinson" OR ABSTRACT:"Parkinson")',
              f'"{s}" AND ("Parkinson" OR "dopaminergic neuron" OR "alpha-synuclein")'):
        for h in search(q):
            if not (h.get("doi") and h.get("authorString") and journal_of(h) and h.get("pubYear")):
                continue
            if named(h, s) and PD.search((h.get("title", "") or "") + " " + (h.get("abstractText") or "")):
                hits.append(h)
        if hits:
            break
    if not hits:
        rows.append(dict(symbol=s, doi="", key="", title="", year="", journal="", authors="", abstract=""))
        print(f"\n### {s}: nothing linking it to PD")
        continue
    h = hits[0]
    first = re.sub(r"[^a-z]", "", h["authorString"].split(",")[0].split()[0].lower()) or "anon"
    ab = re.sub(r"<[^>]+>", "", h.get("abstractText") or "")
    keep = [t.strip() for t in re.split(r"(?<=[.!?])\s+", ab)
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(s)}(?![A-Za-z0-9])", t, re.I) and PD.search(t)]
    rows.append(dict(symbol=s, doi=h["doi"], key=f"{first}{h['pubYear']}", title=re.sub(r"<[^>]+>", "", h["title"]),
                     year=h["pubYear"], journal=journal_of(h), authors=h["authorString"], abstract=" ".join(keep[:3])))
    print(f"\n### {s}  [{first}{h['pubYear']}] {re.sub(r'<[^>]+>', '', h['title'])[:95]}")
    print("   ", (" ".join(keep[:2]) or ab)[:420])

pd.DataFrame(rows).to_csv(HERE / "pd_links_raw.csv", index=False)
print(f"\n{sum(1 for r in rows if r['doi'])} of {len(rows)} genes have a PD-linked paper")
