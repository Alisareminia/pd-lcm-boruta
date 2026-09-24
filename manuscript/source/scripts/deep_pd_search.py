"""A documented, re-runnable literature search for every panel gene, so the published-link column rests
on a stated protocol rather than on whoever did the searching.

Protocol (Europe PMC, run on the date written to the output):
  ABSTRACT:"<symbol>" AND (ABSTRACT:"Parkinson" OR ABSTRACT:"dopaminergic" OR ABSTRACT:"substantia nigra"
                           OR ABSTRACT:"synuclein")
The gene must be named as a whole word in the title or abstract, next to a Parkinson's or dopamine-neuron
term in the same sentence. The sentences are printed so each candidate can be read before it is used.
"""
import datetime, json, re, time, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
GENES = pd.read_csv(HERE / "gene_master.csv").symbol.tolist()
PD = re.compile(r"parkinson|dopaminergic|dopamine neuron|substantia nigra|synuclein|nigrostriatal|MPTP|"
                r"6-OHDA|rotenone|Lewy", re.I)


def search(q, page=25):
    u = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=" + urllib.parse.quote(q)
         + f"&format=json&pageSize={page}&resultType=core")
    for k in range(4):
        try:
            with urllib.request.urlopen(u, timeout=60) as r:
                d = json.loads(r.read().decode())
            return d.get("hitCount", 0), (d.get("resultList") or {}).get("result", [])
        except Exception as exc:
            print("   retry", k + 1, type(exc).__name__); time.sleep(5)
    return 0, []


def jr(h):
    return h.get("journalTitle") or (((h.get("journalInfo") or {}).get("journal") or {}).get("title", ""))


rows = []
for s in GENES:
    q = (f'ABSTRACT:"{s}" AND (ABSTRACT:"Parkinson" OR ABSTRACT:"dopaminergic" OR '
         f'ABSTRACT:"substantia nigra" OR ABSTRACT:"synuclein")')
    n, hits = search(q)
    tok = re.compile(rf"(?<![A-Za-z0-9]){re.escape(s)}(?![A-Za-z0-9])", re.I)
    kept = []
    for h in hits:
        ab = re.sub(r"<[^>]+>", "", h.get("abstractText") or "")
        sents = [t.strip() for t in re.split(r"(?<=[.!?])\s+", ab) if tok.search(t) and PD.search(t)]
        if sents and h.get("doi"):
            kept.append((h, sents))
    print(f"\n######## {s}: {n} abstracts match the query, {len(kept)} with the gene and a PD term in one sentence")
    for h, sents in kept[:6]:
        title = re.sub(r"<[^>]+>", "", h.get("title", ""))
        first = h.get("authorString", "?").split(",")[0]
        print(f"  [{first} {h.get('pubYear')}] {title[:110]}")
        print(f"      {jr(h)[:50]} | doi {h['doi']} | cited {h.get('citedByCount', 0)}")
        for t in sents[:2]:
            print(f"      > {t[:300]}")
        rows.append(dict(symbol=s, n_matching=n, title=title, first_author=first, year=h.get("pubYear"),
                         journal=jr(h), doi=h["doi"], cited=h.get("citedByCount", 0),
                         authors=h.get("authorString", ""), sentences=" || ".join(sents[:3])))
    time.sleep(0.4)

out = pd.DataFrame(rows)
out["searched"] = datetime.date.today().isoformat()
out.to_csv(HERE / "deep_pd_search.csv", index=False)
print(f"\nwrote deep_pd_search.csv: {len(out)} candidate papers across {out.symbol.nunique()} genes")
