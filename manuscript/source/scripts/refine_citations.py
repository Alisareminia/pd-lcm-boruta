"""Keeps a citation only when the paper is really about that gene, and says honestly what tier of
evidence it is.

The naive search has two failure modes: acronym collisions (SCAI matched a cardiology guideline
title listing societies) and free-text matches on "brain" that return cancer papers. So the symbol
must appear as a whole word in the title, guideline-style acronym lists are rejected, and the tier
is decided from the abstract, not from a full-text hit.
"""
import json, re, time, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
M = pd.read_csv(HERE.parent / "gene_master.csv")
PD_RE = re.compile(r"parkinson", re.I)
NEURO_RE = re.compile(r"dopamin|substantia nigra|neurodegener|neuron|synap|brain|nigral|microglia|astrocyt", re.I)


def search(q, page=25):
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=" + urllib.parse.quote(q)
           + f"&format=json&pageSize={page}&resultType=core&sort=CITED%20desc")
    for k in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                d = json.loads(r.read().decode())
            return d.get("hitCount", 0), (d.get("resultList") or {}).get("result", [])
        except Exception as exc:
            print("   retry", k + 1, type(exc).__name__); time.sleep(5)
    return 0, []


def journal_of(h):
    """resultType=core nests the journal; the lite result has it at the top level."""
    return h.get("journalTitle") or (((h.get("journalInfo") or {}).get("journal") or {}).get("title", ""))
def clean_hit(h, sym):
    """True when the title names this gene rather than borrowing its letters."""
    t = h.get("title", "")
    if not (h.get("doi") and h.get("authorString") and journal_of(h) and h.get("pubYear")):
        return False
    if not re.search(rf"(?<![A-Za-z0-9]){re.escape(sym)}(?![A-Za-z0-9])", t, re.I):
        return False
    if re.search(rf"[A-Z]{{2,}}/{re.escape(sym)}|{re.escape(sym)}/[A-Z]{{2,}}/", t):   # society acronym lists
        return False
    return True


rows = []
for r in M.itertuples():
    s = r.symbol
    n_pd, _ = search(f'"{s}" AND "Parkinson"', page=1)
    _, hits = search(f'TITLE:"{s}"')
    hits = [h for h in hits if clean_hit(h, s)]
    tier, best = "none", None
    pd_hits = [h for h in hits if PD_RE.search(h.get("title", "") + " " + (h.get("abstractText") or ""))]
    neuro_hits = [h for h in hits if NEURO_RE.search(h.get("title", ""))]      # title only: an abstract that
    if pd_hits:                                                                # merely mentions neurons is not
        tier, best = "pd", pd_hits[0]                                          # a nervous-system study
    elif neuro_hits:
        tier, best = "neuro", neuro_hits[0]
    if best is None and hits:
        tier, best = "other", hits[0]
    rows.append(dict(symbol=s, n_pd_papers=n_pd, evidence_kind=tier,
                     title=(best or {}).get("title", "").rstrip("."),
                     authors=(best or {}).get("authorString", "").rstrip("."),
                     journal=journal_of(best or {}), year=(best or {}).get("pubYear", ""),
                     doi=(best or {}).get("doi", ""), volume=(best or {}).get("journalVolume", "") or ((best or {}).get("journalInfo") or {}).get("volume", ""),
                     pages=(best or {}).get("pageInfo", "")))
    print(f"{s:9s} pd_mentions={n_pd:5d} {tier:6s} {rows[-1]['title'][:72]}")

def fix_author(a):
    """'Smith AB' -> 'Smith, A. B.', so BibTeX does not read the initials as the surname."""
    parts = a.split()
    if len(parts) >= 2 and re.fullmatch(r"[A-Z]{1,3}", parts[-1]):
        return f"{' '.join(parts[:-1])}, " + " ".join(f"{c}." for c in parts[-1])
    return a


R = pd.DataFrame(rows)
keys, used = [], {}
for r in R.itertuples():
    if not r.doi:
        keys.append(""); continue
    first = re.sub(r"[^a-z]", "", str(r.authors).split(",")[0].split()[0].lower()) or "anon"
    k = f"{first}{r.year}"
    while k in used and used[k] != r.doi:
        k += "a"
    used[k] = r.doi
    keys.append(k)
R["key"] = keys
M = M.drop(columns=[c for c in ("n_pd_papers", "key", "title", "authors", "journal", "year", "doi", "volume",
                                "pages", "evidence_kind") if c in M.columns]).merge(R, on="symbol", how="left")
M.to_csv(HERE.parent / "gene_master.csv", index=False)

with open(HERE.parent / "gene_evidence.bib", "w") as f:
    seen = set()
    for r in R.itertuples():
        if not r.key or r.key in seen:
            continue
        seen.add(r.key)
        au = " and ".join(fix_author(a.strip()) for a in str(r.authors).split(",")[:8] if a.strip())
        f.write(f"@article{{{r.key}, title={{{{{r.title}}}}}, author={{{au}}}, journal={{{r.journal}}}, "
                f"year={{{r.year}}}, volume={{{r.volume}}}, pages={{{r.pages}}}, doi={{{r.doi}}}, "
                f"url={{https://doi.org/{r.doi}}} }}\n\n")
print("\n" + R.evidence_kind.value_counts().to_string())
