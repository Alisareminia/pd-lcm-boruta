"""Second, deeper literature pass for the genes that still have no published link.

Widens the first protocol in three ways, and records everything it finds:
  1. every HGNC alias and previous symbol, plus the approved name, instead of the symbol alone;
  2. a wider disease vocabulary (Parkinson, parkinsonism, Lewy, synuclein, LRRK2, PINK1, parkin, GBA,
     dopaminergic neuron, substantia nigra, MPTP, rotenone, 6-OHDA, paraquat);
  3. full text of open-access papers, read sentence by sentence, because a gene often appears in a
     results paragraph and never in the abstract.
A candidate is kept when one sentence names the gene (or an alias) and a disease term together.
Aliases that are ordinary words or clash with common acronyms are dropped before searching.
"""
import datetime, json, re, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
GENES = ["RBM4B", "FUS", "ZMYND8", "TRRAP", "PPFIA4", "ICE1", "PRKAR2A", "GSTK1", "CFLAR", "L3MBTL1",
         "ADGRB3", "CPNE3", "CNTN4", "SCAI", "SH3GL3", "C11orf54", "IBTK", "SSTR1", "USP12", "CALB2"]
DISEASE = ["Parkinson", "parkinsonism", "Lewy", "synuclein", "LRRK2", "PINK1", "parkin", "glucocerebrosidase",
           "dopaminergic neuron", "substantia nigra", "MPTP", "rotenone", "6-OHDA", "paraquat"]
DIS_RE = re.compile(r"parkinson|lewy|synuclein|LRRK2|PINK1|\bparkin\b|glucocerebrosidase|dopaminergic|"
                    r"substantia nigra|nigrostriatal|MPTP|rotenone|6-OHDA|6-hydroxydopamine|paraquat", re.I)
# aliases that collide with ordinary words or unrelated acronyms (focused ultrasound, calretinin 'CR', ...)
BLOCK = {"FUS", "TLS", "CR", "CAL2", "CASH", "ICE", "MRIT", "PAF400", "TRAP", "BAI3a", "SCAI", "HEL-S-164",
         "CLARP", "FLAME", "BTKI", "UBH1", "RACK7", "CNSA3", "EEN-B2", "PKR2", "L3MBTL", "ESTHASE"}
UA = {"User-Agent": "PD-manuscript/1.0 (mailto:a.saremi.med@gmail.com)"}


def get(url, accept="application/json", tries=3, raw=False):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Accept": accept})
            with urllib.request.urlopen(req, timeout=90) as r:
                body = r.read().decode("utf-8", errors="replace")
            return body if raw else json.loads(body)
        except Exception:
            time.sleep(4 * (k + 1))
    return None


def aliases(sym):
    d = get(f"https://rest.genenames.org/fetch/symbol/{sym}") or {}
    docs = ((d.get("response") or {}).get("docs")) or [{}]
    doc = docs[0] if docs else {}
    names = [sym] + doc.get("alias_symbol", []) + doc.get("prev_symbol", [])
    names = [n for n in dict.fromkeys(names) if n not in BLOCK and len(n) >= 4]
    if sym == "FUS":                                   # the symbol itself is unusable; the protein name is not
        names = ["fused in sarcoma", "FUS/TLS", "FUS protein"]
    if sym == "SCAI":
        names = ["suppressor of cancer cell invasion"]
    return names, doc.get("name", "")


def epmc(q, page=25, extra=""):
    u = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=" + urllib.parse.quote(q)
         + f"&format=json&pageSize={page}&resultType=core{extra}")
    d = get(u) or {}
    return d.get("hitCount", 0), ((d.get("resultList") or {}).get("result")) or []


def sentences_with(text, pats):
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        if any(p.search(s) for p in pats) and DIS_RE.search(s):
            out.append(s.strip())
    return out


rows = []
for i, sym in enumerate(GENES, 1):
    print(f"[{i}/{len(GENES)}] {sym} ...", flush=True)
    names, approved = aliases(sym)
    pats = [re.compile(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", re.I) for n in names]
    gene_q = " OR ".join(f'"{n}"' for n in names)
    dis_q = " OR ".join(f'"{t}"' for t in DISEASE)

    # 1. titles and abstracts, all aliases, wide disease vocabulary
    n_abs, hits = epmc(f'(TITLE:({gene_q}) OR ABSTRACT:({gene_q})) AND ({dis_q})', page=40)
    found = []
    for h in hits:
        text = re.sub(r"<[^>]+>", "", (h.get("title") or "") + ". " + (h.get("abstractText") or ""))
        ss = sentences_with(text, pats)
        if ss and h.get("doi"):
            found.append(("abstract", h, ss))
    time.sleep(1.5)

    # 2. full text of open-access papers
    n_ft, hits = epmc(f'({gene_q}) AND ({dis_q}) AND OPEN_ACCESS:y', page=12)
    for h in hits[:12]:
        pmcid = h.get("pmcid")
        if not pmcid or any(f[1].get("doi") == h.get("doi") for f in found):
            continue
        xml = get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML", accept="application/xml",
                  raw=True)
        if not xml:
            continue
        try:
            body = " ".join(ET.fromstring(xml).itertext())
        except ET.ParseError:
            continue
        ss = sentences_with(re.sub(r"\s+", " ", body), pats)
        if ss and h.get("doi"):
            found.append(("full text", h, ss))
        time.sleep(1.0)

    print(f"\n######## {sym} ({approved}) aliases: {', '.join(names)}")
    print(f"   abstract hits {n_abs}, open-access full-text hits {n_ft}, sentences found in {len(found)} papers")
    for where, h, ss in found[:8]:
        title = re.sub(r"<[^>]+>", "", h.get("title", ""))
        print(f"  [{where}] [{(h.get('authorString') or '?').split(',')[0]} {h.get('pubYear')}] {title[:105]}")
        print(f"      doi {h.get('doi')} | cited {h.get('citedByCount', 0)}")
        for s in ss[:2]:
            print(f"      > {s[:290]}")
        rows.append(dict(symbol=sym, where=where, title=title, year=h.get("pubYear"), doi=h.get("doi"),
                         authors=h.get("authorString", ""), cited=h.get("citedByCount", 0),
                         sentences=" || ".join(ss[:3])))
    time.sleep(1.5)

out = pd.DataFrame(rows)
out["searched"] = datetime.date.today().isoformat()
out.to_csv(HERE / "deep_pd_search2.csv", index=False)
print(f"\nwrote deep_pd_search2.csv: {len(out)} candidate papers across {out.symbol.nunique() if len(out) else 0} genes")
