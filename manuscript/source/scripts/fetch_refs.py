"""Builds the extra bibliography paper 2 needs: the source paper of every external GEO series
(resolved through NCBI), plus the method and resource papers, fetched from CrossRef by DOI so the
entries carry real volumes, pages and links."""
import json, re, time, urllib.parse, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "PD-LCM-manuscript/1.0 (mailto:a.saremi.med@gmail.com)"}

GSE = ["GSE7621", "GSE8397", "GSE20163", "GSE20164", "GSE20292", "GSE49036", "GSE114517", "GSE168496"]

DOIS = {                                            # method and resource papers, by DOI
    "subramanian2005": "10.1073/pnas.0506580102",
    "barbie2009": "10.1038/nature08460",
    "liberzon2015": "10.1016/j.cels.2015.12.004",
    "kolberg2023": "10.1093/nar/gkad347",
    "milacic2024": "10.1093/nar/gkad1025",
    "kanehisa2023": "10.1093/nar/gkac963",
    "agrawal2024": "10.1093/nar/gkad960",
    "oleary2016": "10.1093/nar/gkv1189",
    "poulin2020": "10.1016/j.tins.2020.01.004",
    "blauwendraat2020": "10.1016/S1474-4422(19)30287-X",
    "kuleshov2016": "10.1093/nar/gkw377",
    "efron1979": "10.1214/aos/1176344552",
    "delong1988": "10.2307/2531595",
    "grubert2020": "10.1038/s41586-020-2151-x",
}


def fetch(url):
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45) as r:
                return r.read().decode()
        except Exception as exc:
            print("   retry", k + 1, type(exc).__name__); time.sleep(4)
    return ""


def crossref(doi):
    txt = fetch(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    return json.loads(txt)["message"] if txt else None


def entry(key, m):
    au = " and ".join(f"{a.get('family', '')}, {a.get('given', '')}".strip(", ")
                      for a in (m.get("author") or [])[:10] if a.get("family"))
    year = (m.get("issued", {}).get("date-parts") or [[""]])[0][0]
    title = (m.get("title") or [""])[0].replace("{", "").replace("}", "")
    jour = (m.get("container-title") or [""])[0]
    return (f"@article{{{key}, title={{{{{title}}}}}, author={{{au}}}, journal={{{jour}}}, year={{{year}}}, "
            f"volume={{{m.get('volume', '')}}}, number={{{m.get('issue', '')}}}, pages={{{m.get('page', '')}}}, "
            f"doi={{{m.get('DOI', '')}}}, url={{https://doi.org/{m.get('DOI', '')}}} }}\n\n")


out, gse_map = [], {}
for acc in GSE:                                     # GEO series -> its PubMed paper -> CrossRef
    txt = fetch(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gds&term={acc}[ACCN]&retmode=json")
    ids = json.loads(txt)["esearchresult"]["idlist"] if txt else []
    pmid, title = "", ""
    for uid in ids[:3]:
        s = fetch(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gds&id={uid}&retmode=json")
        if not s:
            continue
        rec = json.loads(s)["result"][uid]
        if rec.get("accession") != acc:
            continue
        title = rec.get("title", "")
        pmids = rec.get("pubmedids") or []
        pmid = str(pmids[0]) if pmids else ""
        break
    key, cited = "", ""
    if pmid:
        px = fetch(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=json")
        rec = json.loads(px)["result"][pmid] if px else {}
        doi = next((x["value"] for x in rec.get("articleids", []) if x["idtype"] == "doi"), "")
        first = (rec.get("sortfirstauthor") or rec.get("lastauthor") or "anon").split()[0].lower()
        key = re.sub(r"[^a-z]", "", first) + str(rec.get("pubdate", "")[:4])
        m = crossref(doi) if doi else None
        if m:
            out.append(entry(key, m)); cited = m.get("title", [""])[0]
        else:
            au = " and ".join(rec.get("authors", [{}])[i].get("name", "") for i in range(min(6, len(rec.get("authors", [])))))
            out.append(f"@article{{{key}, title={{{{{rec.get('title', '').rstrip('.')}}}}}, author={{{au}}}, "
                       f"journal={{{rec.get('fulljournalname', '')}}}, year={{{rec.get('pubdate', '')[:4]}}}, "
                       f"volume={{{rec.get('volume', '')}}}, pages={{{rec.get('pages', '')}}}, doi={{{doi}}}, "
                       f"url={{https://doi.org/{doi}}} }}\n\n")
            cited = rec.get("title", "")
    gse_map[acc] = dict(key=key, geo_title=title, pmid=pmid, paper=cited)
    print(f"{acc:10s} pmid={pmid or '-':10s} key={key or '-':16s} {title[:60]}")

for key, doi in DOIS.items():
    m = crossref(doi)
    if m:
        out.append(entry(key, m)); print(f"{key:18s} {(m.get('title') or [''])[0][:66]}")
    else:
        print(f"{key:18s} NOT FOUND {doi}")

(HERE / "refs_extra.bib").write_text("".join(out))
(HERE / "gse_sources.json").write_text(json.dumps(gse_map, indent=1))
print(f"\nwrote refs_extra.bib with {len(out)} entries")
