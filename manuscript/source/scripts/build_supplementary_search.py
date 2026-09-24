"""Supplementary Table S1: the whole literature search behind the published-link column of Table 3.

Every paper the automated passes returned is listed with the sentence that matched and the decision
taken, and every paper the table cites is listed too, matched by DOI, with the search that found it.
A reader can therefore check both the links we report and the blanks we leave.
"""
import re
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RETRACTED = {"10.1016/j.omtn.2020.07.019"}          # Cai 2020, SSTR1: retracted in 2022, never cited

# the keys Table 3 cites, read from the table builder so the two can never disagree
src = (HERE / "scripts" / "build_table3.py").read_text()
start = src.index("PRIOR = {")
cited = set(re.findall(r'"([a-z]+\d{4}[a-z]?)"\)', src[start:src.index("def prior(r):")]))
cited |= set(re.findall(r'"([a-z]+\d{4}[a-z]?)"\),?\s*\n', src[src.index("FLUID_TEXT = {"):src.index("def prior(r):")]))
cited.discard("uniprot2025")
fluid = pd.read_csv(HERE / "fluid_evidence.csv").fillna("")        # fluid rows whose key lives in this file
cited |= {k for k in fluid.key if k}

bib = (HERE / "references.bib").read_text(encoding="utf-8", errors="replace")


def bib_field(key, field):
    m = re.search(r"@\w+\{" + re.escape(key) + r",(.*?)\n\s*\n", bib + "\n\n", re.S)
    if not m:
        return ""
    f = re.search(field + r"\s*=\s*\{+(.*?)\}+\s*,", m.group(1), re.S | re.I)
    return re.sub(r"\s+", " ", f.group(1)).strip() if f else ""


CITED = {k: dict(doi=bib_field(k, "doi").lower(), title=bib_field(k, "title"), year=bib_field(k, "year"))
         for k in sorted(cited)}
CITED_DOI = {v["doi"]: k for k, v in CITED.items() if v["doi"]}

frames = []
for f, pass_name in (("deep_pd_search.csv", "pass 1: symbol, title and abstract"),
                     ("deep_pd_search2.csv", "pass 2: all HGNC aliases, wider disease terms, open-access full text")):
    if (HERE / f).exists():
        d = pd.read_csv(HERE / f)
        d["search_pass"] = pass_name
        frames.append(d)
S = pd.concat(frames, ignore_index=True, sort=False)
S["doi"] = S.doi.astype(str).str.lower()
S = S.drop_duplicates(["symbol", "doi"])


def decide(r):
    if r.doi in RETRACTED or "retraction notice" in str(r.title).lower():
        return "excluded", "retracted; not cited"
    if r.doi in CITED_DOI:
        return "kept", f"cited in Table 3 ({CITED_DOI[r.doi]})"
    t = f"{r.title} {r.sentences}".lower()
    if "focused ultrasound" in t or "arabidopsis" in t or "angiography" in t or "flip angle" in t:
        return "excluded", "string matches a different entity (acronym or homonym)"
    if re.search(r"\breview\b|advances in|perspective|guideline|insights|mechanisms of", str(r.title).lower()):
        return "excluded", "review or commentary, no primary finding about the gene"
    return "excluded", "gene named only in a list, table, abbreviation list or reference"


S[["decision", "reason"]] = S.apply(lambda r: pd.Series(decide(r)), axis=1)

# papers the table cites that came from the targeted protein-name searches, not the automated passes
missing = [k for k, v in CITED.items() if v["doi"] and v["doi"] not in set(S.doi)]
extra = pd.DataFrame([dict(symbol="", search_pass="pass 3: targeted search by protein name "
                           "(legumain, calretinin, calbindin, endophilin, and others)",
                           where="title/abstract", title=CITED[k]["title"], year=CITED[k]["year"], doi=CITED[k]["doi"],
                           sentences="", decision="kept", reason=f"cited in Table 3 ({k})") for k in missing])
S = pd.concat([S, extra], ignore_index=True, sort=False)
cols = ["symbol", "search_pass", "where", "title", "year", "doi", "sentences", "decision", "reason", "searched"]
S = S[[c for c in cols if c in S.columns]].sort_values(["decision", "symbol", "year"], ascending=[False, True, True])
S.to_csv(HERE / "supplementary_table_S1_literature_search.csv", index=False)

print(f"Supplementary Table S1: {len(S)} rows; {int((S.decision == 'kept').sum())} kept, "
      f"{int((S.decision == 'excluded').sum())} excluded")
print(f"every cited paper present: {all(v['doi'] in set(S.doi) for v in CITED.values() if v['doi'])} "
      f"({len(CITED)} cited keys)")
print(S.reason.str.replace(r"\(.*\)", "", regex=True).value_counts().to_string())
