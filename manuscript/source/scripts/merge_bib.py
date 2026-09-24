"""Merges the three bibliographies into one file, keeping the curated entry when a key repeats."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SOURCES = [HERE.parent / "manuscript" / "references.bib",   # curated for paper 1, hand-checked
           HERE / "refs_extra.bib",                          # methods, resources, external cohorts
           HERE / "gene_evidence.bib",                       # one paper per panel gene
           HERE / "refs_pdlinks.bib"]                        # the mechanism papers behind the PD-link column


def entries(text):
    out = {}
    for m in re.finditer(r"@\w+\{([^,]+),", text):
        start = m.start()
        depth, i = 0, text.index("{", start)
        while i < len(text):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                break
            i += 1
        out[m.group(1).strip()] = text[start:i + 1]
    return out


merged, seen = [], {}
for src in SOURCES:
    if not src.exists():
        print("missing", src); continue
    got = entries(src.read_text())
    new = 0
    for k, v in got.items():
        if k in seen:
            continue
        seen[k] = v; merged.append(v); new += 1
    print(f"{src.name}: {len(got)} entries, {new} new")

import re as _re
text = "\n\n".join(merged) + "\n"
UNICODE = {"\u03b1": r"$\alpha$", "\u03b2": r"$\beta$", "\u03b3": r"$\gamma$",
           "\u03b4": r"$\delta$", "\u03ba": r"$\kappa$", "\u03bc": r"$\mu$",
           "\u03c9": r"$\omega$", "\u2212": "-", "\u2013": "--", "\u2014": "---",
           "\u2019": "'", "\u2018": "'", "\u201c": "``", "\u201d": "''",
           "\u00d7": r"$\times$", "\u2265": r"$\ge$", "\u2264": r"$\le$",
           "\u00b1": r"$\pm$", "\u00b0": r"$^\circ$", "\u2032": "'",
           "\u2192": r"$\rightarrow$"}
for _u, _t in UNICODE.items():                     # fetched titles carry Greek letters and smart quotes
    text = text.replace(_u, _t)
text = _re.sub(r"(?<!\\)&", r"\\&", text)          # journal titles arrive with raw ampersands
text = _re.sub(r"(?<!\\)%", r"\\%", text)
(HERE / "references.bib").write_text(text)
print(f"\nreferences.bib: {len(merged)} entries")
