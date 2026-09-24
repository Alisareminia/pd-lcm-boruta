"""Writes the bibliography as RIS, for import into EndNote (or Zotero, Mendeley, Papers).

Two files are produced:
  endnote_paper_references.ris - only the works the manuscript cites, in the order they are first cited,
                                 so that EndNote's record numbers match the numbers in the reference list;
  endnote_full_library.ris     - every entry in references.bib, cited or not.

The citation order is read from the compiled .bbl, which is what the paper actually printed, rather than
from the .bib file's own order.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
BIB = (HERE / "references.bib").read_text(encoding="utf-8", errors="replace")
BBL = (HERE / "boruta_lcm_pd_signatures.bbl").read_text(encoding="utf-8", errors="replace")

TYPE = {"article": "JOUR", "book": "BOOK", "inproceedings": "CPAPER", "incollection": "CHAP",
        "phdthesis": "THES", "misc": "GEN", "techreport": "RPRT"}

# LaTeX that survives in the .bib, and what it means in plain text
SUBS = [(r"\$\\alpha\$", "\u03b1"), (r"\$\\beta\$", "\u03b2"), (r"\$\\gamma\$", "\u03b3"),
        (r"\$\\delta\$", "\u03b4"), (r"\$\\mu\$", "\u03bc"), (r"\\&", "&"), (r"\\%", "%"),
        (r"\\\$", "$"), (r"\\_", "_"), (r"\\#", "#"), (r"---", "\u2014"), (r"--", "\u2013")]


# LaTeX accents: \"e, \'E, \c{c}, \v{s} and friends, in braced and bare forms
ACCENT = {'"': "\u0308", "'": "\u0301", "`": "\u0300", "^": "\u0302", "~": "\u0303",
          "=": "\u0304", ".": "\u0307", "c": "\u0327", "v": "\u030c", "u": "\u0306",
          "H": "\u030b", "k": "\u0328", "r": "\u030a"}


def accents(s):
    """Turn LaTeX accent macros into the composed unicode letter."""
    import unicodedata

    def one(m):
        mark = ACCENT.get(m.group(1))
        letter = m.group(2)
        return unicodedata.normalize("NFC", letter + mark) if mark else letter

    for _ in range(3):                                      # nested braces need a few passes
        s = re.sub(r'\\([\"\'`^~=.cvuHkr])\{(\w)\}', one, s)
        s = re.sub(r'\\([\"\'`^~=.])(\w)', one, s)
    s = re.sub(r"\\(ss)\b", "\u00df", s)
    s = re.sub(r"\\[oO]\b", lambda m: "\u00f8" if m.group(0)[-1] == "o" else "\u00d8", s)
    return s


def plain(s):
    """A .bib field as a human would read it."""
    for pat, rep in SUBS:
        s = re.sub(pat, rep, s)
    s = accents(s)
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)          # any remaining control sequence
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def entries(bib):
    """Every entry of a .bib file as (key, type, {field: value})."""
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),", bib):
        start = m.end()
        depth, i = 1, m.start(0) + len(m.group(0)) - 1
        i = start
        while i < len(bib) and depth:
            if bib[i] == "{":
                depth += 1
            elif bib[i] == "}":
                depth -= 1
                if not depth:
                    break
            i += 1
        body = bib[start:i]
        fields = {}
        for f in re.finditer(r"(\w+)\s*=\s*(\{.*?\}|\"[^\"]*\"|[^,}]+?)\s*(?:,|$)", body, re.S):
            name, val = f.group(1).lower(), f.group(2).strip()
            if val.startswith("{"):
                d, j = 0, 0
                for j, ch in enumerate(val):
                    d += (ch == "{") - (ch == "}")
                    if not d:
                        break
                val = val[1:j]
            fields[name] = val.strip('"').strip()
        out[m.group(2).strip()] = (m.group(1).lower(), fields)
    return out


def authors(field):
    """BibTeX author list to RIS 'Surname, Given' lines; 'and others' is dropped.

    A name the .bib wraps in its own braces is a corporate author, which RIS marks with a trailing comma
    so that EndNote does not split it into a surname and initials.
    """
    names = []
    for a in re.split(r"\s+and\s+", field.strip()):
        a = a.strip()
        if not a or plain(a).lower() == "others":
            continue
        corporate = a.startswith("{") and a.endswith("}")
        a = plain(a)
        if corporate:
            names.append(a.rstrip(",") + ",")
        elif "," not in a:                                 # 'Tushar Kamath' -> 'Kamath, Tushar'
            bits = a.split()
            names.append(bits[-1] + ", " + " ".join(bits[:-1]) if len(bits) > 1 else a)
        else:
            names.append(a)
    return names


def ris(key, kind, f):
    L = [f"TY  - {TYPE.get(kind, 'JOUR')}"]
    L += [f"AU  - {a}" for a in authors(f.get("author", ""))]
    if f.get("title"):
        L.append(f"TI  - {plain(f['title'])}")
    for tag, name in (("JF", "journal"), ("JO", "journal"), ("VL", "volume"), ("IS", "number"),
                      ("PB", "publisher"), ("SN", "issn"), ("DO", "doi")):
        if f.get(name):
            L.append(f"{tag}  - {plain(f[name])}")
    if f.get("year"):
        L.append(f"PY  - {plain(f['year'])}")
    pages = plain(f.get("pages", ""))
    if pages:
        parts = re.split(r"\u2013|\u2014|-+", pages)
        L.append(f"SP  - {parts[0].strip()}")
        if len(parts) > 1 and parts[-1].strip():
            L.append(f"EP  - {parts[-1].strip()}")
    if f.get("doi"):
        L.append(f"UR  - https://doi.org/{plain(f['doi'])}")
    elif f.get("url"):
        L.append(f"UR  - {plain(f['url'])}")
    L.append(f"LB  - {key}")                                # the BibTeX key, as EndNote's Label
    L.append("ER  - ")
    return "\n".join(L)


E = entries(BIB)
order = re.findall(r"\\entry\{([^}]+)\}", BBL)
cited = [k for k in order if k in E]
rest = [k for k in E if k not in set(cited)]

for name, keys in (("endnote_paper_references.ris", cited),
                   ("endnote_full_library.ris", cited + sorted(rest))):
    text = "\n\n".join(ris(k, *E[k]) for k in keys) + "\n"
    (HERE / name).write_text(text, encoding="utf-8")
    print(f"{name}: {len(keys)} records")

missing_doi = [k for k in cited if not E[k][1].get("doi")]
no_author = [k for k in cited if not authors(E[k][1].get("author", ""))]
print(f"cited works: {len(cited)}; uncited also exported: {len(rest)}")
print(f"without a DOI: {len(missing_doi)}" + (f" ({', '.join(missing_doi)})" if missing_doi else ""))
print(f"without an author: {len(no_author)}" + (f" ({', '.join(no_author)})" if no_author else ""))
