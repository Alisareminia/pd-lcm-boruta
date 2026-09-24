"""Writes a Word file whose citations are EndNote unformatted temporary citations.

EndNote can turn `{Kamath, 2022 #14}` into a real Cite While You Write field, so this script rewrites every
\\citep/\\citet in the manuscript into that form, drops the LaTeX bibliography (EndNote builds its own) and
hands the result to pandoc. The record numbers match the order of endnote_paper_references.ris, which is the
order EndNote assigns on import into an empty library.

Run build_ris.py first: this reads the same .bbl for the citation order, so the two always agree.
"""
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "ENDNOTE"
OUT.mkdir(exist_ok=True)

BIB = (HERE / "references.bib").read_text(encoding="utf-8", errors="replace")
BBL = (HERE / "boruta_lcm_pd_signatures.bbl").read_text(encoding="utf-8", errors="replace")
def expand(tex, depth=0):
    """Inline every \\input, so that citations inside the tables are rewritten too."""
    if depth > 4:
        return tex

    def one(m):
        f = HERE / (m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")
        return expand(f.read_text(encoding="utf-8", errors="replace"), depth + 1) if f.exists() else m.group(0)

    return re.sub(r"\\input\{([^}]+)\}", one, tex)


TEX = expand((HERE / "boruta_lcm_pd_signatures.tex").read_text(encoding="utf-8", errors="replace"))

ORDER = [k for k in re.findall(r"\\entry\{([^}]+)\}", BBL)]
RECNO = {k: i + 1 for i, k in enumerate(ORDER)}


def field(key, name):
    m = re.search(r"@\w+\{" + re.escape(key) + r",(.*?)\n\s*\n", BIB + "\n\n", re.S)
    if not m:
        return ""
    f = re.search(name + r"\s*=\s*\{+(.*?)\}+\s*,", m.group(1), re.S | re.I)
    return re.sub(r"\s+", " ", f.group(1)).strip() if f else ""


def surname(key):
    """The first author's family name, as EndNote will have it after importing the RIS."""
    a = field(key, "author")
    if not a:
        return key
    first = re.split(r"\s+and\s+", a)[0].strip()
    first = re.sub(r"\\[a-zA-Z]+\s*|[{}\\]", "", first).strip()
    # a consortium or society has no comma and several words: EndNote holds the whole name, so use it whole
    if "," not in first and len(first.split()) >= 3:
        return first
    return (first.split(",")[0] if "," in first else first.split()[-1]).strip()


def temp(keys):
    """One EndNote temporary citation covering every key of a \\citep, escaped for LaTeX."""
    parts = []
    for k in keys:
        n = RECNO.get(k)
        parts.append(f"{surname(k)}, {field(k, 'year')}" + (f" \\#{n}" if n else ""))
    return "\\{" + "; ".join(parts) + "\\}"


def replace(m):
    keys = [k.strip() for k in m.group(2).split(",") if k.strip()]
    body = temp(keys)
    if m.group(1) in ("citet", "textcite"):                 # 'Kamath et al. {Kamath, 2022 #14}' reads badly;
        return body                                        # EndNote's author-in-text form is set by the style
    return body


src = re.sub(r"\\(citep|citet|cite|textcite|citealp)\*?(?:\[[^\]]*\])*\{([^}]*)\}", replace, TEX)

# EndNote generates the reference list itself, so the LaTeX one goes
src = src.replace(r"\printbibliography[title={References}]",
                  r"\section*{References}" "\n"
                  r"\noindent\textit{EndNote will replace this heading's contents when the citations are "
                  r"formatted.}")

# Word cannot show an embedded PDF, so point the figures at the PNG copies
src = re.sub(r"(figures/[A-Za-z0-9_]+)\.pdf", r"\1.png", src)

tmp = HERE / ".endnote_docx.tex"
tmp.write_text(src, encoding="utf-8")
docx = OUT / "boruta_lcm_pd_signatures_ENDNOTE.docx"
subprocess.run(["pandoc", str(tmp), "-o", str(docx), "--resource-path", f"{HERE}:{HERE / 'figures'}"], check=True)
tmp.unlink()

n = len(re.findall(r"\\\{[^}]*\\#\d+", src))
print(f"{docx.name}: {n} temporary citations, {len(RECNO)} records referenced")
missing = sorted({k for m in re.finditer(r"\\(?:citep|citet|cite)\*?\{([^}]*)\}", TEX)
                  for k in m.group(1).split(",") if k.strip() and k.strip() not in RECNO})
print("keys with no record number:", missing or "none")
