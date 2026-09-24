import zipfile, re, xml.etree.ElementTree as ET
import pandas as pd
def read_strict_xlsx(path):
    """Every sheet of an .xlsx workbook as a DataFrame - handles 'strict' OOXML, which openpyxl cannot open."""
    zx = zipfile.ZipFile(path)
    wb = ET.fromstring(zx.read("xl/workbook.xml"))
    M = wb.tag.split("}")[0].strip("{")                                  # main namespace, strict or transitional
    rels = ET.fromstring(zx.read("xl/_rels/workbook.xml.rels"))
    target = {r.get("Id"): r.get("Target") for r in rels}
    RID = [k for k in wb.find(f"{{{M}}}sheets")[0].attrib if k.endswith("}id")][0]
    ss = []
    if "xl/sharedStrings.xml" in zx.namelist():
        for si in ET.fromstring(zx.read("xl/sharedStrings.xml")).findall(f"{{{M}}}si"):
            ss.append("".join(t.text or "" for t in si.iter(f"{{{M}}}t")))
    col = lambda ref: sum((ord(ch) - 64) * 26 ** i for i, ch in enumerate(reversed(re.match(r"[A-Z]+", ref).group())))
    out = {}
    for sh in wb.find(f"{{{M}}}sheets"):
        f = "xl/" + target[sh.get(RID)].lstrip("/").replace("xl/", "")
        rows = []
        for r in ET.fromstring(zx.read(f)).iter(f"{{{M}}}row"):
            vals = {}
            for c in r.findall(f"{{{M}}}c"):
                v = c.findtext(f"{{{M}}}v")
                if c.get("t") == "s" and v is not None:
                    v = ss[int(v)]
                elif c.get("t") == "inlineStr":
                    v = "".join(t.text or "" for t in c.iter(f"{{{M}}}t"))
                vals[col(c.get("r")) - 1] = v
            rows.append([vals.get(i) for i in range(max(vals) + 1)] if vals else [])
        out[sh.get("name")] = pd.DataFrame(rows)
    return out
