"""Builds PD_LCM_ext_probe.ipynb - a short look at candidate external cohorts before the real notebook.

Downloads (on Kaggle, from the GEO FTP site) the series matrices of candidate bulk substantia nigra
cohorts and lists their supplementary files, then prints what the parser needs: sample counts, titles,
source names, characteristic fields, and the layout of any processed RNA-seq file.
"""
import pathlib
from kaggle_nb import write_nb
HERE = pathlib.Path(__file__).parent
CELLS = []
def code(s): CELLS.append(("code", s))
CELLS.append(("markdown", "# Probe: what the candidate external cohorts look like"))

code(r'''import re, io, gzip, tarfile, urllib.request, collections
from pathlib import Path
D = Path("/kaggle/working/geo"); D.mkdir(parents=True, exist_ok=True)
CANDIDATES = ["GSE20292", "GSE49036", "GSE8397", "GSE20163", "GSE20164", "GSE20333", "GSE20314", "GSE168496", "GSE114517"]
def geo_dir(acc):
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{acc[:-3]}nnn/{acc}/"
def listing(url):
    try:
        html = urllib.request.urlopen(url, timeout=120).read().decode("utf-8", "replace")
    except Exception as exc:
        return [f"(listing failed: {type(exc).__name__})"]
    return sorted(set(re.findall(r'href="([^"?/][^"]*)"', html)))
def fetch(url, dest):
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    return dest''')

code(r'''for acc in CANDIDATES:
    print("=" * 100); print(acc)
    mats = [f for f in listing(geo_dir(acc) + "matrix/") if f.endswith("_series_matrix.txt.gz")]
    sup = listing(geo_dir(acc) + "suppl/")
    print("  matrices:", mats); print("  supplementary:", sup)
    for m in mats:
        p = fetch(geo_dir(acc) + "matrix/" + m, D / m)
        lines = gzip.open(p, "rt", errors="replace").read().split("\n")
        meta = collections.OrderedDict()
        for l in lines:
            if l.startswith("!Sample_"):
                k = l.split("\t")[0]
                vals = [v.strip('"') for v in l.split("\t")[1:]]
                meta.setdefault(k, []).append(vals)
        n = len(meta.get("!Sample_geo_accession", [[]])[0])
        b0 = next((i for i, l in enumerate(lines) if "!series_matrix_table_begin" in l), None)
        b1 = next((i for i, l in enumerate(lines) if "!series_matrix_table_end" in l), None)
        print(f"  -- {m}: {n} samples, {0 if b0 is None else b1 - b0 - 2} data rows; platform "
              f"{meta.get('!Sample_platform_id', [['?']])[0][:1]}")
        for k in ("!Sample_title", "!Sample_source_name_ch1"):
            for vals in meta.get(k, []):
                print(f"     {k}: {vals[:12]}{' ...' if len(vals) > 12 else ''}")
        for vals in meta.get("!Sample_characteristics_ch1", []):
            c = collections.Counter(v.split(":")[0] for v in vals)
            ex = sorted(set(vals))[:8]
            print(f"     characteristics {dict(c)}: {ex}")''')

code(r'''# the processed RNA-seq files: what is inside
for acc, want in (("GSE168496", ".tsv.gz"), ("GSE114517", "RAW.tar")):
    sup = [f for f in listing(geo_dir(acc) + "suppl/") if f.endswith(want) or want in f]
    print("=" * 100); print(acc, sup)
    for f in sup[:2]:
        p = fetch(geo_dir(acc) + "suppl/" + f, D / f)
        print(f"  {f}: {p.stat().st_size / 1e6:.1f} MB")
        if f.endswith(".tar"):
            with tarfile.open(p) as t:
                names = t.getnames(); print(f"  {len(names)} members; first: {names[:6]}")
                m0 = t.extractfile(names[0]).read()
                txt = gzip.decompress(m0).decode("utf-8", "replace") if names[0].endswith(".gz") else m0.decode("utf-8", "replace")
                print("  first lines of", names[0]); print("\n".join(txt.split("\n")[:6]))
        else:
            txt = gzip.open(p, "rt", errors="replace").read(3000)
            print("\n".join(txt.split("\n")[:4])[:1500])''')

write_nb(HERE / "PD_LCM_ext_probe.ipynb", CELLS, "prb")
