"""Pushes a notebook to Kaggle as a new version of its private kernel.

usage: python3 push_figures.py [kernel-metadata-*.json]

Credentials are read from ~/.kaggle/access_token and never printed.
"""
import json, pathlib, sys, urllib.request

HERE = pathlib.Path(__file__).parent
MFILE = sys.argv[1] if len(sys.argv) > 1 else "kernel-metadata-figures.json"
META = json.loads((HERE / MFILE).read_text())
NB   = (HERE / META["code_file"]).read_text()
TOK  = (pathlib.Path.home() / ".kaggle" / "access_token").read_text().strip()

body = {
    "slug": META["id"],                 # "owner/kernel-slug"; `id` is numeric
    "newTitle": META["title"],
    "text": NB,
    "language": "python",
    "kernelType": "notebook",
    "isPrivate": True,
    "enableGpu": False,
    "enableTpu": False,
    "enableInternet": True,
    "datasetDataSources": META.get("dataset_sources", []),
    "competitionDataSources": META.get("competition_sources", []),
    "kernelDataSources": META.get("kernel_sources", []),
    "modelDataSources": META.get("model_sources", []),
    "categoryIds": [],
}
req = urllib.request.Request(
    "https://www.kaggle.com/api/v1/kernels/push",
    data=json.dumps(body).encode(),
    headers={"Authorization": f"Bearer {TOK}",
             "Content-Type": "application/json"},
    method="POST")
try:
    with urllib.request.urlopen(req) as r:
        out = json.loads(r.read())
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:400]); sys.exit(1)

for k in ("versionNumber", "url", "error"):
    if out.get(k):
        print(f"{k}: {out[k]}")
print("OK" if not out.get("error") else "FAILED")
