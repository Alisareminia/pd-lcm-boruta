"""The external facts for the new Table 3, from two versioned public resources and nothing else.

Open Targets Platform: tractability buckets, drugs and clinical candidates, and the Parkinson disease
association split by evidence type. Human Protein Atlas: the single-nucleus brain atlas category and
the cell types in which the gene is enriched. Every value can be re-queried by anyone with the
Ensembl ID and the release date written into the output.
"""
import datetime, json, time, urllib.error, urllib.request
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
M = pd.read_csv(HERE / "gene_master.csv")[["gene", "symbol"]]
OT = "https://api.platform.opentargets.org/api/v4/graphql"
PD_ID = "MONDO_0005180"                               # Parkinson disease in Open Targets


def gql(q, v, tries=5):
    for k in range(tries):
        req = urllib.request.Request(OT, data=json.dumps({"query": q, "variables": v}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=90).read().decode())
            if "errors" not in d:
                return d["data"]
            print("   OT error", d["errors"][0]["message"][:80])
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            print("   OT retry", k + 1, type(exc).__name__)
        time.sleep(6 * (k + 1))
    return None


Q_TARGET = """query t($id: String!) { target(ensemblId: $id) {
  tractability { label modality value }
  drugAndClinicalCandidates { count } } }"""
Q_ASSOC = """query t($id: String!) { target(ensemblId: $id) {
  associatedDiseases(Bs: ["%s"]) { rows { disease { id } score datatypeScores { id score } } } } }""" % PD_ID
Q_META = "{ meta { dataVersion { year month iteration } } }"

meta = gql(Q_META, {}) or {}
dv = (meta.get("meta") or {}).get("dataVersion") or {}
release = f"{dv.get('year', '?')}.{str(dv.get('month', '?')).zfill(2)}" if dv else "unknown"
print("Open Targets release", release)

rows = []
for r in M.itertuples():
    t = (gql(Q_TARGET, {"id": r.gene}) or {}).get("target") or {}
    tract = [f"{x['modality']}: {x['label']}" for x in (t.get("tractability") or []) if x.get("value")]
    ndrug = ((t.get("drugAndClinicalCandidates") or {}).get("count")) or 0
    a = (gql(Q_ASSOC, {"id": r.gene}) or {}).get("target") or {}
    arows = ((a.get("associatedDiseases") or {}).get("rows")) or []
    dts = {d["id"]: d["score"] for d in arows[0]["datatypeScores"]} if arows else {}
    overall = arows[0]["score"] if arows else 0.0

    hpa = {}
    for k in range(4):
        try:
            with urllib.request.urlopen(f"https://www.proteinatlas.org/{r.gene}.json", timeout=60) as h:
                hpa = json.loads(h.read().decode())
            break
        except Exception as exc:
            print("   HPA retry", k + 1, type(exc).__name__); time.sleep(5)
    sn = hpa.get("RNA single nuclei brain specific nCPM") or {}
    top = sorted(sn.items(), key=lambda kv: -float(kv[1]))[:2]
    rows.append(dict(gene=r.gene, symbol=r.symbol, ot_release=release,
                     ot_pd_overall=overall, ot_pd_genetic=dts.get("genetic_association", 0.0),
                     ot_pd_literature=dts.get("literature", 0.0), ot_pd_animal=dts.get("animal_model", 0.0),
                     ot_pd_pathway=dts.get("affected_pathway", 0.0), ot_pd_rna=dts.get("rna_expression", 0.0),
                     ot_pd_drug=dts.get("known_drug", 0.0) or dts.get("clinical", 0.0),
                     tractable=" | ".join(tract), n_drug_candidates=int(ndrug),
                     hpa_sn_brain_specificity=hpa.get("RNA single nuclei brain specificity", ""),
                     hpa_sn_brain_distribution=hpa.get("RNA single nuclei brain distribution", ""),
                     hpa_sn_brain_top=" | ".join(f"{c} ({float(v):.0f})" for c, v in top),
                     hpa_brain_cluster=hpa.get("Brain expression cluster", "")))
    print(f"{r.symbol:9s} OT {overall:.3f} gen {dts.get('genetic_association', 0):.3f} | tract {len(tract)} "
          f"| drugs {ndrug} | HPA {rows[-1]['hpa_sn_brain_specificity']}: {rows[-1]['hpa_sn_brain_top'][:60]}")

out = pd.DataFrame(rows)
out["accessed"] = datetime.date.today().isoformat()
out.to_csv(HERE / "table3_sources.csv", index=False)
print(f"\nwrote table3_sources.csv ({len(out)} genes)")
