"""Writes the LaTeX tables from the finished results: the 30-gene table, the cohorts, the
discovery models and the external cohorts. No result is recomputed here."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "classifier" / "outputs_rf"
HERE = Path(__file__).resolve().parents[1]


def esc(s):
    s = str(s or "")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def short(summary, n=120):
    """The first sentence of the RefSeq summary, trimmed."""
    s = re.sub(r"\[provided by.*?\]", "", str(summary or "")).strip()
    s = re.sub(r"\s+", " ", s)
    if s.lower().startswith("this gene encodes"):
        s = s[len("this gene encodes "):]
        s = "encodes " + s
    first = s.split(". ")[0]
    return (first[:n].rstrip(" ,;") + "\\ldots") if len(first) > n else first


SHORT_PATH = {                                      # the full GO strings do not fit a 13-column table
    "calcium ion binding involved in regulation of presynaptic cytosolic calcium ion concentration":
        "presynaptic Ca$^{2+}$ binding",
    "excitatory synapse": "excitatory synapse",
    "cerebellar climbing fiber to Purkinje cell synapse": "climbing-fibre synapse (GO label)",
    "regulation of synapse pruning": "synapse pruning"}

G = pd.read_csv(HERE / "gene_master.csv")
G = G.sort_values(["direction", "shap_rank"], ascending=[True, True])

G = pd.read_csv(HERE / "gene_master.csv")
G = G.sort_values(["direction", "shap_rank"], ascending=[True, True])

# ------------------------------------------------------------------ the 30-gene table
# Our own evidence is four bars; everything outside this study is graded on three axes the way a
# reader would ask for it - human, mechanistic, fluid - and each grade carries its source.
BARW = 15.0
SCALE = {"effect": 1.25, "folds": 1.0, "bulk": 0.60, "kept": 1.0}
LINKS = pd.read_csv(HERE / "pd_links.csv").fillna("").set_index("symbol")
FLUID = pd.read_csv(HERE / "fluid_evidence.csv").fillna("").set_index("symbol")


def bar(frac, ghost=False):
    f = min(max(frac, 0.0), 1.0)
    filled, rest = BARW * f, BARW * (1 - f)
    out = rf"\textcolor{{{'mixc' if ghost else 'keptc'}}}{{\rule{{{filled:.2f}mm}}{{2.2pt}}}}" if filled > 0.05 else ""
    if rest > 0.05:
        out += rf"\textcolor{{emptyc}}{{\rule{{{rest:.2f}mm}}{{2.2pt}}}}"
    return out


def profile(r):
    rows = [("effect", abs(r.hedges_g_discovery) / SCALE["effect"], f"$g$ {r.hedges_g_discovery:+.2f}", False),
            ("folds", r.boruta_fold_frequency, rf"{100 * r.boruta_fold_frequency:.0f}\% reselected", False),
            ("bulk", abs(r.g_external_neuron_adjusted) / SCALE["bulk"],
             f"{r.g_external_neuron_adjusted:+.2f}" + ("" if r.replicates_in_bulk else " reversed"),
             not r.replicates_in_bulk),
            ("kept", min(r.retained, 1.0), rf"{100 * r.retained:.0f}\% after the mix", False)]
    return r"\newline".join(rf"{{\tiny\makebox[6.6mm][l]{{{n}}}}}{bar(f, ghost=g)}{{\tiny\ {v}}}"
                           for n, f, v, g in rows)


def cite(key):
    return rf" \citep{{{key}}}" if isinstance(key, str) and key.strip() else ""


def ot_tag(r):
    """The Open Targets association, under every gene, so the column compares straight down."""
    ot = f"OT {r.pd_association_open_targets:.3f}" if r.pd_association_open_targets else "OT 0"
    gen = f", genetic {r.pd_genetic_association:.3f}" if r.pd_genetic_association else ""
    return rf"\newline{{\tiny {ot}{gen}; {int(r.n_pd_papers)} PD papers}}"


def ot_tag(r):
    """The Open Targets association, under every gene, so the column compares straight down."""
    ot = f"OT {r.pd_association_open_targets:.3f}" if r.pd_association_open_targets else "OT 0"
    gen = f", genetic {r.pd_genetic_association:.3f}" if r.pd_genetic_association else ""
    return rf"\newline{{\tiny {ot}{gen}; {int(r.n_pd_papers)} PD papers}}"


def human_cell(r):
    """Human genetic and post-mortem evidence, graded, with the Open Targets score always shown."""
    if r.symbol in LINKS.index and LINKS.loc[r.symbol, "human_text"]:
        t = LINKS.loc[r.symbol]
        return f"{t.human_text}{cite(t.human_key)}{ot_tag(r)}"
    ot, gen = r.pd_association_open_targets, r.pd_genetic_association
    if gen >= 0.05:
        body = f"Moderate: a genetic association in Open Targets{cite('buniello2025')}"
    elif ot >= 0.05:
        body = f"Limited: an Open Targets association with no genetic component{cite('buniello2025')}"
    elif r.evidence_kind == "pd":
        body = f"Limited: named in a Parkinson's study of the gene{cite(r.key)}"
    else:
        body = "None found: the gene is mentioned in Parkinson's papers, never studied in one"
    return f"{body}{ot_tag(r)}"


def mech_cell(r):
    """Molecular or model-system evidence, graded."""
    if r.symbol in LINKS.index and LINKS.loc[r.symbol, "mech_text"]:
        t = LINKS.loc[r.symbol]
        return f"{t.mech_text}{cite(t.mech_key)}"
    where = {"pd": "in Parkinson's disease, though not as a disease mechanism",
             "neuro": "in the nervous system, outside Parkinson's disease",
             "other": "outside the nervous system", "none": "nowhere under its own name"}[r.evidence_kind]
    return f"None established: the gene's own literature sits {where}{cite(r.key)}"


def fluid_cell(r):
    if r.symbol in FLUID.index:
        t = FLUID.loc[r.symbol]
        return f"{t.text}{cite(t.key)}"
    return "None found"


G = pd.read_csv(HERE / "gene_master.csv")
SHORT_PATH = {
    "calcium ion binding involved in regulation of presynaptic cytosolic calcium ion concentration":
        "presynaptic Ca$^{2+}$ binding",
    "excitatory synapse": "excitatory synapse",
    "cerebellar climbing fiber to Purkinje cell synapse": "climbing-fibre synapse",
    "regulation of synapse pruning": "synapse pruning"}
G["profile_score"] = (G.hedges_g_discovery.abs() / SCALE["effect"]).clip(0, 1) + G.boruta_fold_frequency \
    + (G.g_external_neuron_adjusted.abs() / SCALE["bulk"]).clip(0, 1) * G.replicates_in_bulk.astype(float) \
    + G.retained.clip(0, 1)
G = G.sort_values("profile_score", ascending=False)

rows = []
for r in G.itertuples():
    arrow = r"$\uparrow$" if r.direction == "up" else r"$\downarrow$"
    path = ", ".join(SHORT_PATH.get(x.strip(), esc(x.strip())) for x in str(r.pathways).split(",")) \
        if isinstance(r.pathways, str) and r.pathways else ""
    sub = esc(str(r.top_subtype_marker).replace("_", " ")) if isinstance(r.top_subtype_marker, str) else "--"
    lineage = ("CALB1-like" if r.dopamine_lineage_score > 0.5 else
               "SOX6-like" if r.dopamine_lineage_score < -0.5 else "neither")
    rows.append(
        rf"\textit{{{esc(r.symbol)}}} {arrow}\newline{{\tiny {esc(r.locus)}}}\newline"
        rf"{{\tiny $p$ {r.p_discovery:.3f}{'; DEG' if r.deg else ''}}} & "
        rf"{esc(r.full_name)}\newline{{\tiny {esc(short(r.summary, 88))}}}"
        + (rf"\newline{{\tiny\textbf{{{path}}}}}" if path else "") + " & "
        rf"{profile(r)}\newline{{\tiny {lineage}, {sub}}} & "
        rf"{{\footnotesize {human_cell(r)}}} & {{\footnotesize {mech_cell(r)}}} & "
        rf"{{\footnotesize {fluid_cell(r)}}} \\")

HEAD = (r"Gene & Type and main role & Our evidence & Human PD evidence & Mechanistic evidence & "
        r"Blood / CSF \\")
caption = (
    r"\caption{The 30 Boruta genes. \emph{Our evidence} is four bars, each on a fixed scale so a column can "
    r"be read down the page: \emph{effect}, the standardised PD minus control difference across the 63 "
    r"discovery donors ($|g|$, full bar at 1.25); \emph{folds}, the share of training folds in which Boruta "
    r"reselected the gene; \emph{bulk}, the effect in the eight external cohorts once dopamine-neuron "
    r"content is removed (full bar at $|g| = 0.60$, drawn grey and marked \emph{reversed} where the "
    r"direction contradicts discovery); and \emph{kept}, the share of the discovery effect still present "
    r"when each donor's CALB1/SOX6 mix is regressed out of every gene. Beneath them is the gene's lineage "
    r"call and the subtype in which it is the strongest marker \citep{kamath2022}. The three evidence "
    r"columns are graded from the sources they cite: \emph{human} from post-mortem or genetic studies of "
    r"the gene itself, with the Open Targets association with Parkinson disease, its genetic component where "
    r"non-zero, and the Europe PMC paper count printed under every gene \citep{buniello2025}; \emph{mechanistic} from molecular or "
    r"model-system work; \emph{blood / CSF} from fluid measurements in patients. An axis with nothing behind "
    r"it says so, and most genes have nothing on most axes. Rows are ordered by the sum of the "
    r"bars. Bold pathway names pass Benjamini-Hochberg "
    r"correction against the measured background (Fig.~\ref{fig:pathways}); $\uparrow$ higher and "
    r"$\downarrow$ lower in Parkinson's disease; $p$ is the Wilcoxon rank-sum test and DEG marks $p<0.01$. "
    r"Literature searches used Europe PMC \citep{europepmc2024}; function is RefSeq \citep{oleary2016} and "
    r"loci are GRCh38.}")

tex = [r"\begin{landscape}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
       r"\renewcommand{\arraystretch}{1.22}",
       (r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{1.7cm}"
        r">{\raggedright\arraybackslash}p{3.1cm}>{\raggedright\arraybackslash}p{4.7cm}"
        r">{\raggedright\arraybackslash}p{4.3cm}>{\raggedright\arraybackslash}p{4.0cm}"
        r">{\raggedright\arraybackslash}p{3.6cm}@{}}"),
       caption, r"\label{tab:genes}\\", r"\toprule", HEAD, r"\midrule", r"\endfirsthead",
       r"\multicolumn{6}{@{}l}{\textit{Table \ref{tab:genes}, continued}}\\", r"\toprule", HEAD,
       r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"] + rows + [r"\end{longtable}", r"\end{landscape}"]
(HERE / "table_genes.tex").write_text("\n".join(tex) + "\n")

# ------------------------------------------------------------------ discovery cohorts
COH = pd.read_csv(ROOT / "merged" / "kaggle_outputs" / "01_cohort_by_dataset.csv")
PLAT = {"GSE182622": ("RNA-seq (HiSeq 2500)", r"T\v{i}klov\'a et al.\ \citep{tiklova2021}"),
        "GSE20141": ("Affymetrix U133 Plus 2.0", r"Zheng et al.\ \citep{zheng2010}"),
        "GSE24378": ("Affymetrix U133 X3P", r"Zheng et al.\ \citep{zheng2010}"),
        "GSE169755": ("RNA-seq (HiSeq 4000)", r"Zaccaria et al.\ \citep{zaccaria2022}")}
lines = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
         r"\caption{The four laser-capture discovery cohorts, one profile per person.}",
         r"\label{tab:cohort}", r"\begin{tabular}{@{}llrrrl@{}}", r"\toprule",
         r"Dataset & Platform & Control & PD & Units merged & Source \\", r"\midrule"]
for d in ("GSE182622", "GSE20141", "GSE24378", "GSE169755"):
    r = COH[COH.dataset == d].iloc[0]
    lines.append(f"{d} & {PLAT[d][0]} & {int(r.control)} & {int(r.PD)} & {int(r.units_merged)} & {PLAT[d][1]} \\\\")
lines += [r"\midrule",
          f"Total & & {int(COH.control.sum())} & {int(COH.PD.sum())} & {int(COH.units_merged.sum())} & \\\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(HERE / "table_cohorts.tex").write_text("\n".join(lines) + "\n")

# ------------------------------------------------------------------ external cohorts and their AUCs
EXT = json.load(open(OUT / "external_unified" / "external_unified_summary.json"))
per = None
for f in ("external_unified_per_cohort.csv", "per_cohort_metrics.csv"):
    if (OUT / "external_unified" / f).exists():
        per = pd.read_csv(OUT / "external_unified" / f)
if per is None and (OUT / "external_multi" / "per_cohort_metrics.csv").exists():
    per = pd.read_csv(OUT / "external_multi" / "per_cohort_metrics.csv")
A = EXT["auc"]
lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
         (r"\caption{Pooled performance in the eight external bulk-tissue cohorts "
          r"(" + f"{EXT['people']} people, {EXT['control']} control, {EXT['PD']} PD" + r"). "
          r"\emph{Strictly independent} models were retrained without any discovery study from the same "
          r"brain bank. \emph{Neuron score removed} residualises each donor's score on eight "
          r"dopamine-neuron markers. Confidence intervals are 4,000 donor bootstraps.}"),
         r"\label{tab:external}", r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r"Model & AUC [95\% CI] & Neuron score removed & Label shuffles \\", r"\midrule"]
for key, lab, p in (("frozen", "Core classifier, fixed", EXT["perm_p_within_cohorts"]["core"]),
                    ("strict", "Core classifier, strictly independent", None),
                    ("panel", "30-gene panel forest", EXT["perm_p_within_cohorts"]["panel"]),
                    ("panel_strict", "30-gene panel, strictly independent", None),
                    ("neuron", "Dopamine-neuron markers alone", None)):
    a, adj = A[key], A.get(f"{key}|neuron_removed")
    ps = "--" if p is None else ("$<0.001$" if p < 0.001 else f"{p:.3f}")
    adj_s = "--" if adj is None else f"{adj['auc']:.2f} [{adj['ci_lo']:.2f}--{adj['ci_hi']:.2f}]"
    lines.append(f"{lab} & {a['auc']:.2f} [{a['ci_lo']:.2f}--{a['ci_hi']:.2f}] & {adj_s} & {ps} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(HERE / "table_external.tex").write_text("\n".join(lines) + "\n")

print("wrote table_genes.tex, table_cohorts.tex, table_external.tex")
print(G[["symbol", "direction", "shap_rank", "evidence_kind"]].head(8).to_string())
