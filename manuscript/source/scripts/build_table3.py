"""Candidate Table 3, built as its own PDF so the manuscript is not touched.

Every cell is either measured in this study, computed by a named public resource at a stated release,
or a published finding given with its citation. No grades, no labels of our own, no cut-offs we chose:
bars are scaled to the largest value among the 30 genes, rows follow the model's SHAP rank, and the
two replication questions are answered with P values.
"""
import math, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
G = pd.read_csv(HERE / "gene_master.csv")
S = pd.read_csv(HERE / "table3_sources.csv")
B = pd.read_csv(ROOT / "classifier" / "outputs_rf" / "external_multi" / "external_multi_gene_meta.csv")
B = B.rename(columns={"g_external_neuron_adj": "g_external_neuron_adj_meta"})
G = G.merge(S.drop(columns=["symbol"]), on="gene", how="left").merge(
    B[["gene", "g_external_neuron_adj_meta", "se_external_neuron_adj"]], on="gene", how="left")
N_PD, N_CT = 33, 30


def esc(s):
    s = str(s or "")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def p_fmt(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def se_hedges(g, n1=N_PD, n2=N_CT):
    """Large-sample standard error of Hedges' g (Hedges and Olkin)."""
    return math.sqrt((n1 + n2) / (n1 * n2) + g * g / (2 * (n1 + n2)))


# the adjusted effect is tested like any other effect, and the bulk effect against its own SE
G["p_adjusted"] = [2 * norm.sf(abs(g) / se_hedges(g)) for g in G.g_composition_removed]
G["p_bulk"] = [2 * norm.sf(abs(g) / s) if s and s > 0 else np.nan
               for g, s in zip(G.g_external_neuron_adj_meta, G.se_external_neuron_adj)]
G["bulk_same_sign"] = np.sign(G.g_external_neuron_adj_meta) == np.sign(G.hedges_g_discovery)


def bh(p):
    """Benjamini-Hochberg q values over the 30 genes of one test family."""
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full(p.shape, np.nan)
    v = p[ok]
    order = np.argsort(v)
    ranked = v[order] * len(v) / (np.arange(len(v)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


G["q_discovery"], G["q_bulk"], G["q_adjusted"] = bh(G.p_discovery), bh(G.p_bulk), bh(G.p_adjusted)

# bars end at the largest value observed among the 30 genes, not at a number we picked
MAX = {"effect": G.hedges_g_discovery.abs().max(), "folds": G.boruta_fold_frequency.max(),
       "bulk": G.g_external_neuron_adj_meta.abs().max(), "kept": G.retained.max()}
BARW = 13.0


def bar(frac, ghost=False):
    f = min(max(frac, 0.0), 1.0)
    a, b = BARW * f, BARW * (1 - f)
    out = rf"\textcolor{{{'mixc' if ghost else 'keptc'}}}{{\rule{{{a:.2f}mm}}{{2.2pt}}}}" if a > 0.05 else ""
    return out + (rf"\textcolor{{emptyc}}{{\rule{{{b:.2f}mm}}{{2.2pt}}}}" if b > 0.05 else "")


def ours(r):
    lines = [("effect", abs(r.hedges_g_discovery) / MAX["effect"], False,
              rf"$g$ {r.hedges_g_discovery:+.2f}, $q$ {p_fmt(r.q_discovery)}"),
             ("folds", r.boruta_fold_frequency / MAX["folds"], False,
              rf"{100 * r.boruta_fold_frequency:.0f}\% of folds"),
             ("bulk", abs(r.g_external_neuron_adj_meta) / MAX["bulk"], not r.bulk_same_sign,
              rf"$g$ {r.g_external_neuron_adj_meta:+.2f}, $q$ {p_fmt(r.q_bulk)}"),
             ("kept", r.retained / MAX["kept"], False,
              rf"$g$ {r.g_composition_removed:+.2f}, $q$ {p_fmt(r.q_adjusted)}")]
    return r"\newline".join(rf"{{\tiny\makebox[7.4mm][l]{{{n}}}}}{bar(f, gh)}{{\tiny\ {v}}}"
                            for n, f, gh, v in lines)


def short(summary, n=90):
    """First sentence of the RefSeq summary, trimmed."""
    t = re.sub(r"\[provided by.*?\]", "", str(summary or "")).strip()
    t = re.sub(r"\s+", " ", t)
    if t.lower().startswith("this gene encodes"):
        t = "encodes " + t[len("this gene encodes "):]
    first = t.split(". ")[0]
    return (first[:n].rstrip(" ,;") + r"\ldots") if len(first) > n else first


SHORT_PATH = {
    "calcium ion binding involved in regulation of presynaptic cytosolic calcium ion concentration":
        r"presynaptic Ca$^{2+}$ binding",
    "excitatory synapse": "excitatory synapse",
    "cerebellar climbing fiber to Purkinje cell synapse": "climbing-fibre synapse",
    "regulation of synapse pruning": "synapse pruning"}


ROLES = pd.read_csv(HERE / "gene_roles.csv").set_index("symbol")


def breakable(t):
    """Let long compound names wrap at a slash instead of running into the next column."""
    return esc(t).replace("/", r"/\allowbreak ")


def role(r):
    """UniProt protein name and one complete sentence of its function; no truncation."""
    row = ROLES.loc[r.symbol]
    path = ", ".join(SHORT_PATH.get(x.strip(), esc(x.strip())) for x in str(r.pathways).split(",")) \
        if isinstance(r.pathways, str) and r.pathways else ""
    return (breakable(row.protein) + rf"\newline{{\tiny {breakable(row.role)} "
            rf"\citep{{uniprot2025}} [{row.uniprot}]}}"
            + (rf"\newline{{\tiny\textbf{{{path}}}}}" if path else ""))


NONE = r"{\tiny ---}"          # nothing in the resource, as opposed to a cell we forgot to fill


def subtype(r):
    top = esc(str(r.top_subtype_marker).replace("_", " ")) if isinstance(r.top_subtype_marker, str) else ""
    if pd.isna(r.dopamine_lineage_score):
        return NONE
    return rf"{r.dopamine_lineage_score:+.1f}" + (rf"\newline{{\tiny {top}}}" if top else "")


def cell_type(r):
    spec = esc(r.hpa_sn_brain_specificity) if isinstance(r.hpa_sn_brain_specificity, str) else ""
    top = esc(r.hpa_sn_brain_top).replace(" | ", "; ") if isinstance(r.hpa_sn_brain_top, str) and r.hpa_sn_brain_top else ""
    if not spec:
        return ""
    return spec + (rf"\newline{{\tiny {top}}}" if top else "")


def open_targets(r):
    """Overall Parkinson disease association, with the genetic component beneath it where there is one."""
    ov = r.ot_pd_overall if pd.notna(r.ot_pd_overall) else 0.0
    g = r.ot_pd_genetic if pd.notna(r.ot_pd_genetic) else 0.0
    if not ov:
        return NONE
    return f"{ov:.3f}" + (rf"\newline{{\tiny genetic {g:.3f}}}" if g > 0 else "")


# Open Targets' own tractability hierarchy, strongest first; only the top tier per modality is shown
ORDER = {"SM": ["Approved Drug", "Advanced Clinical", "Phase 1 Clinical", "Structure with Ligand",
                "High-Quality Ligand", "High-Quality Pocket", "Med-Quality Pocket", "Druggable Family"],
         "AB": ["Approved Drug", "Advanced Clinical", "Phase 1 Clinical", "UniProt loc high conf",
                "GO CC high conf", "UniProt loc med conf", "UniProt SigP or TMHMM", "GO CC med conf",
                "Human Protein Atlas loc"]}
SHORT = {"Approved Drug": "approved drug", "Advanced Clinical": "advanced clinical",
         "Phase 1 Clinical": "phase 1", "Structure with Ligand": "structure with ligand",
         "High-Quality Ligand": "high-quality ligand", "High-Quality Pocket": "high-quality pocket",
         "Med-Quality Pocket": "medium-quality pocket", "Druggable Family": "druggable family",
         "UniProt loc high conf": "surface/secreted (high confidence)",
         "GO CC high conf": "surface/secreted (high confidence)",
         "UniProt loc med conf": "surface/secreted (medium confidence)",
         "UniProt SigP or TMHMM": "signal peptide or transmembrane",
         "GO CC med conf": "surface/secreted (medium confidence)",
         "Human Protein Atlas loc": "surface/secreted (HPA)"}


def druggability(r):
    have = {}
    for item in (r.tractable.split(" | ") if isinstance(r.tractable, str) and r.tractable else []):
        m, lab = item.split(": ", 1)
        have.setdefault(m, set()).add(lab)
    parts = []
    for m, name in (("SM", "small molecule"), ("AB", "antibody")):
        top = next((lab for lab in ORDER[m] if lab in have.get(m, set())), None)
        if top:
            parts.append(rf"\textit{{{name}}}: {SHORT[top]}")
    n = int(r.n_drug_candidates) if pd.notna(r.n_drug_candidates) else 0
    if n:
        parts.append(rf"\textbf{{{n} drug or clinical candidate{'s' if n > 1 else ''}}}")
    return r"{\tiny " + r"\newline ".join(parts) + "}" if parts else NONE


# published findings, stated by study type first and in the paper's own terms
PRIOR = {
    "GPNMB": [(r"Human GWAS: risk locus, the risk allele raising GPNMB; cell and mouse models: GPNMB binds "
               r"$\alpha$-synuclein and promotes its neuronal uptake", "diazortiz2022")],
    "SOX6": [(r"Human single-nucleus nigra: marks the dopamine population depleted in Parkinson's disease",
              "kamath2022"),
             (r"Mouse: legumain cleaves SOX6 in vulnerable dopamine neurons, and the fragment speeds "
              r"degeneration in A53T $\alpha$-synuclein mice", "nie2025")],
    "CALB1": [(r"Human post-mortem nigra: calbindin-rich compartments lose fewer dopamine neurons", "damier1999")],
    "ELOVL7": [(r"Human case-control study: variant associated with Parkinson's disease", "li2018")],
    "LGMN": [(r"Cell, mouse and human brain: legumain cleaves $\alpha$-synuclein at N103, and the fragment "
              r"aggregates and is toxic to dopamine neurons", "zhang2017"),
             (r"Mouse: it cleaves $\alpha$-synuclein and tau and drives their spread from gut to brain", "ahn2020")],
    "MANF": [(r"Rat 6-hydroxydopamine model: MANF restores nigrostriatal function", "voutilainen2009")],
    "DNAJB1": [(r"Cell model: DNAJB1 clears $\alpha$-synuclein through the Hsp70 system, and its phosphorylation "
                r"by EGFR suppresses aggregation", "huang2025"),
               (r"In vitro: DNAJB1 binds $\alpha$-synuclein fibrils and recruits Hsc70 to take them apart",
                "monistrol2025")],
    "BCL11A": [(r"Mouse: marks a subset of nigral dopamine neurons that is especially vulnerable to "
                r"$\alpha$-synuclein overexpression and oxidative stress", "tolve2021")],
    "PLXNC1": [(r"Mouse development: semaphorin 7A--PLXNC1 signalling separates the nigrostriatal from the "
                r"mesolimbic dopamine pathway", "chabrat2017")],
    "CACNA1A": [(r"Cell model: the Parkinson's kinase LRRK2 regulates the Ca$_\mathrm{V}$2.1 channel", "bedford2016")],
    "CALB2": [(r"Human post-mortem midbrain, 4 patients and 3 controls: calretinin-positive dopamine neurons "
               r"of the substantia nigra were selectively preserved", "mouattprigent1994"),
              (r"Rat culture: calretinin-containing dopamine neurons were spared from levodopa toxicity",
               "isaacs1997")],
    "PRKAR2A": [(r"Human Lewy body disease brain and mice: $\alpha$-synuclein sequesters DNMT1 from the "
                 r"nucleus, hypomethylating CpG islands upstream of PRKAR2A", "desplats2011")],
    "CNTN4": [(r"Patient iPSC dopamine neurons, two brothers with the same PRKN mutations: CNTN4 was among the "
               r"most down-regulated genes in the more severely affected brother", "cukier2022")],
}


FLUID = pd.read_csv(HERE / "fluid_evidence.csv").fillna("").set_index("symbol")
FLUID_TEXT = {
    "CALB2": (r"Plasma: among the proteins most reduced in Parkinson's disease compared with controls, by "
              r"nucleic acid-linked immunoassay", "vijiaratnam2026"),
    "GPNMB": (r"Plasma: raised in Parkinson's disease across 731 patient and 59 control samples, tracking "
              r"severity", "diazortiz2022"),
    "CNTN4": (r"Cerebrospinal fluid: measured across dementias, including Parkinson's disease dementia and "
              r"dementia with Lewy bodies", None),
}


def prior(r):
    parts = [rf"{text} \citep{{{key}}}" for text, key in PRIOR.get(r.symbol, [])]
    if r.symbol in FLUID_TEXT:
        text, key = FLUID_TEXT[r.symbol]
        key = key or (FLUID.loc[r.symbol, "key"] if r.symbol in FLUID.index else "")
        parts.append(rf"{text}" + (rf" \citep{{{key}}}" if key else ""))
    return r"\newline ".join(parts) if parts else NONE


G = G.sort_values("shap_rank")
rows = []
for r in G.itertuples():
    arrow = r"$\uparrow$" if r.direction == "up" else r"$\downarrow$"
    rows.append(
        rf"\textit{{{esc(r.symbol)}}} {arrow}\newline{{\tiny {esc(r.locus)}}}\newline{{\tiny SHAP rank {int(r.shap_rank)}}} & "
        rf"{{\footnotesize {role(r)}}} & {ours(r)} & {subtype(r)} & {open_targets(r)} & "
        rf"{druggability(r)} & {{\footnotesize {prior(r)}}} \\")

release = S.ot_release.iloc[0]
accessed = S.accessed.iloc[0]
searched = (pd.read_csv(HERE / "deep_pd_search.csv").searched.iloc[0]
            if (HERE / "deep_pd_search.csv").exists() else accessed)
HEAD = (r"Gene & Type and main role & This study & DA-neuron subtype & Open Targets & "
        r"Druggability & Published link to Parkinson's disease or dopamine neurons \\")
caption = r"\captionof{table}{The 30 Boruta genes, ordered by SHAP rank.}"
notes = (
    r"\footnotesize\setlength{\parskip}{2pt}\noindent\textbf{Notes to Table 3.} "
    r"\emph{Type and main role}: the UniProt protein "
    r"name and its function, summarised in one sentence \citep{uniprot2025}; bold names are pathway terms that pass "
    r"Benjamini-Hochberg correction against the measured background. \emph{This study}: four measurements, each bar "
    r"scaled to the largest value among the 30 genes. \emph{effect} is Hedges' $g$ between 33 PD and 30 "
    r"control donors; \emph{folds} is the share of training folds in which Boruta reselected the gene; "
    r"\emph{bulk} is the inverse-variance pooled $g$ in eight external bulk cohorts after dopamine-neuron "
    r"content is regressed out (grey bar: sign opposite to discovery); \emph{kept} is $g$ after each donor's "
    r"CALB1/SOX6 composition is regressed out of every gene, the bar showing the share of the original effect "
    r"that remains. Each $q$ is a Benjamini-Hochberg value across the 30 genes within its own family of "
    r"tests: the Wilcoxon rank-sum test for the discovery effect, and $z$ tests on the pooled or adjusted "
    r"effect against its standard error for the other two. "
    r"\emph{DA-neuron subtype}: CALB1 minus SOX6 lineage score and the subtype in which the gene is the "
    r"strongest marker, from single-nucleus profiling of human nigra \citep{kamath2022}. \emph{Open Targets}: the overall "
    r"association score with Parkinson disease, and beneath it the genetic-association component, which "
    r"integrates GWAS and rare-variant evidence, where it is non-zero \citep{buniello2025}. \emph{Druggability}: Open Targets tractability evidence by modality and "
    r"the number of drug or clinical candidates; for each modality only the strongest tier in Open Targets' "
    r"own hierarchy is shown. \emph{Published link}: primary studies tying the gene to Parkinson's disease "
    r"or dopamine-neuron biology, study type first, and any measurement of the gene in patients' blood or "
    r"cerebrospinal fluid. They were found with one protocol for every gene (Methods): Europe PMC "
    r"\citep{europepmc2024}, searched for the gene symbol, its protein name and every HGNC alias together "
    r"with Parkinson's-disease and dopamine-neuron terms, in titles, abstracts and open-access full text; each "
    r"hit was read and kept only when it reports a finding about the gene itself, and reviews, acronym "
    r"collisions and retracted papers were excluded. The whole search, every hit and whether it was kept, is Supplementary Table~S1. An em dash "
    r"(---) means the resource holds no entry for that gene. "
    rf"The literature search ran on {searched}; Open Targets release {release} was accessed on {accessed}; "
    r"$\uparrow$ higher and $\downarrow$ lower in Parkinson's disease; loci are GRCh38.")

tex = [r"\begin{landscape}", r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}",
       caption, r"\label{tab:genes}",
       r"\renewcommand{\arraystretch}{1.25}",
       (r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{1.6cm}"
        r">{\raggedright\arraybackslash}p{3.7cm}>{\raggedright\arraybackslash}p{4.8cm}"
        r">{\raggedright\arraybackslash}p{1.5cm}>{\raggedright\arraybackslash}p{1.3cm}"
        r">{\raggedright\arraybackslash}p{2.8cm}>{\raggedright\arraybackslash}p{5.2cm}@{}}"),
       r"\toprule", HEAD, r"\midrule", r"\endfirsthead",
       r"\multicolumn{7}{@{}l}{\textit{Table 3, continued}}\\", r"\toprule", HEAD, r"\midrule",
       r"\endhead", r"\bottomrule", r"\endfoot"] + rows + [r"\end{longtable}", notes, r"\end{landscape}"]
(HERE / "table3_genes.tex").write_text("\n".join(tex) + "\n")

print(f"bars end at: effect |g| {MAX['effect']:.2f}, folds {MAX['folds']:.2f}, bulk |g| {MAX['bulk']:.2f}, "
      f"kept {MAX['kept']:.2f}")
print(f"filled cells: genetics {(G.ot_pd_genetic.fillna(0) > 0).sum()}, druggability "
      f"{((G.tractable.fillna('') != '') | (G.n_drug_candidates.fillna(0) > 0)).sum()}, published link {len(PRIOR)}")
print(f"bulk P<0.05 with the discovery sign: {int(((G.p_bulk < 0.05) & G.bulk_same_sign).sum())}; "
      f"adjusted P<0.05: {int((G.p_adjusted < 0.05).sum())}")
