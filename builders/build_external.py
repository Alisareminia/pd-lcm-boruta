"""Builds PD_LCM_rf_external.ipynb - secondary notebook: external validation on bulk substantia nigra (GSE7621).

Reads the core notebook's frozen model (core_model.joblib: PCA + Random Forest fitted on all 63
laser-capture people) and data, the Boruta panel, and the GSE7621 series matrix from the Kaggle
dataset externalvalidation2. Nothing is fitted on GSE7621: its genes are z-scored within the cohort
and ranked within each person, exactly the label-free steps used in discovery, and scored by the
frozen models.

GSE7621 is bulk tissue and PD nigra has lost most of its dopamine neurons, so a model could separate
the groups by tracking neuron content alone. Every result is therefore shown next to an 8-gene
dopamine-neuron marker score, and again after removing that component.

Figure (print size, style of Figures 1-5):
  a - ROC of the frozen models in GSE7621, with the neuron-marker score as the benchmark
  b - the core model's score against neuron content: how much is left once neuron loss is removed
  c - every Boruta gene's PD-versus-control effect in the laser-capture cohort and in GSE7621,
      before and after removing neuron content
"""
import pathlib
from kaggle_nb import write_nb
HERE = pathlib.Path(__file__).parent
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md("""# External validation: bulk substantia nigra, GSE7621

**Nothing is fitted on the external cohort.** The core classifier is loaded exactly as saved by
**`pd-lcm-rf-core`** (PCA + Random Forest on all 63 laser-capture people); the Boruta-panel forest is
fitted on the same 63 people. GSE7621 (Affymetrix HG-U133 Plus 2.0, 9 control, 16 PD) goes through
the same label-free steps as discovery: genes z-scored within the cohort, then ranked within each person.

**The confound.** GSE7621 is bulk tissue, and PD nigra has lost most of its dopamine neurons, so a
model could separate the groups by tracking neuron content alone. Each result is therefore set next to an
eight-gene dopamine-neuron marker score, and repeated after that component is removed.""")

code(r'''import os, io, glob, json, time, urllib.request, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
try:
    get_ipython(); IN_NB = True
except NameError:
    IN_NB = False; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import joblib
from scipy.stats import rankdata, spearmanr, binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
warnings.filterwarnings("ignore")

ON_KAGGLE = Path("/kaggle/input").exists()
OUT = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("SMOKE_OUT", "smoke_ext"))
FIG_DIR = OUT / "figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)
SOURCES = {"core": ("rf-core", "outputs_rf/core", "smoke_core"), "panel": ("boruta-panel", "outputs_rf/panel"),
           "external": ("externalvalidation2", "lcm/harmonised", "external")}
def find_any(pattern, source):
    """The file from one named input - several inputs carry files with the same name."""
    roots = ["/kaggle/input"] if ON_KAGGLE else os.environ.get("FIG_EXTRA", ".").split(":")
    hits = [h for r in roots for h in glob.glob(f"{r}/**/{pattern}", recursive=True)]
    hits = sorted((h for h in hits if any(k in h for k in SOURCES[source])), key=len)
    if not hits:
        raise FileNotFoundError(f"{pattern} from {source}")
    return hits[0]

INK, MUTED, RULE = "#1B1D20", "#5E656D", "#C9CED4"
PD_C, CT_C, CORE, PANEL_C, NEURO_C = "#7A2533", "#6F829A", "#23324A", "#6E7F96", "#A3AAB2"
SETS = {"both": ("#8E1B2E", "#5E0F1C", "Boruta & DEG"), "boruta": ("#2E5A87", "#1B3A5C", "Boruta only")}
EFFECT = LinearSegmentedColormap.from_list("effect", ["#1E3350", "#4C6583", "#9AAABB", "#EEECE7", "#C3A09E", "#8C4B52", "#551C28"])
FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "Nimbus Sans", "FreeSans", "DejaVu Sans"]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT_STACK, "font.size": 6.5,
                     "axes.linewidth": 0.5, "axes.edgecolor": "#30343A", "axes.labelcolor": INK, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "xtick.labelsize": 6.0, "ytick.labelsize": 6.0,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
                     "xtick.major.pad": 1.8, "ytick.major.pad": 1.8, "xtick.color": "#30343A", "ytick.color": "#30343A",
                     "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150,
                     "savefig.facecolor": "white", "figure.facecolor": "white"})
print("Kaggle" if ON_KAGGLE else "LOCAL SMOKE RUN")''')

md("## 1. Discovery: the frozen core model and the Boruta panel")

code(r'''cz = np.load(find_any("core_data.npz", "core"), allow_pickle=True)
X, XR, y = cz["X"], cz["XR"], cz["y"].astype(int)
GENES = [str(g) for g in cz["genes"]]
CORE_MODEL = joblib.load(find_any("core_model.joblib", "core"))
SYM = pd.read_csv(find_any("15_gene_symbol_map.csv", "core")).set_index("gene")["symbol"].to_dict()
sym = lambda g: SYM[g] if isinstance(SYM.get(g), str) else g
PANEL = pd.read_csv(find_any("04_boruta_selected_genes.csv", "panel")).gene.tolist()
DE = pd.read_csv(find_any("03_de_results_full.csv", "panel")).set_index("gene")
PI = [GENES.index(g) for g in PANEL]
panel_rfs = [RandomForestClassifier(n_estimators=1000, max_features="sqrt", class_weight="balanced", random_state=42 + k,
                                    n_jobs=-1).fit(X[:, PI], y) for k in range(5)]
print(f"discovery: {len(y)} people x {len(GENES):,} genes; core model = PCA({CORE_MODEL['pca'].n_components_}) + "
      f"Random Forest ({CORE_MODEL['rf'].n_estimators} trees), frozen; Boruta panel {len(PANEL)} genes")''')

md("## 2. GSE7621, through the same label-free steps")

code(r'''if ON_KAGGLE:
    gse = find_any("GSE7621_series_matrix.txt", "external")
    lines = open(gse, encoding="utf-8", errors="replace").read().split("\n")
    titles = [t.strip('"') for t in next(l for l in lines if l.startswith("!Sample_title")).split("\t")[1:]]
    y7 = np.array([0 if "normal" in t.lower() else 1 for t in titles])
    b0 = next(i for i, l in enumerate(lines) if "!series_matrix_table_begin" in l)
    b1 = next(i for i, l in enumerate(lines) if "!series_matrix_table_end" in l)
    M = pd.read_csv(io.StringIO("\n".join(lines[b0 + 1:b1])), sep="\t", index_col=0).apply(pd.to_numeric, errors="coerce")
    LOGE = np.log2(M.clip(lower=1))

    def gconvert(ids, target):
        out = {}
        for s in range(0, len(ids), 3000):
            body = {"organism": "hsapiens", "target": target, "query": list(ids[s:s + 3000])}
            for k in range(5):
                try:
                    req = urllib.request.Request("https://biit.cs.ut.ee/gprofiler/api/convert/convert/",
                                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=300) as fh:
                        for rec in json.loads(fh.read())["result"]:
                            c = rec.get("converted")
                            if c and c not in ("None", "N/A"):
                                out.setdefault(rec["incoming"], set()).add(c)
                    break
                except Exception as exc:
                    print("  g:Convert retry", k + 1, type(exc).__name__); time.sleep(8)
        return out
    pm = LOGE.mean(1)
    def best_probes(ids):
        """Each gene's highest-mean probe on the array."""
        conv = gconvert(ids, "AFFY_HG_U133_PLUS_2")
        return {g: max((p for p in ps if p in LOGE.index), key=lambda p: pm[p]) for g, ps in conv.items()
                if any(p in LOGE.index for p in ps)}
    BP = best_probes(GENES)
    GL = pd.DataFrame({g: LOGE.loc[p].to_numpy(float) for g, p in BP.items()}).T          # genes x samples
    MARK_SYMS = ["TH", "SLC6A3", "SLC18A2", "DDC", "KCNJ6", "ALDH1A1", "NR4A2", "EN1"]
    MP = best_probes(MARK_SYMS)
    ML = pd.DataFrame({s: LOGE.loc[p].to_numpy(float) for s, p in MP.items()}).T
else:                                                                  # local smoke run: the same matrix, prepared earlier
    G7 = pd.read_csv(find_any("gse7621_gene_level.csv", "external"), index_col=0)
    y7 = pd.read_csv(find_any("gse7621_meta.csv", "external")).y.to_numpy(int)
    GL = G7.reindex([g for g in GENES if g in G7.index])
    S2E = {v: k for k, v in SYM.items() if isinstance(v, str)}
    MARK_SYMS = ["TH", "SLC6A3", "SLC18A2", "DDC", "KCNJ6", "ALDH1A1", "NR4A2", "EN1"]
    ML = G7.loc[[S2E[s] for s in MARK_SYMS if S2E.get(s) in G7.index]]

zrow = lambda D: ((D.T - D.T.mean()) / D.T.std(ddof=1).clip(lower=0.05)).T               # z-score each gene within GSE7621
Z7 = zrow(GL).reindex(GENES).fillna(0.0).T.to_numpy()                                   # people x genes, missing genes -> 0
XR7 = np.apply_along_axis(rankdata, 1, Z7) / Z7.shape[1]
NEURO = zrow(ML).mean(0).to_numpy()                                                      # dopamine-neuron content
print(f"GSE7621: {len(y7)} people ({int((y7 == 0).sum())} control, {int(y7.sum())} PD); {GL.shape[0]:,} of "
      f"{len(GENES):,} discovery genes on the array; {ML.shape[0]} of 8 neuron markers")''')

md("## 3. Transfer, and how much of it is neuron loss")

code(r'''S_CORE = CORE_MODEL["rf"].predict_proba(CORE_MODEL["pca"].transform(XR7))[:, 1]
S_PANEL = np.mean([m.predict_proba(Z7[:, PI])[:, 1] for m in panel_rfs], axis=0)
S_NEURO = -NEURO                                                               # fewer neurons -> more PD-like
rng = np.random.default_rng(42)
BOOT = [i for i in (rng.integers(0, len(y7), len(y7)) for _ in range(4000)) if len(set(y7[i])) == 2]
def describe(s):
    a = roc_auc_score(y7, s)
    null = np.array([roc_auc_score(rng.permutation(y7), s) for _ in range(10000)])
    bs = [roc_auc_score(y7[i], s[i]) for i in BOOT]
    resid = s - np.polyval(np.polyfit(NEURO, s, 1), NEURO)
    loo = lambda F: roc_auc_score(y7, cross_val_predict(LogisticRegression(), F, y7, cv=LeaveOneOut(), method="predict_proba")[:, 1])
    return {"auc": a, "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5), "perm_p": (np.sum(null >= a) + 1) / 10001,
            "rho_neuron": spearmanr(s, NEURO)[0], "auc_after_neuron_removed": roc_auc_score(y7, resid),
            "loo_neuron_only": loo(NEURO[:, None]), "loo_neuron_plus_score": loo(np.c_[NEURO, s])}
TR = pd.DataFrame([{"model": n, **describe(s)} for n, s in
                   (("core classifier (frozen)", S_CORE), ("Boruta-panel forest", S_PANEL))])
NEURO_AUC = roc_auc_score(y7, S_NEURO)
TR.to_csv(OUT / "external_transfer.csv", index=False)
pd.DataFrame({"y": y7, "core_score": S_CORE, "panel_score": S_PANEL, "neuron_score": NEURO}).to_csv(OUT / "external_scores.csv", index=False)
pd.set_option("display.width", 200)
print(TR.round(3).to_string(index=False)); print(f"neuron-marker score alone: AUC {NEURO_AUC:.3f}")''')

md("## 4. Gene by gene: do the Boruta genes change the same way in GSE7621?")

code(r'''def hedges(v):
    a, b = v[y7 == 1], v[y7 == 0]; n1, n0 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n0 - 1) * b.var(ddof=1)) / (n1 + n0 - 2))
    return (1 - 3 / (4 * (n1 + n0) - 9)) * (a.mean() - b.mean()) / sp
nc = NEURO - NEURO.mean()
Z7_ADJ = Z7 - np.outer(nc, (Z7 * nc[:, None]).sum(0) / (nc ** 2).sum())        # each gene with neuron content regressed out
onarray = np.array([g in GL.index for g in GENES])
G_RAW = np.array([hedges(Z7[:, i]) if onarray[i] else np.nan for i in range(len(GENES))])
G_ADJ = np.array([hedges(Z7_ADJ[:, i]) if onarray[i] else np.nan for i in range(len(GENES))])
G_LCM = DE.loc[GENES, "hedges_g_meta"].to_numpy()
EFF = pd.DataFrame({"gene": GENES, "symbol": [sym(g) for g in GENES], "g_lcm": G_LCM, "g_gse7621": G_RAW,
                    "g_gse7621_neuron_adjusted": G_ADJ, "boruta": [g in PANEL for g in GENES],
                    "deg": DE.loc[GENES, "pvalue"].to_numpy() < 0.01})
EFF.to_csv(OUT / "external_gene_effects.csv", index=False)
def agreement(col):
    b = EFF[EFF.boruta & EFF[col].notna()]; k = int((np.sign(b[col]) == np.sign(b.g_lcm)).sum())
    a = EFF[EFF[col].notna()]; bg = float((np.sign(a[col]) == np.sign(a.g_lcm)).mean())
    return {"same_direction": k, "n": len(b), "binom_p": binomtest(k, len(b), 0.5, alternative="greater").pvalue,
            "background_share": bg, "spearman": spearmanr(b.g_lcm, b[col])[0]}
AG = {"raw": agreement("g_gse7621"), "neuron_adjusted": agreement("g_gse7621_neuron_adjusted")}
for k, v in AG.items():
    print(f"{k}: {v['same_direction']}/{v['n']} Boruta genes same direction as the laser-capture cohort "
          f"(binomial P = {v['binom_p']:.4f}); all genes {100 * v['background_share']:.0f}%; Spearman {v['spearman']:.2f}")
json.dump({"n_people": int(len(y7)), "n_control": int((y7 == 0).sum()), "n_pd": int(y7.sum()),
           "genes_on_array": int(onarray.sum()), "neuron_marker_auc": NEURO_AUC,
           "transfer": TR.set_index("model").to_dict("index"), "agreement": AG},
          open(OUT / "external_summary.json", "w"), indent=1, default=float)''')

md("## 5. The figure")

code(r'''FW, FH = 183.0, 120.0
fig = plt.figure(figsize=(FW / 25.4, FH / 25.4))
M = fig.add_axes([0, 0, 1, 1]); M.set_xlim(0, FW); M.set_ylim(0, FH); M.axis("off")
def fax(x, y_, w, h): return fig.add_axes([x / FW, y_ / FH, w / FW, h / FH])
def letter(x, y_, L, title):
    M.text(x, y_, L, ha="left", va="baseline", fontsize=9, fontweight="bold", color=INK)
    M.text(x + 4.2, y_, title, ha="left", va="baseline", fontsize=7.2, color=INK)
def rule(x0, x1, y_, lw=0.4, color=RULE): M.plot([x0, x1], [y_, y_], color=color, lw=lw, solid_capstyle="butt")
tc, tp = TR.iloc[0], TR.iloc[1]

# ---------------- a: frozen models in GSE7621 ----------------
letter(2.0, 116.0, "a", "Frozen models on an independent bulk cohort")
axA = fax(12.0, 63.0, 46.0, 46.0)
axA.plot([0, 1], [0, 1], color="#B7BDC4", lw=0.5, ls=(0, (2, 2)))
for s, col, ls, lw in ((S_NEURO, NEURO_C, (0, (1.2, 1.2)), 1.0), (S_PANEL, PANEL_C, (0, (3, 1.6)), 0.9), (S_CORE, CORE, "-", 1.3)):
    f, t, _ = roc_curve(y7, s); axA.step(f, t, where="post", color=col, ls=ls, lw=lw)
axA.set_xlim(-0.01, 1.01); axA.set_ylim(-0.01, 1.01); axA.set_aspect("equal")
axA.set_xticks([0, 0.5, 1]); axA.set_yticks([0, 0.5, 1]); axA.set_xticklabels(["0", "0.5", "1"]); axA.set_yticklabels(["0", "0.5", "1"])
axA.set_xlabel("False-positive rate", fontsize=6.5, labelpad=2); axA.set_ylabel("True-positive rate", fontsize=6.5, labelpad=2)
KEY = [("core classifier", tc.auc, CORE, "-", 1.3, True), ("Boruta-panel forest", tp.auc, PANEL_C, (0, (3, 1.6)), 0.9, False),
       ("neuron markers alone", NEURO_AUC, NEURO_C, (0, (1.2, 1.2)), 1.0, False)]
for i, (lab, a, col, ls, lw, b) in enumerate(KEY):
    yy = 0.30 - i * 0.095
    axA.plot([0.22, 0.31], [yy, yy], color=col, ls=ls, lw=lw, transform=axA.transAxes)
    axA.text(0.34, yy, lab, transform=axA.transAxes, ha="left", va="center", fontsize=5.9, color=INK, fontweight="bold" if b else "normal")
    axA.text(1.0, yy, f"{a:.2f}", transform=axA.transAxes, ha="right", va="center", fontsize=5.9, color=INK,
             fontweight="bold" if b else "normal")
M.text(12.0, 52.0, f"GSE7621: {int((y7 == 0).sum())} control, {int(y7.sum())} PD; trained only on the 63 laser-capture people",
       ha="left", va="center", fontsize=5.7, color=MUTED)
M.text(12.0, 49.0, f"core classifier AUC {tc.auc:.2f} (95% CI {tc.ci_lo:.2f}–{tc.ci_hi:.2f}), label shuffles P = {tc.perm_p:.4f}",
       ha="left", va="center", fontsize=5.7, color=MUTED)

# ---------------- b: beyond neuron loss ----------------
letter(72.0, 116.0, "b", "More than neuron loss")
axB = fax(82.0, 63.0, 46.0, 46.0)
for lab, col in ((0, CT_C), (1, PD_C)):
    m = y7 == lab
    axB.scatter(NEURO[m], S_CORE[m], s=13, color=col, edgecolor="white", linewidth=0.4, zorder=3)
xx = np.linspace(NEURO.min(), NEURO.max(), 50)
axB.plot(xx, np.polyval(np.polyfit(NEURO, S_CORE, 1), xx), color="#8C9299", lw=0.7, ls=(0, (3, 1.6)), zorder=2)
axB.set_xlabel("dopamine-neuron content (8 marker genes, z)", fontsize=6.3, labelpad=2)
axB.set_ylabel("core classifier: probability of PD", fontsize=6.3, labelpad=2)
axB.text(0.98, 0.97, f"Spearman ρ = {tc.rho_neuron:.2f}", transform=axB.transAxes, ha="right", va="top", fontsize=5.8, color=MUTED)
M.scatter([84.0], [111.2], s=13, color=CT_C, edgecolor="white", linewidth=0.4)
M.text(85.6, 111.2, "control", ha="left", va="center", fontsize=5.8)
M.scatter([97.0], [111.2], s=13, color=PD_C, edgecolor="white", linewidth=0.4)
M.text(98.6, 111.2, "PD", ha="left", va="center", fontsize=5.8)
# the numbers that answer the question, as a small ruled table
TX0, TX1, TY = 140.0, 181.0, 106.0
rule(TX0, TX1, TY + 2.2, lw=0.6, color="#3A3F45")
rows_b = [("neuron markers alone", f"{NEURO_AUC:.2f}"), ("core classifier", f"{tc.auc:.2f}"),
          ("core, neuron content removed", f"{tc.auc_after_neuron_removed:.2f}"),
          ("panel forest, neuron content removed", f"{tp.auc_after_neuron_removed:.2f}"),
          ("neurons alone, leave-one-out", f"{tc.loo_neuron_only:.2f}"),
          ("neurons + core score, leave-one-out", f"{tc.loo_neuron_plus_score:.2f}")]
M.text(TX1, TY, "AUC", ha="right", va="center", fontsize=5.9, color=MUTED)
rule(TX0, TX1, TY - 1.8, lw=0.35, color="#8C9299")
for i, (lab, v) in enumerate(rows_b):
    yy = TY - 4.5 - i * 4.0
    bold = "removed" in lab and "core" in lab
    M.text(TX0, yy, lab, ha="left", va="center", fontsize=6.0, color=INK, fontweight="bold" if bold else "normal")
    M.text(TX1, yy, v, ha="right", va="center", fontsize=6.0, color=INK, fontweight="bold" if bold else "normal")
rule(TX0, TX1, TY - 4.5 - (len(rows_b) - 1) * 4.0 - 2.4, lw=0.6, color="#3A3F45")

# ---------------- c: gene by gene ----------------
rule(2.0, 181.0, 44.5)
letter(2.0, 39.8, "c", "Every Boruta gene: effect in the laser-capture cohort and in GSE7621")
P = EFF[EFF.boruta].copy()
P["grp"] = np.where(P.deg, 0, 1)
P = P.sort_values(["grp", "g_lcm"], ascending=[True, False]).reset_index(drop=True)
nG = len(P)
X0, X1 = 42.0, 164.0
cw = (X1 - X0) / nG
ROWS = [("laser-capture, pooled", "g_lcm"), ("GSE7621", "g_gse7621"), ("GSE7621, neuron content removed", "g_gse7621_neuron_adjusted")]
RY = [29.0, 24.5, 20.0]; ch = 4.0
GM = 1.5; norm = Normalize(-GM, GM)
for (lab, col), yy in zip(ROWS, RY):
    M.text(X0 - 1.5, yy, lab, ha="right", va="center", fontsize=6.0, color=INK)
    for j, r in P.iterrows():
        v = r[col]
        face = "#FFFFFF" if not np.isfinite(v) else EFFECT(norm(np.clip(v, -GM, GM)))
        M.add_patch(Rectangle((X0 + j * cw, yy - ch / 2), cw, ch, facecolor=face, edgecolor="white", lw=0.7))
        if col != "g_lcm" and np.isfinite(v) and np.sign(v) != np.sign(r.g_lcm):
            M.plot([X0 + j * cw + 0.8, X0 + (j + 1) * cw - 0.8], [yy - ch / 2 + 0.7, yy + ch / 2 - 0.7], color="#8C9299", lw=0.4)
for j, r in P.iterrows():
    M.text(X0 + (j + 0.5) * cw, RY[-1] - ch / 2 - 0.8, r.symbol, ha="center", va="top", rotation=90, fontsize=5.7,
           fontstyle="italic", color=SETS["both" if r.deg else "boruta"][1])
for grp, k in ((0, "both"), (1, "boruta")):
    js = np.where(P.grp.to_numpy() == grp)[0]
    if len(js):
        M.add_patch(Rectangle((X0 + js.min() * cw + 0.2, RY[0] + ch / 2 + 1.0), (js.max() - js.min() + 1) * cw - 0.4, 0.8,
                              facecolor=SETS[k][0], edgecolor="none"))
        M.text(X0 + (js.min() + js.max() + 1) / 2 * cw, RY[0] + ch / 2 + 2.4, f"{SETS[k][2]}  ({len(js)})", ha="center",
               va="bottom", fontsize=6.0, color=SETS[k][1])
# agreement, stated once at the right of each GSE7621 row
for key, yy in (("raw", RY[1]), ("neuron_adjusted", RY[2])):
    a = AG[key]
    M.text(181.0, yy, f"{a['same_direction']}/{a['n']}", ha="right", va="center", fontsize=6.3, color=INK, fontweight="bold")
M.text(181.0, RY[0], "same sign", ha="right", va="center", fontsize=5.7, color=MUTED)
cax = fig.add_axes([8.0 / FW, 2.2 / FH, 22.0 / FW, 1.6 / FH])
cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=EFFECT); cax.set_xticks([]); cax.set_yticks([])
for s_ in cax.spines.values():
    s_.set_linewidth(0.3); s_.set_color("#B7BDC4")
M.text(7.0, 3.0, f"−{GM:g}", ha="right", va="center", fontsize=5.7, color=MUTED)
M.text(31.0, 3.0, f"+{GM:g}   Hedges' g, PD minus control", ha="left", va="center", fontsize=5.7, color=MUTED)
ar, aa = AG["raw"], AG["neuron_adjusted"]
M.text(181.0, 3.0, f"struck-through cell: opposite sign to the laser-capture cohort   ·   all genes {100 * ar['background_share']:.0f}% same sign"
       f"   ·   after neuron removal P = {aa['binom_p']:.3f}", ha="right", va="center", fontsize=5.7, color=MUTED)

for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"Figure06_external_validation.{ext}", dpi=600)
print("saved", sorted(p.name for p in FIG_DIR.iterdir()))
plt.show() if IN_NB else plt.close(fig)''')

md("## 6. Suggested legend for the manuscript")

code(r'''ar, aa = AG["raw"], AG["neuron_adjusted"]
print(f"""External validation on bulk substantia nigra (GSE7621; {int((y7 == 0).sum())} control, {int(y7.sum())} PD). Nothing was fitted on
this cohort: its genes were z-scored within the cohort and ranked within each person, and scored by models trained only on the 63
laser-capture people. (a) ROC of the frozen core classifier (AUC {tc.auc:.2f}, 95% CI {tc.ci_lo:.2f}-{tc.ci_hi:.2f}; label permutation
P = {tc.perm_p:.4f}), the Boruta-panel forest ({tp.auc:.2f}) and an eight-gene dopamine-neuron marker score ({NEURO_AUC:.2f}). (b) The core
classifier's score against dopamine-neuron content (Spearman {tc.rho_neuron:.2f}); with the neuron component removed its AUC is
{tc.auc_after_neuron_removed:.2f}, and adding it to neuron content raises the leave-one-out AUC from {tc.loo_neuron_only:.2f} to
{tc.loo_neuron_plus_score:.2f}. (c) Effect size of each Boruta gene in the laser-capture cohort (pooled) and in GSE7621, before and after
neuron content is regressed out; {ar['same_direction']}/{ar['n']} and {aa['same_direction']}/{aa['n']} genes share the laser-capture sign
(binomial P = {ar['binom_p']:.3f} and {aa['binom_p']:.3f}), against {100 * ar['background_share']:.0f}% of all genes.""")''')

write_nb(HERE / "PD_LCM_rf_external.ipynb", CELLS, "ext")
