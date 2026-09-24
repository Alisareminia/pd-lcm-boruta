# ============================== LOAD PIPELINE OUTPUTS ==============================
def find_outputs_dir():
    """Locate the pipeline notebook's outputs folder; /kaggle/input wins."""
    hits = []
    for root in ("/kaggle/input", ".", "..", "../.."):
        if not Path(root).exists():
            continue
        found = glob.glob(f"{root}/**/00_manifest.csv", recursive=True)
        if found and root == "/kaggle/input":
            return Path(sorted(found, key=len)[0]).parent
        hits += found
    if not hits:
        raise FileNotFoundError(
            "Could not find 00_manifest.csv. Add the notebook "
            "'alisaremi/pd-lcm-merged-pipeline' as a data source, then re-run.")
    return Path(sorted(hits, key=len)[0]).parent

SRC = find_outputs_dir()
print("Reading pipeline outputs from:", SRC)

TABLES = {p.stem: pd.read_csv(p) for p in sorted(SRC.glob("*.csv"))}

def T(stem):
    if stem not in TABLES:
        raise KeyError(f"{stem!r} not found. Available:\n  " + "\n  ".join(sorted(TABLES)))
    return TABLES[stem]

print(f"Loaded {len(TABLES)} tables")

# ---- gene symbols -------------------------------------------------------
# The pipeline exports the Ensembl -> symbol map for every harmonised gene.
GENE_SYMBOLS = {}
for _g, _s in zip(T("15_gene_symbol_map")["gene"], T("15_gene_symbol_map")["symbol"]):
    if isinstance(_s, str) and _s and _s.lower() != "nan":
        GENE_SYMBOLS[_g] = _s

def sym(g):
    """Ensembl ID -> HGNC symbol. Unnamed transcripts keep their Ensembl ID."""
    s = GENE_SYMBOLS.get(g)
    return s if isinstance(s, str) and s else str(g)

def named(g):
    """True when a real gene symbol exists (i.e. not an unnamed transcript)."""
    return g in GENE_SYMBOLS

_n_named = sum(1 for g in T("06_gene_set_membership")["gene"] if named(g))
_n_tot   = len(T("06_gene_set_membership"))
print(f"Gene symbols: {len(GENE_SYMBOLS)} embedded; "
      f"{_n_named}/{_n_tot} of the Boruta/DEG union have an HGNC name")

HAS_FIGDATA = "15_roc_curves" in TABLES

# ---- study-level facts every figure needs ------------------------------
PREP     = T("01_preprocessing_summary")
N_BG     = int(PREP.loc[PREP["step"].str.contains("model X"), "genes"].iloc[0])
COHORT   = T("01_cohort_by_dataset")
N_PEOPLE = int(COHORT["people"].sum())
N_DS     = len(COHORT)
N_PANEL  = len(T("04_boruta_selected_genes"))
DE_SET   = T("03_de_settings").set_index("statistic")["value"]
DE_RULE  = str(DE_SET["de_rule"])
DE_SHORT = "DEGs" if DE_RULE == "strict" else "nominal DE"
DE_LONG  = str(DE_SET["de_label"])
print(f"{N_PEOPLE} people, {N_DS} datasets, {N_BG:,} genes, {N_PANEL}-gene Boruta panel; DE set: {DE_LONG}")
if not HAS_FIGDATA:
    print("\nNOTE: 15_* exports absent - re-run the pipeline notebook.")
display(T("00_manifest")[["n", "file", "rows", "description"]].head(40))