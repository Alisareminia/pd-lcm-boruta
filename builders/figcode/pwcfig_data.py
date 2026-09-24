import os, re, json, glob, textwrap
from pathlib import Path
import numpy as np, pandas as pd
ON_KAGGLE = Path("/kaggle/input").exists()
OUT = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("FIG_OUT", "."))
FIG_IN = Path(os.environ["PWC_IN"]) if not ON_KAGGLE else Path(sorted(
    {str(Path(h).parent) for h in glob.glob("/kaggle/input/**/gprofiler_results.csv", recursive=True)}, key=len)[0])
GPR = pd.read_csv(FIG_IN / "gprofiler_results.csv")
CALR = pd.read_csv(FIG_IN / "gprofiler_random_lists.csv")
SUM = json.load(open(FIG_IN / "pathway_composition_summary.json"))
CAL = SUM["gprofiler"]
LISTLAB = {"up": "Higher in PD (18 genes)", "down": "Lower in PD (12 genes)", "all": "All 30 panel genes"}
SRCLAB = {"GO:BP": "GO BP", "GO:MF": "GO MF", "GO:CC": "GO CC", "REAC": "Reactome", "KEGG": "KEGG", "WP": "WikiPW"}
wrap = lambda t, w: textwrap.wrap(t, w)
def cap(t): return t[0].upper() + t[1:]
print(f"g:Profiler terms: " + ", ".join(f"{k} {int((GPR.list == k).sum()):,} touched, {int(GPR[GPR.list == k].significant.sum())} significant"
                                        for k in ("up", "down", "all")))
