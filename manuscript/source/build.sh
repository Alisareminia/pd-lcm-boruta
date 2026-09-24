#!/bin/bash
# Builds boruta_lcm_pd_signatures.pdf and boruta_lcm_pd_signatures.docx from one source. TinyTeX lives under $HOME, so no sudo is needed.
# latexmk and biblatex's bibtex backend disagree about when to re-run bibtex, so the passes are explicit.
set -euo pipefail
export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"
cd "$(dirname "$0")"

python3 scripts/build_tables.py
python3 scripts/build_table3.py        # the gene table the paper now uses

rm -f boruta_lcm_pd_signatures.aux boruta_lcm_pd_signatures.bbl boruta_lcm_pd_signatures.bcf boruta_lcm_pd_signatures.blg boruta_lcm_pd_signatures.run.xml boruta_lcm_pd_signatures.fdb_latexmk boruta_lcm_pd_signatures.fls
pdflatex -interaction=nonstopmode boruta_lcm_pd_signatures.tex > /dev/null
bibtex boruta_lcm_pd_signatures > /dev/null || true
pdflatex -interaction=nonstopmode boruta_lcm_pd_signatures.tex > /dev/null
pdflatex -interaction=nonstopmode boruta_lcm_pd_signatures.tex > /dev/null

# Word cannot render an embedded PDF image, so the docx pass points at the PNG copies
sed -E 's/(figures\/[A-Za-z0-9_]+)\.pdf/\1.png/g' boruta_lcm_pd_signatures.tex > .boruta_lcm_pd_signatures_docx.tex
pandoc .boruta_lcm_pd_signatures_docx.tex -o boruta_lcm_pd_signatures.docx --bibliography=references.bib --citeproc --resource-path=.:figures
rm -f .boruta_lcm_pd_signatures_docx.tex

echo "--- log check ---"
echo "errors:    $(grep -c '^!' boruta_lcm_pd_signatures.log || true)"
echo "undefined: $(grep -ci 'undefined' boruta_lcm_pd_signatures.log || true)"
grep "Output written" boruta_lcm_pd_signatures.log
