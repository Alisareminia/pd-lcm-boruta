# EndNote hand-off

Everything needed to turn this manuscript into a Word document with live EndNote citations.
It takes about ten minutes and needs no retyping of references.

## Files

| File | What it is |
| --- | --- |
| `endnote_paper_references.ris` | **Import this.** The 77 works the paper cites, in the order they are cited |
| `boruta_lcm_pd_signatures_ENDNOTE.docx` | **Open this.** The manuscript, with every citation written as an EndNote temporary citation |
| `endnote_full_library.ris` | Optional: all 123 references gathered for the project, including 46 the final text does not cite |
| `boruta_lcm_pd_signatures_REFERENCE.pdf` | The typeset paper, to check the result against |

## Steps

**1. Make a new, empty EndNote library.**
`File → New…`, name it, save it.

This matters. EndNote gives each record a number as it is imported, and the temporary citations in the
Word file refer to those numbers. Importing into an empty library makes record 1 the first reference,
record 77 the last, exactly as the Word file expects. Importing into a library that already holds
references shifts every number.

**2. Import the references.**
`File → Import → File…`
- Import File: `endnote_paper_references.ris`
- Import Option: **Reference Manager (RIS)**
- Duplicates: **Import All**
- Text Translation: **Unicode (UTF-8)** — the reference list contains names such as Smajić and Tïklová

You should end with **77 references**.

**3. Open the Word file.**
`boruta_lcm_pd_signatures_ENDNOTE.docx`. Its citations look like `{Kamath, 2022 #7}`. That is EndNote's
own unformatted form, not a mistake.

**4. Choose the output style** in the EndNote tab in Word (Vancouver, APA, or whatever the journal asks for).

**5. Click `Update Citations and Bibliography`** in the EndNote tab.

Every `{Author, Year #n}` becomes a real Cite While You Write field, and EndNote builds the reference list
under the References heading at the end. The style can be changed afterwards at any time and everything
reformats.

## If something does not match

EndNote matches on the record number first and on author and year second. If a citation stays unformatted
or opens a picker dialog, it is almost always because the library was not empty at step 1. Starting again
with a new library fixes it.

## Notes

- 165 temporary citations point at 77 distinct references; several citations name more than one work,
  written as `{Author, Year #1; Author, Year #2}`.
- Two references are organisations rather than people (the GBD 2021 Nervous System Disorders Collaborators,
  and The UniProt Consortium). They are stored as corporate authors and should not be split into a surname
  and initials.
- Every reference carries its DOI, checked against CrossRef. The one exception is Pedregosa et al. 2011
  (scikit-learn, *JMLR*), because that journal does not issue DOIs.
- The BibTeX key of each record is kept in EndNote's **Label** field, so any reference can be traced back
  to the source bibliography.
- Table 3 is long and in landscape in the typeset paper. Word will render it as a plain table; compare
  against the reference PDF before submitting.
