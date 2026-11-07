# Corpus

Corpus documents are not committed to this repository. Run `corpus/fetch.py` to download them. See `MANIFEST.md` for URLs and SHA-256 checksums of all documents.

Source: FAST-NUCES public website (`https://nu.edu.pk`) · Documents: 6 core documents + 1 excluded · Pages: 214 total pages

| # | Required property | Document(s) that provide it | Evidence or notes |
| --- | --- | --- | --- |
| 1 | At least 6 documents, 40 pages | `Academic_Rules_Handbook_2026.pdf` (53 p), `Academic-Rules-Undergrad-Aug-2023.pdf` (18 p), `Student_Handbook_2026.pdf` (42 p), `Annual_Report_2023-24.pdf` (85 p), `Fee_Structure_2026.pdf` (4 p), `27.pdf` (12 p) | Total: 6 distinct PDF documents across 214 pages, exceeding the 40-page minimum requirement. |
| 2 | Two versions of one policy | Older: `undergraduate-academic-rules-aug-2023.pdf` (`doc_version: "2023-08"`, `superseded: true`)<br>Newer: `academic-rules-handbook-jun-2026.pdf` (`doc_version: "2026-06"`, `superseded: false`) | The 2026 Handbook replaces the 2023 Rules. Changes include minimum attendance (75% in 2023 vs 80% in 2026), academic warning CGPA (1.75 in 2023 vs 2.00 in 2026), and grade capping on course repeats (capped at B in 2023 vs uncapped/most recent in 2026). Stored in chunk metadata under `doc_version` and `superseded`. |
| 3 | Scanned, image-only PDF | `faculty-entrepreneurship-and-technical-start-ups-policy.pdf` (`27.pdf?20220530`, 12 pages) | Published as pure scanned raster bitmaps with no digital text layer. `pdffonts` reports 0 embedded fonts; raw text extraction with `pypdf` yields 0 characters per page. Verified with `pdffonts corpus/raw/27.pdf`. Triggers OCR fallback (`pdf2image` + `pytesseract`) in `indexer/pipeline/extract.py`. |
| 4 | A table with real information | `Fee_Structure_2026.pdf` / `FeeStructure.html` | Tabular grids defining fee per credit hour (BS: PKR 10,000, MS: PKR 12,500) and other university administrative charges (Late registration: 3,000, Transcript: 1,500). Cleaned into structured markdown tables in `clean.py` to preserve row-to-column cell binding. |
| 5 | Long definitions | `Student_Handbook_2026.pdf` (Section 4: Student Code of Conduct & Academic Integrity) | Contains multi-paragraph legalistic definitions of plagiarism, academic misconduct, disciplinary probation, and student rights, testing the chunker's ability to preserve paragraph context across semantic splits. |
| 6 | A deliberately absent topic | `Transport and Hostel Facilities.pdf` (Hostel accommodation, mess rules, and university transport routes) | Plausibly queried by students but deliberately omitted from the ingested corpus. Used to evaluate honest abstention (`must_abstain: true`, requiring `answered_from_context: false` and `confidence: low`). |
| 7 | Exact-term content | `academic-rules-handbook-jun-2026.pdf`, `Fee_Structure_2026.pdf` | Specific clause numbers (Clause `1.21` for attendance, Clause `5.4` for academic warning and probation drops), numerical CGPA thresholds (`2.00`, `1.75`, `3.50`), and exact fee amounts (`10,000` PKR). |

## Corpus Verification Details

### Property 2: Version Filtering
When a user asks current academic policy questions, the retrieval layer automatically applies the metadata filter `where={"superseded": False}` so that repealed clauses from `undergraduate-academic-rules-aug-2023.pdf` are not presented as active university policy. When the user explicitly requests historical rules (e.g. queries mentioning `"2023 rules"` or `"old policy"`), the filter is relaxed to permit comparative retrieval.

### Property 3: OCR Pipeline Validation
`pdffonts corpus/raw/27.pdf` confirmed that no fonts are embedded in the PDF stream. The extract stage evaluates page character length; pages with fewer than 50 extracted characters are automatically routed to Tesseract OCR at 300 DPI, tagging the resulting page dict with `had_ocr=True`.

### Property 6: Deliberate Topic Omission
Hostel and transport boarding rules are entirely excluded from the indexing run. The evaluation suite includes two explicit golden questions (`q13` and `q14`) asserting that RAHBAR declines to answer and returns `answered_from_context: false` rather than fabricating campus hostel allotment policies.
