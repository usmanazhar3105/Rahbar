# Failure Analysis

This document details four deliberate failure experiments conducted across the RAHBAR RAG pipeline stages. In each experiment, a single component was broken in isolation, observed under test queries, diagnosed using observability metrics and structured logs, and then restored.

---

## Stage A · Extraction without OCR

### What We Broke
In `indexer/pipeline/extract.py`, we bypassed the OCR fallback path by forcing `had_ocr = False` and disabling the `pytesseract` and `pdf2image` execution branch. Scanned image-only PDFs (specifically `Faculty Entrepreneurship and Technical Start-ups Policy`, `27.pdf?20220530`) were processed exclusively using standard `pypdf` text extraction.

### Test Procedure
1. Modify `indexer/pipeline/extract.py` to skip `pdf2image` and `pytesseract` even when raw text characters are $<50$.
2. Run the indexer: `python -m pipeline.main --corpus corpus/raw`.
3. Query the API: *"What percentage of equity may a faculty member hold in a university spin-off startup?"*

### Symptom
The user receives an honest abstention answer:
```json
{
  "answer": "I cannot answer this question because the provided corpus does not contain information regarding faculty equity in startup spin-offs.",
  "citations": [],
  "answered_from_context": false,
  "confidence": "low"
}
```
If the relevance floor is not reached, the retriever returns 0 chunks and the API abstains completely. If low-similarity chunks from unrelated policies are pulled in, the LLM correctly reports that the context lacks the answer.

### Diagnosis
- **Metric**: The per-document chunk count logged by `indexer/pipeline/store.py` revealed `Faculty Entrepreneurship Policy: 0 chunks stored` (or empty strings). In evaluation, `recall@5` for queries targeting this document plummeted to `0.00`.
- **Log Line**: Indexer log showed:
  ```text
  [extract] 27.pdf: page 1 extracted 0 chars (had_ocr=False)
  [chunk] 27.pdf produced 0 non-empty chunks
  ```
- **Direct Inspection**: Running `pdffonts 27.pdf` showed an empty font table, confirming the file is a pure bitmap image with no embedded digital text layer.

### Remedy
Restore the OCR detection threshold in `indexer/pipeline/extract.py`:
```python
if len(text.strip()) < 50:
    images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
    text = pytesseract.image_to_string(images[0])
    had_ocr = True
```

### Key Insight
> "If I hadn't caused this on purpose, I would have found it by looking at the **per-source chunk counts printed by the indexer during the Store stage**."

---

## Stage B · Bad Chunking (100-Token Fixed Windows, No Overlap)

### What We Broke
In `indexer/pipeline/chunk.py`, we replaced structure-aware splitting with naive fixed-size chunking. We set `chunk_size = 100`, `overlap = 0`, and disabled heading detection and paragraph boundary splitting.

### Test Procedure
1. Set `chunk_size=100`, `overlap=0`, and split text strictly by word/token offsets regardless of sentence punctuation or headings.
2. Re-index the corpus and execute `python eval/run_eval.py --golden eval/golden.yaml`.
3. Query exact-term and multi-clause questions (e.g., Clause 5.4 regarding Academic Warning conditions).

### Symptom
The user sees fragmented, contradictory, or truncated answers. When asking *"Under what circumstances is a student dropped from the university?"*, the model answered that a student is warned for CGPA $< 2.00$, but omitted the condition *"dropped after three consecutive semesters"* because that clause was split across an arbitrary 100-token boundary into the next chunk which was not retrieved.

### Diagnosis
- **Metric**: In `eval/results.json`, `recall@5` for `exact_term` questions fell from `1.00` to `0.45`, and overall `recall@5` dropped by 22 points (triggering the `MAX_RECALL5_DROP` CI gate failure).
- **Log Line**: The API's structured retrieval log showed low cosine similarity scores:
  ```json
  {"request_id": "8b7e...", "top_k": 5, "similarity_scores": [0.32, 0.31, 0.28, 0.25, 0.22], "abstained": true}
  ```
  Chunks were cut off mid-sentence (e.g. `"...student whose cumulative grade point average ("`), stripping the semantic context needed for high vector similarity.

### Remedy
Restore structure-aware chunking in `indexer/pipeline/chunk.py`:
- Identify section headings (`1. Attendance`, `5.4 Academic Warning`).
- Split on natural paragraph breaks (`\n\n`) and sentence delimiters.
- Budget target chunk size to 400 tokens with an 80-token overlap window.

### Key Insight
> "If I hadn't caused this on purpose, I would have found it by looking at the **breakdown metric `by_type["exact_term"]["recall@5"]` in `eval/results.json` and inspecting the truncated raw text of retrieved passages**."

---

## Stage E · No Superseded Filter (Source & Metadata Filtering)

### What We Broke
In `api/app/retrieval.py` and `api/app/retrieval/chroma.py`, we omitted the metadata filter `where={"superseded": False}` when executing ChromaDB queries, allowing repealed policy documents to compete equally with current regulations.

### Test Procedure
1. Modify `retrieve()` to pass `where=None` unconditionally.
2. Query standard current policy questions:
   - *"What is the minimum attendance percentage required to sit final exams?"*
   - *"What happens to my grade if I repeat a course?"*

### Symptom
The user receives outdated or dangerous policy advice:
- The model reports: *"The minimum attendance requirement is 75 percent"* (citing the repealed August 2023 rules instead of the current 2026 handbook requirement of 80%).
- For course repeats, the model claims grades are capped at `B` (2023 rule), rather than the current policy where the most recent grade stands uncapped.

### Diagnosis
- **Metric**: Golden evaluation questions of type `version_sensitive` failed.
- **Log Line**: Retrieval debug logs revealed that the 2023 chunk had higher cosine similarity than the 2026 chunk:
  ```json
  {
    "retrieved": [
      {"source": "undergraduate-academic-rules-aug-2023.pdf", "section": "1. Attendance", "score": 0.88},
      {"source": "academic-rules-handbook-jun-2026.pdf", "section": "1.21 Attendance", "score": 0.86}
    ]
  }
  ```
  Because the query did not request historical rules, retrieving superseded passages produced incorrect answers.
- **Citations**: The returned citation pointed to `undergraduate-academic-rules-aug-2023.pdf` for a present-tense query.

### Remedy
Reinstate the metadata filter in `retrieval.py`:
```python
is_historical = any(phrase in question.lower() for phrase in ["old version", "previous policy", "2023 rules"])
where_filter = None if is_historical else {"superseded": False}
results = collection.query(query_embeddings=[query_vector], n_results=top_k, where=where_filter)
```

### Key Insight
> "If I hadn't caused this on purpose, I would have found it by looking at the **`source` filenames listed in the citations of `AskResponse` and checking the `where` parameter in retrieval logs**."

---

## Stage F · High LLM Temperature (Temperature = 1.2)

### What We Broke
In `api/app/providers/llm/hosted.py`, we set the Groq provider generation temperature to `1.2` (default is `0.1` or `0.0`).

### Test Procedure
1. Set environment variable `TEMPERATURE=1.2` in the API configuration.
2. Run the eval suite: `python eval/run_eval.py --golden eval/golden.yaml`.
3. Observe generation stability across multiple identical questions.

### Symptom
1. **Schema Breakage**: The API logged repeated Pydantic `ValidationError` exceptions because the model output conversational chatter before or after the JSON payload, unquoted keys, or malformed markdown blocks.
2. **Hallucinated Citations**: The model answered with confident prose but cited fabricated clause numbers (e.g. citing `clause 9.1` or non-existent sections) not present in the retrieved passages.
3. **Flaky Answers**: Identical requests returned differing numbers, policy interpretations, and confidence levels.

### Diagnosis
- **Metric**:
  - `citation_validity` in `eval/results.json` dropped from `1.00` to `0.72` (failing the `MIN_CITATION_VALIDITY = 0.95` gate).
  - Prometheus metric `rahbar_llm_validation_retries_total` and `rahbar_llm_fallback_total` spiked significantly.
- **Log Line**:
  ```text
  WARNING: Model response validation failed on attempt 1: 1 validation error for Answer (JSONDecodeError)
  ERROR: Both attempts failed schema validation. Returning safe fallback.
  ```

### Remedy
Enforce deterministic decoding in `api/app/providers/llm/hosted.py`:
- Set `temperature = 0.1` (or `0.0`) in the LLM completion payload.
- Enforce strict JSON mode (`response_format={"type": "json_object"}`).

### Key Insight
> "If I hadn't caused this on purpose, I would have found it by looking at the **`citation_validity` score in evaluation results and the `rahbar_llm_validation_retries_total` counter in `/metrics`**."

---

## Architectural Conclusion

| Failure Stage | Layer Responsible | Typical Misdiagnosis | Actual Root Cause |
| --- | --- | --- | --- |
| **Stage A (Extract)** | Data Ingestion | "The LLM does not understand start-up policies." | Scanned PDF was indexed as empty string due to skipped OCR. |
| **Stage B (Chunk)** | Indexing Pipeline | "The model cannot follow complex multi-part logic." | 100-token window severed clause from heading and dropped recall. |
| **Stage E (Filter)** | Retrieval Layer | "The LLM is hallucinating older rules." | Chroma query lacked `where={"superseded": False}` filter. |
| **Stage F (Generation)** | LLM Decoding | "The prompt template is wrong." | High temperature caused ungrounded tokens and schema failure. |

> **Key takeaway from the assignment:**
> **Only Stage F is the model's fault. Check stages A through E first.**
> In a RAG pipeline, generation errors are rarely caused by model deficiency. If extraction is empty, chunking is fractured, retrieval misses the right passage, or repealed versions are fed into context, even the most capable LLM will hallucinate or fail. Fix the pipeline before blaming the model.
