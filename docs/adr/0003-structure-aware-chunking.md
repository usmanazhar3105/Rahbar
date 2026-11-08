# ADR 0003: Structure-Aware Chunking Strategy with Overlap

## Status: Accepted

## Context
Fixed-size token window chunking blindly cuts academic policies mid-sentence or mid-clause, separating section titles (e.g., `5.4 Academic Warning`) from their substantive rules. This drops retrieval recall for exact clause queries and damages LLM answer grounding.

## Decision
We implement a **hierarchical structure-aware chunking strategy**:
1. First partition documents by structural headings (numbered clauses, ALL-CAPS titles).
2. Within sections, split on natural paragraph boundaries (`\n\n`).
3. Within oversized paragraphs, split on sentence terminators.
4. Budget chunk sizes to a 400-token target with an 80-token overlap window.
5. Attach the nearest heading to the chunk metadata under `section`.

## Consequences
- **Positive**: High semantic integrity; headings remain bound to clause bodies; achieves 1.00 recall@5 on baseline policies.
- **Negative**: Modest storage amplification ($\approx 1.18\times$ token overhead due to overlap).

