from __future__ import annotations

from ..schemas import Passage


class StubRetriever:
    """Stub retriever returning static policy passages for offline testing."""

    def retrieve(self, query: str, top_k: int = 5, relevance_floor: float = 0.3) -> list[Passage]:
        return [
            Passage(
                chunk_id="chunk-stub-1",
                text="The minimum CGPA requirement for graduation is 2.00. Students with CGPA below 2.00 are placed on academic warning.",
                source="academic_rules_handbook_jun2026.pdf",
                section="Section 5: Academic Warning",
                page=12,
                score=0.95,
            ),
            Passage(
                chunk_id="chunk-stub-2",
                text="Attendance of at least 75% in each course is required to be eligible to sit for the end semester examinations.",
                source="academic_rules_handbook_jun2026.pdf",
                section="Section 3: Attendance",
                page=8,
                score=0.88,
            ),
            Passage(
                chunk_id="chunk-stub-3",
                text="Students must register for courses during the announced registration window prior to semester start.",
                source="academic_rules_handbook_jun2026.pdf",
                section="Section 2: Course Registration",
                page=5,
                score=0.82,
            ),
        ]
