from __future__ import annotations

from ...schemas import Answer, Citation, Passage


class StubProvider:
    name = "stub"

    def generate(self, question: str, passages: list[Passage]) -> Answer:
        citations = []
        if passages:
            first_p = passages[0]
            citations = [
                Citation(
                    source=first_p.source,
                    section=first_p.section,
                    page=first_p.page or 1,
                )
            ]
        return Answer(
            reasoning="Grounded answer based on academic policy.",
            answer=f"Based on {citations[0].source if citations else 'the policy'}, {passages[0].text if passages else 'No information available.'}",
            citations=citations,
            answered_from_context=bool(passages),
            confidence="high" if passages else "low",
        )
