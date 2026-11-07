"""Question-answering endpoints: POST /api/ask and POST /api/chat."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..schemas import AskRequest, AskResponse
from ..services.answer import answer_question

router = APIRouter(prefix="/api")


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    state = request.app.state
    return answer_question(body.question, state.retriever, state.provider, state.settings)


@router.post("/chat", response_model=AskResponse)
def chat(body: AskRequest, request: Request) -> AskResponse:
    state = request.app.state
    return answer_question(body.question, state.retriever, state.provider, state.settings)
