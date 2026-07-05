from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.services import generation

router = APIRouter()


class HypothesisRequest(BaseModel):
    statement: str
    document_ids: Optional[list] = None
    num_hypotheses: int = 8
    model: Optional[str] = None
    mode: str = "single"


@router.get("")
async def list_hypotheses(problem_id: str = None):
    return {"message": "Use POST /api/v1/problems to generate hypotheses"}


@router.post("/generate")
async def generate_hypotheses(request: HypothesisRequest):
    result = await generation.run_full_generation(
        statement=request.statement,
        document_ids=request.document_ids,
        num_hypotheses=request.num_hypotheses,
        model=request.model,
        mode=request.mode,
    )
    return {
        "parsed_problem": result["parsed_problem"],
        "hypotheses": result["hypotheses"],
        "references": result.get("references", []),
    }
