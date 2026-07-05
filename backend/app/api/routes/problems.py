import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import select

from app.services.generation import run_full_generation
from app.db.database import get_session
from app.db.orm import Problem, Hypothesis

router = APIRouter()


class ProblemCreate(BaseModel):
    statement: Optional[str] = None
    problem_description: Optional[str] = None
    target_kpi: Optional[str] = None
    target_delta: Optional[str] = None
    constraints: Optional[dict] = None
    domain: Optional[str] = None
    document_ids: Optional[list[str]] = None
    num_hypotheses: int = 8
    model: Optional[str] = None
    mode: str = "single"
    
    def get_statement(self) -> str:
        return self.statement or self.problem_description or ""


class ProblemResponse(BaseModel):
    problem_id: str
    statement: str
    parsed_problem: dict
    hypotheses: list[dict]
    context_chunks_count: int
    references: list[dict] = []


@router.get("")
async def list_problems():
    async with get_session() as session:
        stmt = select(Problem).order_by(Problem.created_at.desc())
        result = await session.execute(stmt)
        problems = result.scalars().all()
        
        return {
            "problems": [
                {
                    "problem_id": str(p.id),
                    "statement": p.statement,
                    "target_kpi": p.target_kpi,
                    "domain": p.domain,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                }
                for p in problems
            ]
        }


@router.get("/{problem_id}")
async def get_problem(problem_id: str):
    async with get_session() as session:
        problem = await session.get(Problem, problem_id)
        if not problem:
            raise HTTPException(status_code=404, detail="Problem not found")
        
        hyp_stmt = select(Hypothesis).where(Hypothesis.problem_id == problem_id)
        hyp_result = await session.execute(hyp_stmt)
        hypotheses = hyp_result.scalars().all()
        
        return {
            "problem_id": str(problem.id),
            "statement": problem.statement,
            "target_kpi": problem.target_kpi,
            "target_delta": problem.target_delta,
            "constraints": problem.constraints,
            "domain": problem.domain,
            "document_ids": problem.document_ids,
            "created_at": problem.created_at.isoformat() if problem.created_at else None,
            "hypotheses": [
                {
                    "statement": h.statement,
                    "mechanism": h.mechanism,
                    "citations": h.citations or [],
                    "novelty": h.novelty,
                    "feasibility": h.feasibility,
                    "impact": h.impact,
                    "risk": h.risk,
                    "confidence": h.confidence,
                    "composite_score": h.composite_score,
                    "reasoning_trace": h.reasoning_trace,
                    "reasoning_text": h.reasoning_trace,
                    "risks": h.risks or [],
                    "verification_plan": h.verification_plan,
                    "scores": {
                        "novelty": h.novelty,
                        "feasibility": h.feasibility,
                        "impact": h.impact,
                        "risk": h.risk,
                        "confidence": h.confidence,
                    },
                }
                for h in hypotheses
            ],
        }


@router.post("")
async def create_problem(request: ProblemCreate):
    statement = request.get_statement()
    if not statement.strip():
        raise HTTPException(status_code=400, detail="Statement cannot be empty")

    problem_id = str(uuid.uuid4())

    result = await run_full_generation(
        statement=statement,
        document_ids=request.document_ids,
        num_hypotheses=request.num_hypotheses,
        model=request.model,
        mode=request.mode,
    )

    try:
        async with get_session() as session:
            problem_db = Problem(
                id=problem_id,
                statement=statement,
                target_kpi=request.target_kpi,
                target_delta=request.target_delta,
                constraints=request.constraints,
                domain=request.domain,
                document_ids=request.document_ids,
            )
            session.add(problem_db)
            
            for hyp in result.get("hypotheses", []):
                hypothesis_db = Hypothesis(
                    problem_id=problem_id,
                    statement=hyp.get("statement", ""),
                    mechanism=hyp.get("mechanism", ""),
                    citations=hyp.get("citations", []),
                    novelty=hyp.get("novelty", 0),
                    feasibility=hyp.get("feasibility", 0),
                    impact=hyp.get("impact", 0),
                    risk=hyp.get("risk", 0),
                    confidence=hyp.get("confidence", 0),
                    composite_score=hyp.get("composite_score", 0),
                    reasoning_trace=hyp.get("reasoning_trace"),
                    risks=hyp.get("risks", []),
                    verification_plan=hyp.get("verification_plan"),
                )
                session.add(hypothesis_db)
            
            await session.commit()
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to save problem to PostgreSQL: {e}")

    hypotheses_with_scores = []
    for hyp in result.get("hypotheses", []):
        hyp_copy = hyp.copy()
        hyp_copy["scores"] = {
            "novelty": hyp.get("novelty", 0),
            "feasibility": hyp.get("feasibility", 0),
            "impact": hyp.get("impact", 0),
            "risk": hyp.get("risk", 0),
            "confidence": hyp.get("confidence", 0),
        }
        hyp_copy["reasoning_text"] = hyp.get("reasoning_trace", "")
        hypotheses_with_scores.append(hyp_copy)

    return ProblemResponse(
        problem_id=problem_id,
        statement=statement,
        parsed_problem=result["parsed_problem"],
        hypotheses=hypotheses_with_scores,
        context_chunks_count=len(result.get("context_chunks", [])),
        references=result.get("references", []),
    )
