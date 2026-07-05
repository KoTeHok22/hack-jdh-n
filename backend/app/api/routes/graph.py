from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from app.db.database import get_session
from app.db.orm import Problem, Hypothesis
from app.services.graph_builder import _build_graph_from_hypotheses

router = APIRouter()


@router.get("")
async def get_graph(problem_id: str = None):
    if not problem_id:
        return {
            "nodes": [],
            "edges": [],
            "stats": {"total_nodes": 0, "total_edges": 0}
        }
    
    async with get_session() as session:
        problem = await session.get(Problem, problem_id)
        if not problem:
            raise HTTPException(status_code=404, detail="Problem not found")

        hyp_stmt = select(Hypothesis).where(Hypothesis.problem_id == problem_id)
        hyp_result = await session.execute(hyp_stmt)
        hypotheses = hyp_result.scalars().all()
        
        if not hypotheses:
            return {
                "nodes": [],
                "edges": [],
                "stats": {"total_nodes": 0, "total_edges": 0},
                "message": "No hypotheses found for this problem"
            }
        
        hyp_dicts = [
            {
                "statement": h.statement,
                "mechanism": h.mechanism,
            }
            for h in hypotheses
        ]
        
        graph = _build_graph_from_hypotheses(
            problem_statement=problem.statement or "",
            hypotheses=hyp_dicts,
        )
        
        return graph
