from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
import pandas as pd
import io
import json

from app.db.database import get_session
from app.db.orm import Hypothesis, Problem

router = APIRouter()


@router.get("/{path_id}")
async def export_handler(
    path_id: str,
    format: str = Query("json"),
    problem_id: str = Query(None),
):
    if path_id in ("json", "csv", "xlsx"):
        actual_format = path_id
        actual_problem_id = problem_id
    else:
        actual_format = format
        actual_problem_id = path_id

    if not actual_problem_id:
        raise HTTPException(status_code=400, detail="problem_id is required")

    if actual_format not in ["json", "csv", "xlsx"]:
        raise HTTPException(status_code=400, detail="Unsupported format. Use json, csv, or xlsx")

    return await _export_impl(actual_problem_id, actual_format)


async def _export_impl(problem_id: str, format: str):
    async with get_session() as session:
        problem = await session.get(Problem, problem_id)
        if not problem:
            raise HTTPException(status_code=404, detail="Problem not found")

        stmt = select(Hypothesis).where(Hypothesis.problem_id == problem_id)
        result = await session.execute(stmt)
        hypotheses = result.scalars().all()

        if not hypotheses:
            raise HTTPException(status_code=404, detail="No hypotheses found for this problem")

        if format == "json":
            data = {
                "problem_id": problem_id,
                "problem_statement": problem.statement,
                "hypotheses_count": len(hypotheses),
                "hypotheses": [
                    {
                        "statement": h.statement,
                        "mechanism": h.mechanism,
                        "citations": h.citations,
                        "novelty": h.novelty,
                        "feasibility": h.feasibility,
                        "impact": h.impact,
                        "risk": h.risk,
                        "composite_score": h.composite_score,
                        "confidence": h.confidence,
                        "risks": h.risks,
                        "verification_plan": h.verification_plan,
                        "reasoning_trace": h.reasoning_trace,
                    }
                    for h in hypotheses
                ]
            }
            json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            return StreamingResponse(
                io.BytesIO(json_bytes),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=hypotheses_{problem_id[:8]}.json"
                }
            )

        elif format == "csv":
            data = [
                {
                    "statement": h.statement,
                    "mechanism": h.mechanism,
                    "novelty": h.novelty,
                    "feasibility": h.feasibility,
                    "impact": h.impact,
                    "risk": h.risk,
                    "composite_score": h.composite_score,
                    "confidence": h.confidence,
                    "reasoning_trace": h.reasoning_trace or "",
                }
                for h in hypotheses
            ]
            df = pd.DataFrame(data)
            csv_data = df.to_csv(index=False)
            
            return StreamingResponse(
                io.BytesIO(csv_data.encode('utf-8')),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=hypotheses_{problem_id[:8]}.csv"
                }
            )

        elif format == "xlsx":
            data = [
                {
                    "Гипотеза": h.statement,
                    "Механизм": h.mechanism,
                    "Новизна": h.novelty,
                    "Реализуемость": h.feasibility,
                    "Эффект": h.impact,
                    "Риск": h.risk,
                    "Composite": h.composite_score,
                    "Уверенность": h.confidence,
                    "Риски": json.dumps(h.risks or [], ensure_ascii=False),
                    "План проверки": h.verification_plan or "",
                    "Цепочка рассуждений": h.reasoning_trace or "",
                }
                for h in hypotheses
            ]
            df = pd.DataFrame(data)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Hypotheses')
                
                if any(h.reasoning_trace for h in hypotheses):
                    cot_data = [
                        {
                            "Гипотеза": h.statement,
                            "Цепочка рассуждений": h.reasoning_trace,
                            "Источники": json.dumps(h.citations or [], ensure_ascii=False),
                        }
                        for h in hypotheses if h.reasoning_trace
                    ]
                    if cot_data:
                        df_cot = pd.DataFrame(cot_data)
                        df_cot.to_excel(writer, index=False, sheet_name='Chain-of-Thought')
            
            output.seek(0)
            
            return StreamingResponse(
                output,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={
                    "Content-Disposition": f"attachment; filename=hypotheses_{problem_id[:8]}.xlsx"
                }
            )
