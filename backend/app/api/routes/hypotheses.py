from fastapi import APIRouter

router = APIRouter()


@router.get("")
async def list_hypotheses(problem_id: str = None):
    return {"message": "Use POST /api/v1/problems to generate hypotheses"}
