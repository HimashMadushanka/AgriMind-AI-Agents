from fastapi import APIRouter, HTTPException
from api.schemas import FarmerQuestion, AssistantResponse
from agents.orchestrator import handle_farmer_question

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/ask", response_model=AssistantResponse)
def ask(payload: FarmerQuestion):
    try:
        result = handle_farmer_question(payload.question, payload.latitude, payload.longitude)
        return AssistantResponse(question=payload.question, advisory=result["advisory"], result=result)
    except Exception as exc:
        # Keep the returned message safe; add structured server-side logging in production.
        raise HTTPException(status_code=503, detail="The assistant could not process this request safely. Please try later or contact an agricultural officer.") from exc
