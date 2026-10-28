from tools.assess_farming_risk import assess_farming_risk


def make_decision(question: str, problem: dict, knowledge: dict, weather: dict) -> dict:
    risk = assess_farming_risk(question, weather.get("forecast") if weather.get("available") else None)
    needs_expert = (not knowledge.get("evidence_found", False)) or problem.get("status") != "classified" or risk["risk_level"] == "high"
    return {"risk": risk, "needs_expert_review": bool(needs_expert), "reason": "Evidence is missing, the classifier is unavailable, or the case contains urgent signals." if needs_expert else "Evidence is available; advice still requires careful grounding in the cited documents."}
