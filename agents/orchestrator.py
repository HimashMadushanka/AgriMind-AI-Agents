from agents.problem_analysis_agent import analyze_problem
from agents.agricultural_knowledge_agent import retrieve_evidence
from agents.weather_risk_agent import check_weather
from agents.decision_support_agent import make_decision
from agents.advisory_generation_agent import generate_advisory


def handle_farmer_question(question: str, latitude: float | None = None, longitude: float | None = None) -> dict:
    problem = analyze_problem(question)
    knowledge = retrieve_evidence(question)
    weather = check_weather(latitude, longitude)
    decision = make_decision(question, problem, knowledge, weather)
    advisory = generate_advisory(question, problem, knowledge, weather, decision)
    return {"question": question, "problem_analysis": problem, "knowledge": knowledge, "weather": weather,
            "decision_support": decision, "advisory": advisory}
