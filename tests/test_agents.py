from agents.problem_analysis_agent import analyze_problem

def test_blank_question_requests_input():
    result = analyze_problem("   ")
    assert result["status"] == "needs_input"


def test_advisory_fails_closed_without_verified_evidence(monkeypatch):
    from agents.advisory_generation_agent import generate_advisory

    # No key or model is needed for this branch; the agent must not invent advice.
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-unused")
    answer = generate_advisory(
        question="My rice plants are turning yellow",
        problem={"status": "classified", "category": "Disease management"},
        knowledge={"evidence": [], "evidence_found": False},
        weather={"available": False},
        decision={"needs_expert_review": True},
    )
    assert "verified agricultural documents were found" in answer
    assert "agricultural officer" in answer
