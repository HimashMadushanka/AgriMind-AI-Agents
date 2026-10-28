from tools.assess_farming_risk import assess_farming_risk

def test_risk_tool_returns_valid_shape():
    result = assess_farming_risk("I have a question about paddy farming")
    assert result["risk_level"] in {"high", "medium", "unknown"}
    assert "reason" in result
