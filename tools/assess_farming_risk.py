def assess_farming_risk(question: str, weather: dict | None = None) -> dict:
    """A conservative triage example. Replace rules with expert-reviewed local criteria."""
    text = (question or "").lower()
    high_signals = ["pesticide poisoning", "poisoning", "can't breathe", "cannot breathe", "severe flooding", "කුඹුර යටවෙලා", "විෂ වීම", "හුස්ම ගන්න බැහැ"]
    medium_signals = ["spreading", "rapidly spreading", "many plants", "large area", "පැතිරෙනවා", "ගොඩක් පැළ", "කහ වෙනවා", "වියළෙනවා"]
    if any(term in text for term in high_signals):
        level, reason = "high", "Potentially urgent safety or widespread-damage wording; seek prompt qualified help."
    elif any(term in text for term in medium_signals):
        level, reason = "medium", "Symptoms may need timely inspection; more information is needed."
    else:
        level, reason = "unknown", "Insufficient evidence for a reliable risk estimate."
    return {"risk_level": level, "reason": reason, "weather_available": bool(weather)}
