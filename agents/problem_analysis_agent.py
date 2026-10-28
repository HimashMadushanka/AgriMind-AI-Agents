import re
from tools.classify_problem import classify_problem


def analyze_problem(question: str) -> dict:
    if not question or not question.strip():
        return {"status": "needs_input", "follow_up_question": "කරුණාකර ඔබගේ වී වගාවේ ගැටලුව විස්තර කරන්න."}
    contains_sinhala = bool(re.search(r"[\u0D80-\u0DFF]", question))
    try:
        classification = classify_problem(question)
        return {"status": "classified", "language_hint": "si-LK" if contains_sinhala else "en-LK", **classification}
    except FileNotFoundError as exc:
        return {"status": "model_not_trained", "message": str(exc), "language_hint": "si-LK" if contains_sinhala else "en-LK"}
