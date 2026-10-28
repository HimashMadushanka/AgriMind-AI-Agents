import os


def generate_advisory(question: str, problem: dict, knowledge: dict, weather: dict, decision: dict) -> str:
    """Generate cautious Sinhala/English guidance grounded in retrieved evidence."""
    evidence = knowledge.get("evidence", [])
    if not evidence:
        # Fail closed: without retrieved, reviewable source material, do not use
        # a general-purpose model to invent farming treatment instructions.
        if any("\u0D80" <= char <= "\u0DFF" for char in question):
            return ("මෙම ප්‍රශ්නයට ගැළපෙන සත්‍යාපිත කෘෂිකාර්මික ලේඛන හමු නොවීය. "
                    "ඒ නිසා නිශ්චිත රෝග විනිශ්චයක් හෝ ප්‍රතිකාරයක් නිර්දේශ කළ නොහැක. "
                    "කරුණාකර ඔබගේ ප්‍රදේශයේ කෘෂිකර්ම නිලධාරියෙකුගෙන් උපදෙස් ලබාගන්න.")
        return ("No matching verified agricultural documents were found, so I cannot safely "
                "recommend a diagnosis or treatment. Please consult your local agricultural officer.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ("Gemini is not configured. Retrieved evidence is available, but no generated "
                "advisory was produced. Please review the cited documents with an agricultural officer. "
                f"Evidence documents found: {len(evidence)}.")
    from google import genai
    client = genai.Client(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    evidence_text = "\n\n".join([f"Source: {e['source']}\nEvidence: {e['text']}" for e in evidence])
    prompt = f"""You are a cautious agricultural information assistant for Sri Lankan paddy farming.
Answer in the same language as the farmer. Do not diagnose from insufficient evidence and do not invent pesticide/fertilizer rates.
Use only the supplied source extracts for agricultural claims. If evidence is missing or uncertain, say so and recommend contacting a qualified agricultural officer.
Farmer question: {question}
Problem classifier output: {problem}
Weather context: {weather}
Decision flags: {decision}
Retrieved source extracts:\n{evidence_text or 'No source documents found.'}
Include a short explanation, safe next steps, uncertainty, and source file names when available."""
    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text or "No advisory text was returned. Please contact an agricultural officer."
