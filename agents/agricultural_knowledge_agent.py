from tools.search_agricultural_knowledge import search_agricultural_knowledge


def retrieve_evidence(question: str, k: int = 5) -> dict:
    evidence = search_agricultural_knowledge(question, k=k)
    return {"evidence": evidence, "evidence_found": bool(evidence), "warning": None if evidence else "No approved documents were found. Do not invent agricultural advice."}
