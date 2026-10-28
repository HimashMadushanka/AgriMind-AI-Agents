"""Load the validated best model and classify a Sinhala or English question.

Run notebooks 07, 08 (optional), and 09 before using CLASSIFIER_BACKEND=auto.
The default auto mode uses the model chosen in selected_model.json; before model
selection it falls back to the TF-IDF + LinearSVC baseline.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SELECTION_PATH = ROOT / "models" / "model_evaluation_results" / "selected_model.json"


def _backend_name() -> str:
    requested = os.getenv("CLASSIFIER_BACKEND", "auto").strip().lower()
    if requested != "auto":
        return requested
    if SELECTION_PATH.exists():
        try:
            selected = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
            return str(selected.get("selected_model", "tfidf_linear_svc")).strip().lower()
        except (OSError, json.JSONDecodeError):
            pass
    return "tfidf_linear_svc"


@lru_cache(maxsize=1)
def _load_sklearn_model() -> Any:
    import joblib

    path = ROOT / "models" / "problem_classifier" / "tfidf_linear_svc.joblib"
    if not path.exists():
        raise FileNotFoundError(
            "Baseline model not found. Run notebooks/07_baseline_nlp_models.ipynb first."
        )
    return joblib.load(path)


@lru_cache(maxsize=1)
def _load_xlm_model() -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = ROOT / "models" / "problem_classifier" / "xlm_roberta_base" / "best_model"
    if not path.exists():
        raise FileNotFoundError(
            "XLM-RoBERTa model not found. Run notebooks/08_train_xlm_roberta_classifier.ipynb first."
        )
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def classify_problem(question: str) -> dict[str, Any]:
    """Return a predicted farming-problem category for a question.

    `model_score` is an uncalibrated softmax score for XLM-RoBERTa, not a
    guaranteed probability that the answer is correct. LinearSVC returns only
    a label here, so no confidence value is reported for that model.
    """
    if not isinstance(question, str) or len(question.strip()) < 2:
        raise ValueError("question must be a non-empty string with at least two characters")

    backend = _backend_name()
    if backend == "tfidf_linear_svc":
        model = _load_sklearn_model()
        label = str(model.predict([question.strip()])[0])
        return {"category": label, "model": backend}

    if backend == "xlm_roberta_base":
        import torch

        tokenizer, model, device = _load_xlm_model()
        inputs = tokenizer(question.strip(), return_tensors="pt", truncation=True, max_length=160)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(model(**inputs).logits[0], dim=-1)
        index = int(torch.argmax(probabilities).item())
        id2label = {int(key): value for key, value in model.config.id2label.items()}
        return {
            "category": str(id2label.get(index, index)),
            "model": backend,
            "model_score": float(probabilities[index].item()),
            "score_note": "uncalibrated model score; not a verified probability",
        }

    raise ValueError(
        "CLASSIFIER_BACKEND must be auto, tfidf_linear_svc, or xlm_roberta_base."
    )
