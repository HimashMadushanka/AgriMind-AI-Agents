# AI Paddy Farming Intelligence — File-by-file guide

## Start here

1. `README.md` — setup, run commands, model strategy, and warnings.
2. `.env.example` — safe configuration template. Copy it to `.env`; never commit `.env`.
3. `requirements.txt` — Python libraries used by the project.

## Dataset files

- `datasets/raw/AgriMind_Bilingual_Paddy_Farmer_QA_12720_Pairs.csv` — source A: one row contains Sinhala and English Q&A.
- `datasets/raw/AgriMind_Sinhala_English_Paddy_Farmer_QA_25440.csv` — source B: one row for each language-specific Q&A.
- `datasets/raw/sinhala_paddy_qa.csv` — convenient Sinhala-only export.
- `datasets/raw/english_paddy_qa.csv` — convenient English-only export.
- `datasets/processed/multilingual_paddy_qa.csv` — canonical merged file, 25,440 language-specific rows.
- `datasets/processed/cleaned_paddy_qa.csv` — whitespace-normalized and key-validated data.
- `datasets/processed/model_ready_paddy_qa.csv` — question text plus category label for classification.
- `datasets/processed/paddy_train.csv` — training examples for learning model parameters.
- `datasets/processed/paddy_validation.csv` — validation examples for choosing the model and settings.
- `datasets/processed/paddy_test.csv` — untouched final test set for final evaluation.
- `datasets/reports/` — data-quality report, class/language counts, and issue catalog.
- `datasets/external/agricultural_documents/` — place official/verified documents here for reference.
- `datasets/external/weather_data/` — optional licensed weather files or API snapshots.

## Notebook files — run in numeric order

| Notebook | What happens when you run it |
|---|---|
| `00_project_setup_and_environment.ipynb` | Checks Python, package versions, and GPU availability. |
| `01_combine_sinhala_english_datasets.ipynb` | Validates the two supplied CSV representations and writes one combined file without duplicating them. |
| `02_dataset_validation_and_cleaning.ipynb` | Checks required columns, blank text, IDs, and category consistency; saves cleaned data. |
| `03_exploratory_data_analysis.ipynb` | Plots language/category counts and question lengths; summarizes data provenance. |
| `04_duplicate_and_data_leakage_check.ipynb` | Finds duplicated questions and checks whether an issue ID maps to one category. |
| `05_multilingual_dataset_preparation.ipynb` | Creates the question/label table and numeric label mapping used by models. |
| `06_train_validation_test_split.ipynb` | Creates train/validation/test datasets while keeping all variants of one issue in one split. |
| `07_baseline_nlp_models.ipynb` | Trains a word/character TF-IDF + LinearSVC baseline and measures validation metrics. |
| `08_train_xlm_roberta_classifier.ipynb` | Fine-tunes multilingual XLM-RoBERTa and records validation results. Requires model downloads and is the most compute-intensive notebook. |
| `09_compare_nlp_models.ipynb` | Compares available model results using validation macro-F1, then accuracy as a tie-breaker. |
| `10_model_evaluation_and_error_analysis.ipynb` | Evaluates the selected model once on the held-out test set and shows mistakes and per-language accuracy. |
| `11_multilingual_embedding_experiments.ipynb` | Tests multilingual sentence embeddings on a sample for semantic similarity. |
| `12_agricultural_semantic_search.ipynb` | Searches official/expert-reviewed documents placed in `knowledge_base/documents/`. |
| `13_rag_question_answering.ipynb` | Retrieves evidence and, when configured, sends the evidence to Gemini for a cautious answer. |
| `14_gemini_api_integration.ipynb` | Checks the Gemini API key and runs a minimal API request. |
| `15_ai_agent_tool_calling.ipynb` | Demonstrates a controlled registry of Python tools used by an agent. |
| `16_multi_agent_workflow.ipynb` | Runs the end-to-end orchestrator across classifier, knowledge, weather/risk, decision support, and advisory. |
| `17_end_to_end_system_testing.ipynb` | Runs smoke tests for key project components. |

## Application and reusable code

- `tools/classify_problem.py` — loads the selected classification model. `CLASSIFIER_BACKEND=auto` follows notebook 09's choice, otherwise it defaults to TF-IDF.
- `tools/search_agricultural_knowledge.py` — extracts text from PDF/TXT/Markdown files, embeds chunks, and returns semantically similar evidence. Embeddings are cached during the app process.
- `tools/get_weather_forecast.py` — calls Open-Meteo only when coordinates are supplied.
- `tools/assess_farming_risk.py` — simple conservative triage rules; it is a prototype, not an expert-validated agronomic risk model.
- `agents/problem_analysis_agent.py` — handles empty questions and invokes the classifier.
- `agents/agricultural_knowledge_agent.py` — retrieves source evidence.
- `agents/weather_risk_agent.py` — calls weather only when location is supplied.
- `agents/decision_support_agent.py` — combines risk, classifier status, and evidence availability to decide whether expert review is needed.
- `agents/advisory_generation_agent.py` — gives a safe fallback when no evidence is found; otherwise uses Gemini if configured.
- `agents/orchestrator.py` — coordinates the agents.
- `api/main.py`, `api/routes.py`, `api/schemas.py` — FastAPI application, endpoints, and input validation.
- `frontend/app.py` — Streamlit demo for farmer questions.
- `database/db_connection.py` — SQLAlchemy database connection setup; persistent case storage needs explicit schema and privacy design before production.
- `tests/` — tests for the classifier's surrounding tools, retrieval behavior, agents, and API routes.
- `models/` — local model checkpoints and metrics created by running training notebooks; large trained model files are not pre-bundled.

## How to interpret accuracy

The supplied dataset marks every row `synthetic_template_generated_bilingual` and `not_expert_reviewed`; it contains no real farmer queries. The issue-grouped split is designed to reduce template leakage, but it does not replace testing on new, realistic questions collected and reviewed by agricultural experts. Do not publish a target accuracy before running the notebooks. Use test metrics and per-language/per-category errors to decide whether the model is ready for further testing.
