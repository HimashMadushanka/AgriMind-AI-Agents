# AI Paddy Farming Intelligence

A rice leaf image classifier built with EfficientNet-B0, with an optional Gemini explanation based on a local agricultural knowledge base.

## Requirements

Use Python 3.13. Install dependencies in a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch can use a CPU or a supported GPU. If pip selects a CPU build but you want GPU acceleration, install the matching PyTorch and torchvision builds for your CUDA version using the official PyTorch installer instructions.

## Local data and model files

Git ignores `/data/` and trained model files. They are not included in a fresh clone. Put the rice image dataset in `data/raw/`, with one folder for each class: `bacterial_leaf_blight`, `brown_spot`, `Healthy`, `leaf_blast`, `leaf_scald`, `narrow_brown_spot`, `neck_blast`, `rice_hispa`, `sheath_blight`, and `tungro`. Then run the notebooks in order:

1. `01_create_image_dataset.ipynb`
2. `02_validate_and_explore.ipynb`
3. `03_split_dataset.ipynb`
4. `04_train_efficientnet.ipynb`
5. `05_model_evaluation.ipynb`
6. `06_predict_image_and_llm.ipynb`
7. `07_create_client.ipynb` for the optional Gemini explanation

The training cell starts a new training run each time you execute it; it saves the best validation checkpoint to `models/rice_disease_model.pth`. To reuse a saved model, skip the training cell and run evaluation, prediction, or the app. The first training run may download pretrained EfficientNet weights. The app needs the trained model file; the image dataset is needed for notebook training and sample predictions.

## Run the app

```powershell
.venv\Scripts\Activate.ps1
streamlit run app.py
```

To enable Gemini explanations, copy `.env.example` to `.env` and set `GEMINI_API_KEY` (or configure the same key in Streamlit secrets). If Gemini is unavailable or the key is not configured, the app shows the verified local guidance instead.

The classifier is a screening aid. Confirm uncertain cases with a qualified agricultural officer.
