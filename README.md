# AI Paddy Farming Intelligence

An AI-assisted rice leaf screening app built for paddy farming. Upload a leaf image to receive the model’s top disease predictions and, when configured, a plain-language explanation grounded in a curated agricultural knowledge base.

<p align="center">
  <a href="https://agrimind-ai-agents.streamlit.app/"><strong>🌾 Open the live app</strong></a>
</p>

## What it does

- Classifies rice leaf images into 10 supported categories using EfficientNet-B0.
- Shows the top three model scores. Scores are not calibrated probabilities.
- Provides locally available agricultural guidance for each category.
- Optionally uses Gemini to explain that guidance in simple language. Gemini is instructed to use the supplied information and not invent pesticide names, doses, or treatments.
- Accepts JPG, JPEG, PNG, BMP, and WebP images up to 20 MB and 40 megapixels.

### Supported categories

Healthy, bacterial leaf blight, brown spot, leaf blast, leaf scald, narrow brown spot, neck blast, rice hispa, sheath blight, and tungro.

## Try the hosted app

Visit **[agrimind-ai-agents.streamlit.app](https://agrimind-ai-agents.streamlit.app/)**, upload a rice leaf image, and select **Analyze image**. The classifier works without Gemini; Gemini explanations depend on the hosted app’s secret configuration.

## Model evaluation

The evaluation artifacts included in this project report **98.41% accuracy** and **0.983 macro F1** on a held-out test split of 1,007 images. These results describe this dataset and split; they do not guarantee performance on field photographs or confirm a plant diagnosis.

## Run locally

The project targets Python 3.13.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run streamlit_app.py
```

The app requires the trained checkpoint at `models/rice_disease_model.pth`. Model weights are excluded from Git by `.gitignore`; place the checkpoint there before launching. The image dataset is only needed to run the training and evaluation notebooks.

### Optional Gemini explanations

Add your own key to `.env`:

```dotenv
GEMINI_API_KEY=your-api-key
GEMINI_MODEL=gemini-2.5-flash
```

For Streamlit Community Cloud, add the key under **App settings → Secrets** instead of committing it. Never commit `.env`, API keys, or other credentials. If Gemini is not configured or temporarily unavailable, the app displays the verified local guidance.

## Train and evaluate

The raw dataset is not included. Place the image folders under `data/raw/` using these exact folder names:

```text
data/raw/
├── bacterial_leaf_blight/
├── brown_spot/
├── Healthy/
├── leaf_blast/
├── leaf_scald/
├── narrow_brown_spot/
├── neck_blast/
├── rice_hispa/
├── sheath_blight/
└── tungro/
```

Run the notebooks in order:

1. `notebook/01_create_image_dataset.ipynb` — build the image manifest.
2. `notebook/02_validate_and_explore.ipynb` — validate images and remove exact duplicates.
3. `notebook/03_split_dataset.ipynb` — create stratified train, validation, and test splits.
4. `notebook/04_train_efficientnet.ipynb` — train and save the best validation checkpoint.
5. `notebook/05_model_evaluation.ipynb` — evaluate the checkpoint on the test split.
6. `notebook/06_predict_image_and_llm.ipynb` — run a sample prediction and explanation.
7. `notebook/07_create_client.ipynb` — optional standalone client workflow.

The training notebook starts a new training run when its training cell is executed. EfficientNet pretrained weights may be downloaded on the first run. To use the app, ensure the resulting checkpoint is present at `models/rice_disease_model.pth`.

## Deploy on Streamlit Community Cloud

1. Push the app code and `requirements.txt` to a GitHub repository.
2. Ensure the deployed app has `models/rice_disease_model.pth` at the path above. The checkpoint is Git-ignored, so deployment will not receive it unless you explicitly include it or arrange a download during startup.
3. Create a Community Cloud app from the repository and set the main file to `streamlit_app.py`.
4. Add `GEMINI_API_KEY` in the app’s **Secrets** settings if Gemini explanations are desired.

## Project structure

```text
streamlit_app.py                  Streamlit application
knowledge_base/                   Curated disease and pest guidance
models/                           Trained checkpoint and training history
notebook/                         Data, training, evaluation, and client workflow
requirements.txt                  Python dependencies
```

## Important note

This app is a screening aid. A model prediction is not a confirmed diagnosis, and an image classified as healthy may still have an unrecognized problem. Check the crop and consult a qualified agricultural officer before making treatment decisions.
