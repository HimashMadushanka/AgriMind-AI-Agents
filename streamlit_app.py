
from pathlib import Path
from io import BytesIO
import hashlib
import json
import os

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import efficientnet_b0
from dotenv import load_dotenv


# 1. CONFIGURATION
st.set_page_config(
    page_title="AgriMind | Rice leaf screening",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="auto",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "rice_disease_model.pth"
KNOWLEDGE_PATH = ROOT / "knowledge_base" / "disease_information.json"
HERO_IMAGE_PATH = ROOT / "assets" / "agrimind-rice-field.png"
MAX_IMAGE_PIXELS = 40_000_000

load_dotenv(ROOT / ".env")


# 2. DASHBOARD DESIGN
# 3. DISPLAY DISEASE NAME
def display_name(label):

    if label.lower() == "healthy":
        return "Healthy"

    return label.replace("_", " ").title()


# 4. LOAD TRAINED MODEL
@st.cache_resource
def load_trained_model(model_path, model_mtime_ns, model_size_bytes):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    try:
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(
            model_path,
            map_location=device
        )

    required = {
        "model_state_dict",
        "class_to_idx",
        "image_size",
        "mean",
        "std"
    }

    missing = required - set(checkpoint)

    if missing:
        raise ValueError(f"Missing checkpoint fields: {missing}")

    class_to_idx = checkpoint["class_to_idx"]

    model = efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        len(class_to_idx)
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    image_transform = transforms.Compose([
        transforms.Resize((
            checkpoint["image_size"],
            checkpoint["image_size"]
        )),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=checkpoint["mean"],
            std=checkpoint["std"]
        )
    ])

    idx_to_class = {
        int(index): name
        for name, index in class_to_idx.items()
    }

    return model, image_transform, idx_to_class, device


# 5. IMAGE CLASSIFICATION
def classify_image(
    image,
    model,
    image_transform,
    idx_to_class,
    device
):

    image_tensor = image_transform(
        image.convert("RGB")
    ).unsqueeze(0).to(device)

    with torch.inference_mode():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    top_count = min(3, len(idx_to_class))

    scores, indices = torch.topk(
        probabilities,
        k=top_count
    )

    results = []

    for score, index in zip(scores, indices):

        results.append({
            "label": idx_to_class[int(index)],
            "score": float(score)
        })

    return results


# 6. GEMINI EXPLANATION
def explain_with_gemini(
    disease,
    verified_information,
    source,
    question
):

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY")
        if secret_key:
            api_key = str(secret_key).strip()
    except Exception:
        pass

    model_name = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash"
    )

    if not api_key:
        return (
            "Gemini is not configured. "
            "Here is the verified agricultural guidance instead:\n\n"
            f"{verified_information}"
        )

    if not verified_information.strip():
        raise ValueError(
            "Verified disease information is missing."
        )

    from google import genai
    import httpx
    from google.genai import errors

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are an agricultural assistant for Sri Lankan rice farmers. Use everyday words, short sentences, and explain scientific terms simply.

Predicted disease: {display_name(disease)}

This is an image model prediction, not a confirmed diagnosis.

Use only the verified information below. Explain it in simple everyday words and short sentences.
Do not invent pesticide names, doses or treatments.

Verified agricultural information:
{verified_information}

Reference source:
{source or "Not specified"}

Farmer question:
{question or "Explain this disease in simple English."}

Give an answer using these headings:

1. Predicted Disease
2. Symptoms
3. Management Guidance
4. Prevention
5. Important Note
"""

    try:

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        return response.text

    except Exception as error:
        status_code = getattr(error, "status_code", None)
        error_text = str(error).upper()
        is_fallback_error = (
            status_code in (429, 503)
            or "429" in error_text
            or "503" in error_text
            or "RESOURCE_EXHAUSTED" in error_text
            or isinstance(error, httpx.TransportError)
        )
        if is_fallback_error:
            return (
                "The AI service is temporarily unavailable. "
                "Here is the verified agricultural guidance instead:\n\n"
                f"{verified_information}"
            )
        raise

    finally:

        close_fn = getattr(client, "close", None)

        if callable(close_fn):
            close_fn()


with st.sidebar:
    st.title("AgriMind", icon=":material/grass:")
    st.caption("Rice leaf screening")
    st.radio(
        "Workspace",
        ["Leaf screening", "About AgriMind"],
        captions=["Upload and analyze a leaf photo", "Model and evaluation details"],
        key="workspace_page",
        label_visibility="collapsed",
    )
    st.badge("10 disease and pest categories", icon=":material/category:", color="green")

if st.session_state.get("workspace_page") == "About AgriMind":
    st.title("About AgriMind", icon=":material/info:")
    st.caption("AI-assisted rice leaf screening for paddy farming.")

    overview_cols = st.columns(3, gap="medium", wrap=True)
    overview_cols[0].metric("Model", "EfficientNet-B0", border=True)
    overview_cols[1].metric("Categories", "10", border=True)
    overview_cols[2].metric("Test split", "1,007 images", border=True)

    with st.container(border=True):
        st.subheader("Evaluation")
        metric_cols = st.columns(2, gap="medium", wrap=True)
        metric_cols[0].metric("Accuracy", "98.41%")
        metric_cols[1].metric("Macro F1", "0.983")
        st.caption(
            "Results are from the project's held-out test split and may not "
            "reflect performance on field photos."
        )

    with st.expander("Supported categories"):
        st.write(
            "Healthy, bacterial leaf blight, brown spot, leaf blast, leaf scald, "
            "narrow brown spot, neck blast, rice hispa, sheath blight, and tungro."
        )

    st.info(
        "Use predictions as screening support. Confirm uncertain cases with a "
        "qualified agricultural officer.",
        icon=":material/health_and_safety:",
    )
    st.stop()


# 7. MAIN DASHBOARD
with st.container(key="agrimind_brand_title"):
    st.title("AgriMind", icon=":material/eco:")
    st.caption("AI-powered rice leaf screening")
st.html(
    """
    <style>
    .st-key-agrimind_brand_title h1 {
        color: var(--st-primary-color);
        font-size: 3.5rem;
        font-weight: 750;
        letter-spacing: -0.04em;
        line-height: 1.05;
    }
    @media (max-width: 640px) {
        .st-key-agrimind_brand_title h1 {
            font-size: 2.5rem;
        }
    }
    </style>
    """
)
if HERO_IMAGE_PATH.is_file():
    with st.container(key="agrimind_hero_banner"):
        st.image(
            str(HERO_IMAGE_PATH),
            alt="A drone surveying a green rice paddy at sunrise",
            width="stretch",
        )
    st.html(
        """
        <style>
        .st-key-agrimind_hero_banner img {
            height: 220px;
            width: 100%;
            object-fit: cover;
        }
        </style>
        """
    )


# 8. CHECK TRAINED MODEL
if not MODEL_PATH.exists():
    st.error("Trained model not found!", icon=":material/error:")
    st.info("Run Notebook 04_train_efficientnet.ipynb first.")
    st.stop()

try:
    model, image_transform, idx_to_class, device = load_trained_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime_ns,
        MODEL_PATH.stat().st_size,
    )
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

st.badge(
    f"Model ready \u00b7 {str(device).upper()}",
    icon=":material/check_circle:",
    color="green",
)


# 9. IMAGE UPLOAD AND PREDICTION
upload_col, result_col = st.columns(
    [1, 1.15],
    gap="large",
    wrap=True,
)

with upload_col:
    with st.container(border=True):
        st.subheader("Upload an image", icon=":material/upload:")
        st.caption("JPG \u00b7 PNG \u00b7 BMP \u00b7 WebP \u00b7 Max 20 MB")
        uploaded = st.file_uploader(
            "Choose a rice leaf photo",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            max_upload_size=20,
        )

        image = None
        if uploaded is not None:
            try:
                with Image.open(BytesIO(uploaded.getvalue())) as opened_image:
                    if opened_image.width * opened_image.height > MAX_IMAGE_PIXELS:
                        raise ValueError(
                            f"Image exceeds the {MAX_IMAGE_PIXELS // 1_000_000}-megapixel limit."
                        )
                    image = opened_image.convert("RGB")

                st.image(
                    image,
                    caption=uploaded.name,
                    width="stretch",
                    alt=f"Uploaded rice leaf image: {uploaded.name}",
                )
                st.caption(f"Image size: {image.width} x {image.height}")
            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
                Image.DecompressionBombError,
            ) as error:
                st.error(f"Invalid or unsupported image file: {error}")

        image_id = (
            hashlib.sha256(uploaded.getvalue()).hexdigest()
            if uploaded else None
        )
        if st.session_state.get("image_id") != image_id:
            st.session_state["image_id"] = image_id
            st.session_state.pop("prediction", None)
            st.session_state.pop("llm_answer", None)

        analyze = st.button(
            "Analyze image",
            icon=":material/search:",
            type="primary",
            width="stretch",
            disabled=image is None,
        )
        if analyze and image is not None:
            with st.spinner("Analyzing image..."):
                try:
                    st.session_state["prediction"] = classify_image(
                        image,
                        model,
                        image_transform,
                        idx_to_class,
                        device,
                    )
                    st.session_state.pop("llm_answer", None)
                except Exception as error:
                    st.error(f"Prediction failed: {error}")

with result_col:
    with st.container(border=True):
        st.subheader("Prediction", icon=":material/analytics:")
        prediction = st.session_state.get("prediction")

        if prediction:
            best = prediction[0]
            st.markdown(f"## {display_name(best['label'])}")
            st.metric(
                "Model score",
                f"{best['score'] * 100:.2f}%",
                help="This is not a calibrated probability or confirmed diagnosis.",
                border=True,
            )


            chart = pd.DataFrame(
                {
                    "Model score (%)": [item["score"] * 100 for item in prediction]
                },
                index=[display_name(item["label"]) for item in prediction],
            )
            st.markdown("#### Top predictions")
            st.bar_chart(
                chart,
                alt="Top three predicted rice leaf categories and model scores",
            )
        else:
            st.info(
                "Upload a photo to see the prediction.",
                icon=":material/info:",
            )


# 10. AGRICULTURAL GUIDANCE
if st.session_state.get("prediction"):
    st.subheader("Guidance for this result", icon=":material/volunteer_activism:")
    predicted_label = st.session_state["prediction"][0]["label"]
    knowledge = {}

    if KNOWLEDGE_PATH.exists():
        try:
            knowledge = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            st.warning(f"Knowledge base error: {error}")

    entry = knowledge.get(predicted_label, {})
    verified_information = str(entry.get("verified_information", "")).strip()
    source = str(entry.get("source", ""))

    if not verified_information:
        st.warning(
            "Add verified agricultural information to "
            "disease_information.json before generating advice."
        )

    with st.container(border=True):
        st.caption("Your image is not sent to Gemini.")

        with st.form("farmer_question_form"):
            question = st.text_area(
                "Ask about this result",
                placeholder="How can I manage this?",
                height=100,
            )
            generate_explanation = st.form_submit_button(
                "Get guidance",
                icon=":material/auto_awesome:",
                type="primary",
                width="stretch",
                disabled=not bool(verified_information),
            )

        if generate_explanation:
            with st.spinner("Preparing guidance..."):
                try:
                    st.session_state["llm_answer"] = explain_with_gemini(
                        predicted_label,
                        verified_information,
                        source,
                        question,
                    )
                except Exception as error:
                    st.error(str(error))

        if st.session_state.get("llm_answer"):
            st.markdown(st.session_state["llm_answer"])
            if source:
                st.caption(f"Reference: {source}")

            source_url = str(entry.get("source_url", "")).strip()
            if source_url:
                st.link_button(
                    "Open agricultural source",
                    source_url,
                    icon=":material/open_in_new:",
                )


# 11. FOOTER
st.space("small")
st.caption("Decision support only \u00b7 Confirm uncertain cases with an agriculture officer.")
