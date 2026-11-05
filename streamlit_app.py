
from pathlib import Path
from io import BytesIO
import hashlib
import importlib
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
import localization

localization = importlib.reload(localization)
LANGUAGE_PROMPT_NAMES = localization.LANGUAGE_PROMPT_NAMES
LANGUAGES = localization.LANGUAGES
localized_disease_name = localization.localized_disease_name
tr = localization.tr


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
# Preliminary review thresholds; validate these with expert-labeled field images.
MIN_PREDICTION_SCORE = 0.65
MIN_TOP_TWO_MARGIN = 0.10

load_dotenv(ROOT / ".env")


# 2. DASHBOARD DESIGN
# 3. DISPLAY DISEASE NAME
def display_name(label, language):
    return localized_disease_name(label, language)


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


def prediction_needs_review(prediction):
    """Flag weak or ambiguous scores; these are not calibrated probabilities."""
    if not prediction:
        return False

    top_score = prediction[0]["score"]
    second_score = prediction[1]["score"] if len(prediction) > 1 else 0.0
    return (
        top_score < MIN_PREDICTION_SCORE
        or top_score - second_score < MIN_TOP_TWO_MARGIN
    )


# 6. GEMINI EXPLANATION
def explain_with_gemini(
    disease,
    verified_information,
    source,
    question,
    language,
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
            f"{tr(language, 'fallback_unconfigured')}\n\n"
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

Respond entirely in {LANGUAGE_PROMPT_NAMES.get(language, "English")}.

Predicted disease: {display_name(disease, language)}

This is an image model prediction, not a confirmed diagnosis.

Use only the verified information below. Explain it in simple everyday words and short sentences.
Do not invent pesticide names, doses or treatments.

Verified agricultural information:
{verified_information}

Reference source:
{source or "Not specified"}

Farmer question:
{question or "Explain the symptoms and management guidance."}

Use short, clear headings in the requested language. Preserve the meaning of the verified information.
Do not add new facts, chemical names, doses, or treatment instructions.
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
                f"{tr(language, 'fallback_unavailable')}\n\n"
                f"{verified_information}"
            )
        raise

    finally:

        close_fn = getattr(client, "close", None)

        if callable(close_fn):
            close_fn()


previous_language = st.session_state.get("language_code", "en")
language = previous_language

with st.sidebar:
    st.title("AgriMind", icon=":material/grass:")
    st.caption(tr(language, "rice_leaf_screening"))
    st.subheader(tr(language, "navigation"))
    st.radio(
        tr(language, "navigation"),
        ["screening", "about"],
        format_func=lambda page: tr(
            language, "nav_screening" if page == "screening" else "nav_about"
        ),
        captions=[
            tr(language, "leaf_screening_caption"),
            tr(language, "about_caption"),
        ],
        key="workspace_page",
        label_visibility="collapsed",
    )
    st.space("small")
    language = st.selectbox(
        tr(language, "language"),
        options=list(LANGUAGES),
        format_func=lambda code: LANGUAGES[code],
        key="language_code",
    )
    if language != previous_language:
        st.session_state.pop("llm_answer", None)
    st.badge(tr(language, "category_count"), icon=":material/category:", color="green")

if st.session_state.get("workspace_page", "screening") == "about":
    st.title(tr(language, "about_app"), icon=":material/info:")
    st.caption(tr(language, "about_description"))

    overview_cols = st.columns(3, gap="medium", wrap=True)
    overview_cols[0].metric(tr(language, "model"), "EfficientNet-B0", border=True)
    overview_cols[1].metric(tr(language, "categories"), "10", border=True)
    overview_cols[2].metric(
        tr(language, "test_split"), tr(language, "test_images"), border=True
    )

    with st.container(border=True):
        st.subheader(tr(language, "evaluation"))
        metric_cols = st.columns(2, gap="medium", wrap=True)
        metric_cols[0].metric(tr(language, "accuracy"), "98.41%")
        metric_cols[1].metric(tr(language, "macro_f1"), "0.983")
        st.caption(tr(language, "held_out_note"))

    with st.expander(tr(language, "supported_categories")):
        st.write(
            ", ".join(
                localized_disease_name(label, language)
                for label in (
                    "Healthy", "bacterial_leaf_blight", "brown_spot", "leaf_blast",
                    "leaf_scald", "narrow_brown_spot", "neck_blast", "rice_hispa",
                    "sheath_blight", "tungro",
                )
            )
        )

    st.info(tr(language, "screening_warning"), icon=":material/health_and_safety:")
    st.stop()


# 7. MAIN DASHBOARD
with st.container(key="agrimind_brand_title"):
    st.title("AgriMind", icon=":material/eco:")
    st.caption(tr(language, "tagline"))
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
            alt=tr(language, "hero_alt"),
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
    st.error(tr(language, "model_missing"), icon=":material/error:")
    st.info(tr(language, "train_notebook"))
    st.stop()

try:
    model, image_transform, idx_to_class, device = load_trained_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime_ns,
        MODEL_PATH.stat().st_size,
    )
except Exception as error:
    st.error(tr(language, "model_load_failed", error=error))
    st.stop()

st.badge(
    tr(language, "model_ready", device=str(device).upper()),
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
        st.subheader(tr(language, "upload"), icon=":material/upload:")
        st.caption(tr(language, "file_types"))
        uploaded = st.file_uploader(
            tr(language, "upload_label"),
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            max_upload_size=20,
        )

        image = None
        if uploaded is not None:
            try:
                with Image.open(BytesIO(uploaded.getvalue())) as opened_image:
                    if opened_image.width * opened_image.height > MAX_IMAGE_PIXELS:
                        raise ValueError(
                            tr(
                                language,
                                "image_too_large",
                                max=MAX_IMAGE_PIXELS // 1_000_000,
                            )
                        )
                    image = opened_image.convert("RGB")

                st.image(
                    image,
                    caption=uploaded.name,
                    width="stretch",
                    alt=f"Uploaded rice leaf image: {uploaded.name}",
                )
                st.caption(
                    tr(language, "image_size", width=image.width, height=image.height)
                )
            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
                Image.DecompressionBombError,
            ) as error:
                st.error(tr(language, "invalid_image", error=error))

        image_id = (
            hashlib.sha256(uploaded.getvalue()).hexdigest()
            if uploaded else None
        )
        if st.session_state.get("image_id") != image_id:
            st.session_state["image_id"] = image_id
            st.session_state.pop("prediction", None)
            st.session_state.pop("llm_answer", None)
            st.session_state["confirm_rice_leaf"] = False

        rice_leaf_confirmed = st.checkbox(
            tr(language, "confirm_leaf"),
            key="confirm_rice_leaf",
            disabled=image is None,
        )
        st.caption(tr(language, "image_type_limit"))

        analyze = st.button(
            tr(language, "analyze"),
            icon=":material/search:",
            type="primary",
            width="stretch",
            disabled=image is None or not rice_leaf_confirmed,
        )
        if analyze and image is not None:
            with st.spinner(tr(language, "analyzing")):
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
                    st.error(tr(language, "prediction_failed", error=error))

with result_col:
    with st.container(border=True):
        st.subheader(tr(language, "prediction"), icon=":material/analytics:")
        prediction = st.session_state.get("prediction")

        if prediction:
            best = prediction[0]
            needs_review = prediction_needs_review(prediction)
            if needs_review:
                st.error(
                    tr(language, "invalid_leaf_image"),
                    icon=":material/image_not_supported:",
                )
                st.caption(tr(language, "uncertainty_note"))
            else:
                st.markdown(f"## {display_name(best['label'], language)}")
                st.metric(
                    tr(language, "model_score"),
                    f"{best['score'] * 100:.2f}%",
                    help=tr(language, "score_help"),
                    border=True,
                )

                chart = pd.DataFrame(
                    {
                        tr(language, "chart_score"): [
                            item["score"] * 100 for item in prediction
                        ]
                    },
                    index=[
                        display_name(item["label"], language) for item in prediction
                    ],
                )
                st.markdown(f"#### {tr(language, 'top_predictions')}")
                st.bar_chart(
                    chart,
                    alt=tr(language, "top_predictions"),
                )
        else:
            st.info(
                tr(language, "upload_hint"),
                icon=":material/info:",
            )


# 10. AGRICULTURAL GUIDANCE
if st.session_state.get("prediction"):
    prediction = st.session_state["prediction"]
    if prediction_needs_review(prediction):
        st.warning(tr(language, "uncertain_guidance"), icon=":material/health_and_safety:")
    else:
        st.subheader(tr(language, "guidance"), icon=":material/volunteer_activism:")
        predicted_label = prediction[0]["label"]
        knowledge = {}

        if KNOWLEDGE_PATH.exists():
            try:
                knowledge = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                st.warning(tr(language, "knowledge_error", error=error))

        entry = knowledge.get(predicted_label, {})
        verified_information = str(entry.get("verified_information", "")).strip()
        source = str(entry.get("source", ""))

        if not verified_information:
            st.warning(tr(language, "guidance_missing"))

        with st.container(border=True):
            st.caption(tr(language, "image_privacy"))

            with st.form("farmer_question_form"):
                question = st.text_area(
                    tr(language, "ask_question"),
                    placeholder=tr(language, "question_placeholder"),
                    height=100,
                )
                generate_explanation = st.form_submit_button(
                    tr(language, "get_guidance"),
                    icon=":material/auto_awesome:",
                    type="primary",
                    width="stretch",
                    disabled=not bool(verified_information),
                )

            if generate_explanation:
                with st.spinner(tr(language, "preparing_guidance")):
                    try:
                        st.session_state["llm_answer"] = explain_with_gemini(
                            predicted_label,
                            verified_information,
                            source,
                            question,
                            language,
                        )
                    except Exception as error:
                        st.error(tr(language, "ai_error", error=error))

            if st.session_state.get("llm_answer"):
                st.markdown(st.session_state["llm_answer"])
                if source:
                    st.caption(tr(language, "reference", source=source))

                source_url = str(entry.get("source_url", "")).strip()
                if source_url:
                    st.link_button(
                        tr(language, "open_source"),
                        source_url,
                        icon=":material/open_in_new:",
                    )

st.caption(tr(language, "footer"))

