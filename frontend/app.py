from pathlib import Path
import sys
import streamlit as st
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from agents.orchestrator import handle_farmer_question

st.set_page_config(page_title="Paddy Farming Intelligence", page_icon="🌾", layout="centered")
st.title("🌾 Paddy Farming Intelligence")
st.caption("Prototype only — verify advice with an agricultural officer.")
question = st.text_area("Describe your paddy-farming problem / ඔබගේ වී වගාවේ ගැටලුව විස්තර කරන්න", height=140, placeholder="Type in Sinhala or English...")
with st.expander("Optional: provide location for contextual weather data"):
    use_location = st.checkbox("I choose to provide coordinates")
    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=7.0, format="%.5f", disabled=not use_location)
    lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=80.0, format="%.5f", disabled=not use_location)
if st.button("Analyze question", type="primary"):
    if len(question.strip()) < 3:
        st.warning("Please enter a little more detail.")
    else:
        with st.spinner("Reviewing the question..."):
            result = handle_farmer_question(question, lat if use_location else None, lon if use_location else None)
        st.subheader("Guidance")
        st.write(result["advisory"])
        with st.expander("Technical results and sources"):
            st.json(result)
