import streamlit as st
import tempfile, os

# --- Load Bedrock configuration BEFORE anything else ---
from src.config import load_config, apply_env_from_config

cfg = load_config("config.yaml")
apply_env_from_config(cfg)

# (Optional Debug Print)
print(f"✅ [DEBUG] Loaded LLM provider: {os.getenv('MODEL_PROVIDER', 'bedrock')}")
print(f"✅ [DEBUG] Bedrock model: {os.getenv('BEDROCK_MODEL_ID')}")

# --- Now import everything else ---
from src.config_loader import load_and_set_env
from src.agent_builder import (
    run_voice_to_text,
    run_clinical_understanding,
    run_rcm_coding,
    run_front_office,
    run_compliance_quality,
    run_document_task
)

# --- Streamlit UI setup ---
st.set_page_config(page_title="🏥 AI Medical Scribe", layout="wide")
st.title("🏥 AI Medical Scribe & RCM Assistant")
st.markdown("Upload a clinical audio file to run the full automation pipeline.")

uploaded_file = st.file_uploader("🎧 Upload clinical audio", type=["mp3", "wav", "m4a"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(audio_path)
    st.success("✅ File uploaded successfully!")

    state = {"audio_file": audio_path}

    # --- Step 1: Voice to Text ---
    with st.spinner("🎙️ Transcribing..."):
        state = run_voice_to_text(state)
    st.text_area("📝 Transcript", state.get("transcript", ""), height=200)

    # --- Step 2: Clinical Understanding ---
    with st.spinner("🧠 Extracting insights..."):
        state = run_clinical_understanding(state)
    st.json({
        "Diagnoses": state.get("diagnoses", []),
        "Symptoms": state.get("symptoms", []),
        "Procedures": state.get("procedures", []),
        "Medications": state.get("medications", [])
    })

    # --- Step 3: Coding ---
    with st.spinner("💰 Generating ICD/CPT codes..."):
        state = run_rcm_coding(state)
    st.json(state.get("coding_data", []))

    # --- Step 4: Front Office ---
    with st.spinner("🏢 Running front office automation..."):
        state = run_front_office(state)
    st.json(state.get("front_office", {}))

    # --- Step 5: Compliance & Quality ---
    with st.spinner("⚖️ Running compliance & quality audit..."):
        state = run_compliance_quality(state)
    st.json(state.get("compliance_quality", {}))

    # --- Step 6: Document Generation ---
    with st.spinner("📄 Generating clinical summary PDF..."):
        state = run_document_task(state)
    
    st.text_area("📄 Final Clinical Summary", state.get("document", ""), height=300)

    # --- PDF Download Link ---
    output_dir = "output"
    if os.path.exists(output_dir):
        pdf_files = [f for f in os.listdir(output_dir) if f.endswith(".pdf")]
        if pdf_files:
            latest_pdf = os.path.join(output_dir, sorted(pdf_files)[-1])
            with open(latest_pdf, "rb") as f:
                st.download_button("⬇️ Download PDF", f, file_name=os.path.basename(latest_pdf))
