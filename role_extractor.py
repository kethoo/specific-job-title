import streamlit as st
import tempfile
import os
from cv_assessmentt import CVAssessmentSystem
import openai

# ------------------- STREAMLIT CONFIG -------------------

st.set_page_config(page_title="Role Extractor", layout="wide")
st.title("🎯 Role Section Extractor")

# ------------------- LOAD API KEY -------------------

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ Please set your OpenAI API key as an environment variable named 'OPENAI_API_KEY'.")
    st.stop()
else:
    st.success("🔐 API key loaded successfully.")

# ------------------- UPLOAD JOB DESCRIPTION -------------------

req_file = st.file_uploader("📄 Upload Tender / Job Description", type=["pdf", "docx", "doc"])
tender_text = ""

if req_file:
    suffix = os.path.splitext(req_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(req_file.read())
        tmp_path = tmp.name

    # Use your existing CVAssessmentSystem to extract text
    system = CVAssessmentSystem(api_key=api_key)
    tender_text = system.load_job_requirements(tmp_path)
    st.success(f"✅ File uploaded and text extracted from {req_file.name}")

# ------------------- ROLE INPUT -------------------

role_name = st.text_input("🧩 Enter Role Title", placeholder="e.g. Team Leader, Key Expert 1")

# ------------------- EXTRACTION FUNCTION -------------------

def extract_expert_section_llm(full_text: str, expert_name: str, api_key: str) -> str:
    """Extracts the section describing the specified role from the tender/job description."""
    if not full_text or not expert_name or not api_key:
        return ""
    client = openai.OpenAI(api_key=api_key)
    prompt = f"""Extract all text describing the role, qualifications, and responsibilities of "{expert_name}".
Return full sections with all details, preserving structure.
Separate multiple sections with "---SECTION BREAK---".
Document:
{full_text[:30000]}"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract exactly what's requested, no commentary."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=4000,
    )
    text = resp.choices[0].message.content.strip()
    return text.replace("---SECTION BREAK---", "\n\n" + "-"*60 + "\n\n") if text else ""

# ------------------- EXTRACTION LOGIC -------------------

if st.button("🚀 Extract Role Section", disabled=not (tender_text and role_name)):
    with st.spinner("Extracting role section..."):
        extracted_text = extract_expert_section_llm(tender_text, role_name, api_key)
        if extracted_text:
            st.success(f"✅ Extracted section for role: {role_name}")
            st.text_area("📄 Extracted Role Section:", value=extracted_text, height=500)
        else:
            st.warning("⚠️ No section found for that role. Try a different title or wording.")

# ------------------- FOOTER -------------------

st.markdown("---")
st.markdown("Made for internal use – extracts specific role descriptions from tender/job documents.")
