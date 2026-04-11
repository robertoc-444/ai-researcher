import streamlit as st
from google import genai
from google.genai import types
from google.oauth2 import service_account
from anthropic import AnthropicVertex
import json
import tempfile
import os
import time
from docx import Document

# ==========================================
# 1. PAGE SETUP & UI PROTECTION
# ==========================================
st.set_page_config(page_title="Multi-Agent Researcher", page_icon="🎓", layout="wide")

# Temporarily commented out to debug the "White Screen" issue
# st.markdown("""
#     <style>
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
#     .stApp [data-testid="stToolbar"] {display: none;}
#     </style>
#     """, unsafe_allow_html=True)

st.title("🎓 Professional AI Research Suite (v2.0)")

# ==========================================
# 2. SIDEBAR: ACCESS & MODEL CONFIG
# ==========================================
with st.sidebar:
    st.header("🔐 Access & Brains")
    access_key = st.text_input("Enter Access Key", type="password")
    
    if access_key != st.secrets["APP_PASSWORD"]:
        st.warning("Please enter the correct Access Key.")
        st.stop()

    st.divider()
    
    # 2026 Model Selection 
    gemini_options = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3.1-flash-lite"]
    claude_options = ["claude-3-5-sonnet@20240620", "claude-3-opus@20240229"]
    
    st.subheader("🤖 Agent Configuration")
    model_a_name = st.selectbox("Agent A (Researcher)", options=gemini_options, index=0, help="Gemini 2.5 Pro is recommended for deep research.")
    model_b_name = st.selectbox("Agent B (Critic)", options=claude_options + gemini_options, index=0, help="Claude is often a more rigorous peer reviewer.")

    st.divider()
    
    with st.expander("💡 View Prompting Guide", expanded=False):
        guide_data = [
            {"Aspect": "Specificity", "❌ Bad": "Tell me about renewable energy.", "✅ Excellent": "Analyze the 10-year LCOE between offshore wind and solar thermal."},
            {"Aspect": "Insurance", "❌ Bad": "How does disaster insurance work?", "✅ Excellent": "Analyze index-based triggers for tropical cyclones vs 'cat-in-a-box' modeling."},
            {"Aspect": "Files", "❌ Bad": "What's in these PDFs?", "✅ Excellent": "Synthesize risk factors in Section 4 and cross-reference with 2026 market trends."}
        ]
        st.table(guide_data)

    st.header("⚙️ Research Context")
    uploaded_files_ui = st.file_uploader("Upload Documents", accept_multiple_files=True, type=['pdf', 'docx', 'txt'])
    
    if st.button("🗑️ Clear Chat", type="primary"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 3. AUTHENTICATION & CLIENTS
# ==========================================
creds_info = st.secrets["GCP_SERVICE_ACCOUNT"]
credentials = service_account.Credentials.from_service_account_info(
    creds_info,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# Client A: Google GenAI (For Gemini)
client = genai.Client(
    vertexai=True, 
    project=st.secrets["GOOGLE_CLOUD_PROJECT"], 
    location=st.secrets["GOOGLE_CLOUD_LOCATION"],
    credentials=credentials
)

# Client B: Anthropic Vertex (For Claude)
anthropic_client = AnthropicVertex(
    project_id=st.secrets["GOOGLE_CLOUD_PROJECT"],
    region=st.secrets["GOOGLE_CLOUD_LOCATION"],
    credentials=credentials
)

# ==========================================
# 4. SYSTEM PROMPTS (ACADEMIC UPGRADE)
# ==========================================
agent_a_prompt = """Role: Post-Graduate Research Scientist & Synthesis Engine.
You operate at the level of platforms like Elicit, Consensus, and scite.ai. Your goal is to synthesize complex literature, map scientific consensus, and extract grounded data strictly from high-quality sources.

STRICT RULES:
1. ACADEMIC SOURCES ONLY: When using the Google Search tool, actively target scientific journals, academic books, government reports, and original source materials. Do not rely on generic blogs.
2. EXACT INLINE CITATIONS: EVERY empirical claim, statistic, or factual statement MUST include an exact inline citation (e.g., [Smith et al., 2023] or [DocumentName, p. 4]). 
3. NO HALLUCINATIONS: If the provided documents or search results do not contain the answer, explicitly state "Insufficient data in available sources." Do not guess.
4. CONFLICTING DATA: If sources disagree, explicitly highlight the contrast.

REQUIRED OUTPUT STRUCTURE:
[VERIFICATION LOG] List the specific search queries you ran and the databases/journals you targeted.
---
### 📊 Consensus Meter
*State in one sentence if the evidence shows: Strong Consensus, Emerging Consensus, Divided/Debated, or Insufficient Evidence.*

### 📑 Executive Synthesis
*A high-level, master's-level summary of the findings.*

### 🔬 Detailed Evidence & Extraction
*Deep dive into the data. Group by themes. Use heavy inline citations for every claim [Author, Year].*

### 📚 Reference List
*Provide EXACT citations (APA format preferred) for all sources. Include Authors, Year, Title, Journal/Book Name, and the exact URL or DOI.*"""

agent_b_prompt = """Role: Principal Investigator & Academic Peer Reviewer.
Your job is to relentlessly critique the Researcher's draft before it reaches the user. 

EVALUATION CRITERIA:
1. Source Quality: Are the sources cited high-quality (scientific journals, books, primary documents, .edu/.gov) rather than generic websites? (If no -> FAIL)
2. Exact Citations: Are there factual claims missing inline brackets [Author, Year]? (If yes -> FAIL)
3. Reference List Accuracy: Are the references formatted academically and do they include exact URLs, journals, or DOIs? (If no -> FAIL)
4. Grounding: Does the draft sound like it is guessing, or is it grounded in the cited literature? (If guessing -> FAIL)

If the draft fails any criteria, reject it with specific actionable feedback.
Output ONLY strict JSON: {"status": "PASS", "feedback": ""} or {"status": "FAIL", "feedback": "Specific reason and what to fix"}. 
DO NOT wrap the response in markdown blocks (```json). DO NOT include any conversational text. Just the raw JSON brackets."""

# ==========================================
# 5. THE MULTI-AGENT PIPELINE
# ==========================================
def get_docx_text(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def
