import streamlit as st
from google import genai
from google.genai import types
import json
import tempfile
import os
import time
from docx import Document # Requirement for Word doc processing

# ==========================================
# 1. PAGE SETUP & UI PROTECTION
# ==========================================
st.set_page_config(page_title="Multi-Agent Researcher", page_icon="🎓", layout="wide")

# CSS to hide the "View Source", "Fork", and Streamlit menu
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp [data-testid="stToolbar"] {display: none;}
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 AI Researcher (cater beta1.2)")

# ==========================================
# 2. ACCESS CONTROL
# ==========================================
with st.sidebar:
    st.header("🔐 Access Control")
    access_key = st.text_input("Enter Access Key", type="password")
    if access_key != st.secrets["APP_PASSWORD"]:
        st.warning("Please enter the correct Access Key.")
        st.stop()

# ==========================================
# 3. SESSION STATE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 4. SIDEBAR & FILE UPLOAD
# ==========================================
with st.sidebar:
    st.divider()
    st.header("⚙️ Research Context")
# --- PROMPT GUIDE INSERTION ---
    with st.expander("💡 View Prompting Guide", expanded=False):
        st.info("Agent A requires Master's level specificity. Use the examples below to avoid 'Low Confidence' errors.")
        
        guide_data = [
            {"Aspect": "Specificity", "❌ Bad": "Tell me about renewable energy.", "✅ Excellent": "Analyze the comparative efficiency and 10-year LCOE between offshore wind and molten salt solar thermal systems."},
            {"Aspect": "Files", "❌ Bad": "What do these PDFs say?", "✅ Excellent": "Based on the uploaded quarterly reports, synthesize primary risk factors in Section 4 and cross-reference with 2026 market trends."},
            {"Aspect": "Formatting", "❌ Bad": "Write a quick summary.", "✅ Excellent": "Provide a technical meta-analysis. Structure with an executive summary and a deep-dive into the study methodology."},
            {"Aspect": "Parametric", "❌ Bad": "How does parametric insurance work?", "✅ Excellent": "Analyze index-based triggers for tropical cyclones. Compare 'cat-in-a-box' vs 'distance-to-site' modeling."},
            {"Aspect": "Math", "❌ Bad": "Explain the math behind triggers.", "✅ Excellent": "Provide a technical model for calculating drought insurance trigger thresholds based on NDVI."}
        ]
        st.table(guide_data)
    st.divider()
    # --- END OF GUIDE ---
    uploaded_files_ui = st.file_uploader("Upload Documents", accept_multiple_files=True, type=['pdf', 'docx', 'txt'])
    
    if st.button("🗑️ Clear Chat / Start Over", type="primary"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 5. AUTHENTICATION
# ==========================================
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# ==========================================
# 6. SYSTEM PROMPTS (Restored High Rigor)
# ==========================================
agent_a_prompt = """
Role: You are a highly rigorous, technical research assistant. Your primary directive is absolute factual accuracy. All research must be delivered at the academic depth of a Master's degree graduate.

Core Directives:
1. Handling Uncertainty: If you lack high confidence, state ONLY: "I cannot provide a reliable answer." 
2. Academic Consensus: If a topic lacks consensus, map out the leading competing theories citing foundational sources.
3. Estimations: Never guess. Preface with: "Warning: The following is an estimation."
4. Blended Sourcing: Synthesize Google Search and User Documents into a single cohesive response.

Mandatory Verification Protocol:
Generate a [VERIFICATION LOG] block at the top containing:
* Source Check (Web, Docs, or Both)
* Quote/Canonical Source Extraction
* Consensus Check
* Confidence Check

Output a horizontal line (---), then the final response.
"""

agent_b_prompt = """
You are a ruthless peer reviewer. Review the draft for hallucinations or unsupported claims.
Output ONLY strict JSON: {"status": "PASS", "feedback": ""} OR {"status": "FAIL", "feedback": "reason"}
"""

# ==========================================
# 7. THE MULTI-AGENT PIPELINE
# ==========================================

def get_docx_text(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def process_files_to_parts(files):
    """Converts mixed file types into Gemini-compatible Parts."""
    parts = []
    for f in files:
        if f.name.endswith('.docx'):
            text = get_docx_text(f)
            parts.append(types.Part.from_text(text=f"Content from {f.name}:\n{text}"))
        elif f.name.endswith('.txt'):
            text = f.read().decode("utf-8")
            parts.append(types.Part.from_text(text=f"Content from {f.name}:\n{text}"))
        elif f.name.endswith('.pdf'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.getbuffer())
                tmp_path = tmp.name
            try:
                g_file = client.files.upload(file=tmp_path, config={'display_name': f.name})
                while g_file.state.name == "PROCESSING":
                    time.sleep(2)
                    g_file = client.files.get(name=g_file.name)
                parts.append(types.Part.from_uri(file_uri=g_file.uri, mime_type="application/pdf"))
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)
    return parts

def run_research_pipeline(user_input, chat_history, files):
    status_box = st.empty()
    status_box.info("Researcher initialized...")

    MODEL_NAME = 'gemini-3-flash-preview'

    config_a = types.GenerateContentConfig(
        system_instruction=agent_a_prompt, 
        temperature=0.0,
        tools=[types.Tool(google_search=types.GoogleSearch())]
    )
    
    config_b = types.GenerateContentConfig(
        system_instruction=agent_b_prompt, 
        temperature=0.0, 
        response_mime_type="application/json"
    )
    
    # 1. Build Conversation History
    contents_a = []
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "model"
        contents_a.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    
    # 2. Build current interaction parts (Mixed Text/Files)
    current_parts = process_files_to_parts(files) if files else []
    current_parts.append(types.Part.from_text(text=user_input))
    contents_a.append(types.Content(role="user", parts=current_parts))

    # 3. Agent A Draft
    status_box.info("Agent A is synthesizing web search and document data...")
    response_a = client.models.generate_content(model=MODEL_NAME, contents=contents_a, config=config_a)
    draft_response = response_a.text
    
    # 4. Agent B loop
    attempt = 0
    while attempt < 2:
        status_box.warning(f"Agent B is conducting Peer Review (Attempt {attempt+1})...")
        response_b = client.models.generate_content(
            model=MODEL_NAME, 
            contents=f"Review this for the query '{user_input}':\n\n{draft_response}", 
            config=config_b
        )
        
        try:
            res = json.loads(response_b.text)
            if res['status'] == "PASS":
                status_box.success("Final Response Approved.")
                return draft_response
            else:
                status_box.error(f"Critic Failed: {res['feedback']}")
                rev_prompt = f"Peer review failed: {res['feedback']}. Revise the response now."
                contents_a.append(types.Content(role="model", parts=[types.Part.from_text(text=draft_response)]))
                contents_a.append(types.Content(role="user", parts=[types.Part.from_text(text=rev_prompt)]))
                response_a = client.models.generate_content(model=MODEL_NAME, contents=contents_a, config=config_a)
                draft_response = response_a.text
        except:
            return draft_response # Fallback to draft if logic fails
        attempt += 1
    return draft_response

# ==========================================
# 8. CHAT INTERFACE
# ==========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter research query..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        final_output = run_research_pipeline(prompt, st.session_state.messages, uploaded_files_ui)
        st.markdown(final_output)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": final_output})
