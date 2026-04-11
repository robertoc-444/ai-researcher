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

# Commented out for safety during initial load
# st.markdown("""<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}</style>""", unsafe_allow_html=True)

st.title("🎓 Professional AI Research Suite (v2.1)")

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
    model_a_name = st.selectbox("Agent A (Researcher)", options=gemini_options, index=0)
    model_b_name = st.selectbox("Agent B (Critic)", options=claude_options + gemini_options, index=0)

    st.divider()
    st.header("🎛️ Agent Parameters")

    # 1. Critic Grading Weights
    st.subheader("Critic Evaluation Weights (1-10)")
    st.caption("Adjust how strictly Agent B evaluates the draft.")
    weight_source = st.slider("Source Quality", min_value=1, max_value=10, value=8)
    weight_citation = st.slider("Citation Accuracy", min_value=1, max_value=10, value=10)
    weight_format = st.slider("Formatting/Structure", min_value=1, max_value=10, value=5)

    # 2. Pipeline Controls
    st.subheader("Pipeline Controls")
    max_attempts_ui = st.number_input("Max Revision Attempts", min_value=1, max_value=5, value=2)
    
    # 3. Model Generation Parameters
    st.subheader("Model Parameters")
    temp_a = st.slider("Agent A Temperature (Creativity)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    st.divider()
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

client = genai.Client(
    vertexai=True, 
    project=st.secrets["GOOGLE_CLOUD_PROJECT"], 
    location=st.secrets["GOOGLE_CLOUD_LOCATION"],
    credentials=credentials
)

anthropic_client = AnthropicVertex(
    project_id=st.secrets["GOOGLE_CLOUD_PROJECT"],
    region=st.secrets["GOOGLE_CLOUD_LOCATION"],
    credentials=credentials
)

# ==========================================
# 4. SYSTEM PROMPTS (MASTER'S LEVEL)
# ==========================================
# Current Date Note included to prevent "Future Hallucinations"
agent_a_prompt = f"""Role: Post-Graduate Research Scientist.
Current Date: April 2026.
Synthesize high-quality literature and map scientific consensus.
STRICT RULES:
1. ACADEMIC SOURCES ONLY: Target journals, books, and gov reports. No generic blogs.
2. EXACT INLINE CITATIONS: Every claim MUST have an inline citation [Author, Year].
3. NO HALLUCINATIONS: Do not invent DOIs or citations. If a source isn't found, say so.
4. 2026 CONTEXT: Only cite papers published up to the current date.

REQUIRED OUTPUT STRUCTURE:
[VERIFICATION LOG] Search queries used.
---
### 📊 Consensus Meter
### 📑 Executive Synthesis
### 🔬 Detailed Evidence & Extraction (Grouped by theme, heavy [Author, Year] citations)
### 📚 Reference List (APA format with exact URLs/DOIs)"""

def get_agent_b_prompt(w_source, w_citation, w_format):
    return f"""Role: Principal Investigator.
Evaluate the draft based on the following weighted criteria (1-10 scale):
- Source Quality (Weight: {w_source}/10)
- Exact Inline Citations [Author, Year] (Weight: {w_citation}/10)
- Formatting and Reference List Accuracy (Weight: {w_format}/10)

Output ONLY strict JSON: {{"status": "PASS", "feedback": ""}} or {{"status": "FAIL", "feedback": "reason"}}.
If you use quotes in feedback, use single quotes ('). No conversational text.
No markdown."""

# ==========================================
# 5. THE PIPELINE & SELF-CORRECTION
# ==========================================
def get_docx_text(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def process_files_to_parts(files):
    parts = []
    for f in files:
        if f.name.endswith('.docx'):
            parts.append(types.Part.from_text(text=f"Content from {f.name}:\n{get_docx_text(f)}"))
        elif f.name.endswith('.txt'):
            parts.append(types.Part.from_text(text=f"Content from {f.name}:\n{f.read().decode('utf-8')}"))
        elif f.name.endswith('.pdf'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.getbuffer())
                tmp_path = tmp.name
            g_file = client.files.upload(file=tmp_path, config={'display_name': f.name})
            while g_file.state.name == "PROCESSING": time.sleep(1)
            parts.append(types.Part.from_uri(file_uri=g_file.uri, mime_type="application/pdf"))
    return parts

def run_research_pipeline(user_input, chat_history, files, a_model, b_model, max_attempts, temp_a, weights):
    status_box = st.empty()
    current_attempt = 1
    feedback_loop = ""

    while current_attempt <= max_attempts:
        status_box.info(f"Attempt {current_attempt}: {a_model} researching...")

        # Build Content A
        contents_a = []
        for msg in chat_history:
            role = "user" if msg["role"] == "user" else "model"
            contents_a.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
        
        current_parts = process_files_to_parts(files) if files else []
        if feedback_loop:
            instruction = f"REVISION REQUIRED. Previous draft failed: {feedback_loop}\nFix and ensure exact [Author, Year] citations."
            current_parts.append(types.Part.from_text(text=f"{instruction}\n\nQuery: {user_input}"))
        else:
            current_parts.append(types.Part.from_text(text=user_input))
        contents_a.append(types.Content(role="user", parts=current_parts))

        # Phase 1: Research
        config_a = types.GenerateContentConfig(
            system_instruction=agent_a_prompt, 
            temperature=temp_a, 
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
        response_a = client.models.generate_content(model=a_model, contents=contents_a, config=config_a)
        draft = response_a.text

        # Phase 2: Review
        status_box.warning(f"Attempt {current_attempt}: {b_model} verifying integrity...")
        dynamic_b_prompt = get_agent_b_prompt(*weights)
        
        try:
            if "claude" in b_model.lower():
                message = anthropic_client.messages.create(
                    model=b_model, 
                    max_tokens=1024, 
                    system=dynamic_b_prompt, 
                    messages=[{"role": "user", "content": f"Query: {user_input}\n\nDraft:\n{draft}"}]
                )
                response_b_text = message.content[0].text
            else:
                config_b = types.GenerateContentConfig(
                    system_instruction=dynamic_b_prompt, 
                    temperature=0.0, 
                    response_mime_type="application/json"
                )
                response_b = client.models.generate_content(
                    model=b_model, 
                    contents=f"Query: {user_input}\n\nDraft:\n{draft}", 
                    config=config_b
                )
                response_b_text = response_b.text
            
            # The Indestructible Parser
            text = response_b_text.strip()
            start = text.find('{')
            end = text.rfind('}') + 1
            res = json.loads(text[start:end])
            
            if res.get('status') == "PASS":
                status_box.success(f"Verified Pass on Attempt {current_attempt}!")
                return f"✅ **Verified Pass (Attempt {current_attempt})**\n\n---\n\n{draft}"
            else:
                feedback_loop = res.get('feedback', 'General failure')
                if current_attempt < max_attempts:
                    status_box.error(f"Attempt {current_attempt} rejected. Correcting...")
                    current_attempt += 1
                    time.sleep(2)
                else:
                    return f"🚨 **FINAL CRITIC FLAG:** {feedback_loop}\n\n---\n\n{draft}"
        except Exception:
            return f"*(Review format error - Outputting raw draft)*\n\n{draft}"

# ==========================================
# 6. CHAT INTERFACE
# ==========================================
if "messages" not in st.session_state: st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your research query..."):
    st.chat_message("user").markdown(prompt)
    with st.chat_message("assistant"):
        weights = (weight_source, weight_citation, weight_format)
        final_output = run_research_pipeline(
            prompt, 
            st.session_state.messages, 
            uploaded_files_ui, 
            model_a_name, 
            model_b_name,
            max_attempts_ui,
            temp_a,
            weights
        )
        st.markdown(final_output)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": final_output})

# ==========================================
# 7. FOOTER
# ==========================================
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.9em; padding-bottom: 20px;'>"
    "Suggestions for improvement? Contact robert.cater@gmail.com - 2026"
    "</div>", 
    unsafe_allow_html=True
)
