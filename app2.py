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

# Temporarily commented out to prevent CSS-induced white screens
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
    
    # Properly indented: App stops here if the password doesn't match
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
# Load the service account info with explicit Cloud Platform scope
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
# 4. SYSTEM PROMPTS (ACADEMIC RIGOR)
# ==========================================
agent_a_prompt = """Role: Post-Graduate Research Scientist & Synthesis Engine.
You operate at the level of platforms like Elicit, Consensus, and scite.ai. Your goal is to synthesize complex literature, map scientific consensus, and extract grounded data strictly from high-quality sources.

STRICT RULES:
1. ACADEMIC SOURCES ONLY: When using Google Search, actively target scientific journals, academic books, government reports, and original source materials. Use search modifiers like 'site:edu' or 'DOI' internally.
2. EXACT INLINE CITATIONS: EVERY empirical claim, statistic, or factual statement MUST include an exact inline citation (e.g., [Smith et al., 2023] or [DocumentName, p. 4]).
3. NO HALLUCINATIONS: If sources do not contain the answer, state "Insufficient data in available sources." Do not guess.

REQUIRED OUTPUT STRUCTURE:
[VERIFICATION LOG] List specific search queries and databases targeted.
---
### 📊 Consensus Meter
*State the current academic consensus level.*

### 📑 Executive Synthesis
*A master's-level summary of findings.*

### 🔬 Detailed Evidence & Extraction
*Deep dive with heavy inline citations [Author, Year].*

### 📚 Reference List
*Provide EXACT APA citations including Authors, Year, Title, Journal/Book Name, and URL/DOI.*"""

agent_b_prompt = """Role: Principal Investigator & Academic Peer Reviewer.
Evaluate the draft based on:
1. Source Quality: Are they high-quality academic/primary sources?
2. Exact Citations: Is every factual claim supported by an inline [Author, Year]?
3. Reference List: Are exact URLs/DOIs included?
4. Grounding: Is the logic based on cited evidence?

Output ONLY strict JSON: {"status": "PASS", "feedback": ""} or {"status": "FAIL", "feedback": "reason"}. 
No markdown. No conversation."""

# ==========================================
# 5. THE MULTI-AGENT PIPELINE
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

def run_research_pipeline(user_input, chat_history, files, a_model, b_model):
    status_box = st.empty()
    status_box.info(f"Agents Initializing: {a_model} + {b_model}...")

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

    contents_a = []
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "model"
        contents_a.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    
    current_parts = process_files_to_parts(files) if files else []
    current_parts.append(types.Part.from_text(text=user_input))
    contents_a.append(types.Content(role="user", parts=current_parts))

    # Phase 1: Research (Gemini)
    response_a = client.models.generate_content(model=a_model, contents=contents_a, config=config_a)
    draft = response_a.text

    # Phase 2: Peer Review (Claude or Gemini)
    status_box.warning(f"Agent B ({b_model}) is verifying sources...")
    
    try:
        if "claude" in b_model.lower():
            message = anthropic_client.messages.create(
                model=b_model,
                max_tokens=1024,
                system=agent_b_prompt,
                messages=[{"role": "user", "content": f"User Query: {user_input}\n\nDraft:\n{draft}"}]
            )
            response_b_text = message.content[0].text
        else:
            response_b = client.models.generate_content(
                model=b_model, contents=f"User Query: {user_input}\n\nDraft:\n{draft}", config=config_b
            )
            response_b_text = response_b.text
        
        # Robust JSON cleaning
        start_idx = response_b_text.find('{')
        end_idx = response_b_text.rfind('}') + 1
        res = json.loads(response_b_text[start_idx:end_idx])
        
        if res['status'] == "PASS":
            status_box.success("Research Verified.")
            return f"✅ **Academic Peer Review Passed** (Verified by {b_model})\n\n---\n\n{draft}"
        else:
            status_box.error(f"Failed Review: {res['feedback']}")
            return f"🚨 **CRITIC FLAG:** {res['feedback']}\n\n---\n\n{draft}"
            
    except Exception as e:
        status_box.error("Format issue in Peer Review. Outputting draft.")
        return f"*(Peer review format error - Manual verification required)*\n\n{draft}"

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
        final_output = run_research_pipeline(prompt, st.session_state.messages, uploaded_files_ui, model_a_name, model_b_name)
        st.markdown(final_output)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": final_output})
