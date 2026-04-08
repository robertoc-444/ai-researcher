import streamlit as st
from google import genai
from google.genai import types
import json
import tempfile
import os
import time

# ==========================================
# 1. PAGE SETUP & UI CONFIG
# ==========================================
st.set_page_config(page_title="Multi-Agent Researcher", page_icon="🎓", layout="wide")

# CSS to hide the "View Source" / GitHub toolbar and hamburger menu
hide_ui_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stApp [data-testid="stToolbar"] {display: none;}
            </style>
            """
st.markdown(hide_ui_style, unsafe_allow_html=True)

st.title("🎓 Master's-Level AI Researcher")

# ==========================================
# 2. ACCESS CONTROL (Sidebar Password)
# ==========================================
with st.sidebar:
    st.header("🔐 Access Control")
    access_key = st.text_input("Enter Access Key", type="password")
    
    # If the password doesn't match the one in Streamlit Secrets, stop execution
    if access_key != st.secrets["APP_PASSWORD"]:
        st.warning("Please enter the correct Access Key to enable the researcher.")
        st.stop()

# ==========================================
# 3. SESSION STATE (The "Memory")
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 4. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.divider()
    st.header("⚙️ Research Context")
    st.markdown("Upload PDFs to include them in the research analysis.")
    
    uploaded_files_ui = st.file_uploader("Upload Documents", accept_multiple_files=True)
    
    st.divider()
    
    if st.button("🗑️ Clear Chat / Start Over", type="primary"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 5. AUTHENTICATION & API SETUP
# ==========================================
API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

# ==========================================
# 6. SYSTEM PROMPTS
# ==========================================
agent_a_prompt = """
Role: You are a highly rigorous, technical research assistant. Your primary directive is absolute factual accuracy. All research must be delivered at the academic and technical depth of a Master's degree graduate.

Core Directives:
1. Handling Uncertainty: If you lack high confidence, state ONLY: "I cannot provide a reliable answer."
2. Academic Consensus: If a topic lacks consensus, map out the leading theories citing foundational sources.
3. Estimations: Never guess. Preface with: "Warning: The following is an estimation."
4. Blended Sourcing: Use Google Search and User Documents equally. Synthesize facts.

Mandatory Verification Protocol:
Generate a [VERIFICATION LOG] at the top:
* Source Check (Web, Docs, or Both)
* Quote Extraction
* Consensus Check
* Confidence Check
---
Final response follows.
"""

agent_b_prompt = """
You are a ruthless peer reviewer. Review the draft for hallucinations or unsupported claims.
Output ONLY strict JSON: {"status": "PASS", "feedback": ""} OR {"status": "FAIL", "feedback": "reason"}
"""

# ==========================================
# 7. THE MULTI-AGENT PIPELINE
# ==========================================

def process_files(files):
    """Saves files to temp and uploads to Gemini."""
    gemini_files = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getbuffer())
            tmp_path = tmp.name
        
        success = False
        for attempt in range(3):
            try:
                g_file = client.files.upload(
                    file=tmp_path, 
                    config={'display_name': f.name}
                )
                gemini_files.append(g_file)
                success = True
                break 
            except Exception:
                time.sleep(2)
        
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    return gemini_files

def run_research_pipeline(user_input, chat_history, files):
    status_box = st.empty()
    status_box.info("Initializing multi-agent analysis...")

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
    
    # 1. Build History
    contents_a = []
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "model"
        contents_a.append({"role": role, "parts": [{"text": msg["content"]}]})
    
    # 2. Process Files using Part.from_uri to avoid 400 Errors
    gemini_files = process_files(files) if files else []
    
    file_parts = [
        types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type) 
        for f in gemini_files
    ]
    
    current_parts = file_parts + [types.Part.from_text(text=user_input)]
    contents_a.append({"role": "user", "parts": current_parts})

    # 3. Agent A Phase
    status_box.info("Agent A is synthesizing data...")
    response_a = client.models.generate_content(model='gemini-2.5-flash', contents=contents_a, config=config_a)
    draft_response = response_a.text
    
    # 4. Agent B Phase (Review Loop)
    attempt = 0
    max_retries = 2
    while attempt < max_retries:
        status_box.warning(f"Peer Review (Attempt {attempt + 1})...")
        
        review_prompt = f"User Query: {user_input}\n\nDraft:\n{draft_response}"
        response_b = client.models.generate_content(model='gemini-2.5-flash', contents=review_prompt, config=config_b)
        
        try:
            critique_data = json.loads(response_b.text)
            status = critique_data.get("status")
            feedback = critique_data.get("feedback")
        except:
            status = "FAIL"
            feedback = "Peer review output error."
            
        if status == "PASS":
            status_box.success("Approved by Critic.")
            return draft_response
        else:
            status_box.error(f"Review Failed: {feedback}")
            st.toast("Agent A is revising...")
            
            revision_prompt = f"Fix these issues: {feedback}"
            revision_contents = contents_a + [
                {"role": "model", "parts": [{"text": draft_response}]}, 
                {"role": "user", "parts": [{"text": revision_prompt}]}
            ]
            
            response_a_revised = client.models.generate_content(model='gemini-2.5-flash', contents=revision_contents, config=config_a)
            draft_response = response_a_revised.text
            
        attempt += 1
        
    return draft_response

# ==========================================
# 8. THE CHAT INTERFACE
# ==========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a follow-up or research query..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        final_output = run_research_pipeline(prompt, st.session_state.messages, uploaded_files_ui)
        st.markdown(final_output)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": final_output})
