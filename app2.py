import streamlit as st
from google import genai
from google.genai import types
import json
import tempfile
import os
import time

# ==========================================
# 1. PAGE SETUP & UI PROTECTION
# ==========================================
st.set_page_config(page_title="Multi-Agent Researcher", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp [data-testid="stToolbar"] {display: none;}
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Master's-Level AI Researcher")

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
    # UPDATED: Added docx to supported types
    uploaded_files_ui = st.file_uploader("Upload Documents", accept_multiple_files=True, type=['pdf', 'docx', 'txt'])
    
    if st.button("🗑️ Clear Chat / Start Over", type="primary"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 5. AUTHENTICATION
# ==========================================
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# ==========================================
# 6. SYSTEM PROMPTS
# ==========================================
agent_a_prompt = """
Role: Master's-Level Research Assistant. 
Directive: Absolute factual accuracy. Synthesize Google Search and uploaded documents.
Verification: Generate a [VERIFICATION LOG] at the top checking Sources, Quotes, and Confidence.
"""

agent_b_prompt = """
Role: Ruthless Peer Reviewer. Check for hallucinations.
Output ONLY JSON: {"status": "PASS", "feedback": ""} OR {"status": "FAIL", "feedback": "reason"}
"""

# ==========================================
# 7. THE MULTI-AGENT PIPELINE
# ==========================================

def process_files(files):
    gemini_files = []
    for f in files:
        # Detect the extension to create the correct temp file
        file_extension = os.path.splitext(f.name)[1]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            tmp.write(f.getbuffer())
            tmp_path = tmp.name
        
        try:
            # Upload without forcing a MIME type - let Google detect it
            g_file = client.files.upload(
                file=tmp_path, 
                config={'display_name': f.name}
            )
            
            # Polling loop to wait for processing (Important for Word/PDF)
            while g_file.state.name == "PROCESSING":
                time.sleep(2)
                g_file = client.files.get(name=g_file.name)
            
            if g_file.state.name == "FAILED":
                st.error(f"File {f.name} failed to process.")
                continue
                
            gemini_files.append(g_file)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    return gemini_files

def run_research_pipeline(user_input, chat_history, files):
    status_box = st.empty()
    status_box.info("Researcher is active...")

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
    
    # Build History
    contents_a = []
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "model"
        contents_a.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    
    # Process current parts (Files + Text)
    current_parts = []
    if files:
        processed = process_files(files)
        for g_file in processed:
            # We use the MIME type returned by the API after upload
            current_parts.append(types.Part.from_uri(file_uri=g_file.uri, mime_type=g_file.mime_type))
    
    current_parts.append(types.Part.from_text(text=user_input))
    contents_a.append(types.Content(role="user", parts=current_parts))

    # Agent A
    status_box.info("Synthesizing research...")
    response_a = client.models.generate_content(model=MODEL_NAME, contents=contents_a, config=config_a)
    draft_response = response_a.text
    
    # Agent B loop
    attempt = 0
    while attempt < 2:
        status_box.warning(f"Peer Review (Attempt {attempt+1})...")
        response_b = client.models.generate_content(
            model=MODEL_NAME, 
            contents=f"Review this research for the query '{user_input}':\n\n{draft_response}", 
            config=config_b
        )
        
        try:
            res = json.loads(response_b.text)
            if res['status'] == "PASS":
                status_box.success("Research Verified.")
                return draft_response
            else:
                status_box.error(f"Revision needed: {res['feedback']}")
                rev_prompt = f"Peer review failed: {res['feedback']}. Revise the response."
                contents_a.append(types.Content(role="model", parts=[types.Part.from_text(text=draft_response)]))
                contents_a.append(types.Content(role="user", parts=[types.Part.from_text(text=rev_prompt)]))
                response_a = client.models.generate_content(model=MODEL_NAME, contents=contents_a, config=config_a)
                draft_response = response_a.text
        except:
            return draft_response
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
