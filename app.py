import os
import warnings
import io
import contextlib

# Suppress all library noise before any imports
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

import logging
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("sentence_transformers").setLevel(logging.CRITICAL)
logging.getLogger("gemini").setLevel(logging.INFO)  # keep gemini logs visible

import streamlit as st
import tempfile
from embedding_manager import EmbeddingManager
from gemini_integration import GeminiIntegration
from pdf_loader import load_pdf_chunks  # only chunking, not full load

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# --- Session state init ---
if 'embedding_manager' not in st.session_state:
    st.session_state.embedding_manager = None  # created ONCE, lives here
if 'pdf_loaded' not in st.session_state:
    st.session_state.pdf_loaded = False
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = None
if 'gemini_client' not in st.session_state:
    st.session_state.gemini_client = None      # created ONCE, lives here
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("🔬 AI Research Assistant")
st.markdown("Upload a PDF and enter your API key (either order), then start chatting.")

# --- Sidebar: API key ---
with st.sidebar:
    st.header("⚙️ Setup")

    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        help="Get your key from https://aistudio.google.com/app/apikey"
    )

    # Only instantiate GeminiIntegration once — not on every rerun
    if api_key_input and st.session_state.gemini_client is None:
        try:
            st.session_state.gemini_client = GeminiIntegration(api_key=api_key_input)
            st.success("✅ Gemini connected")
        except Exception as e:
            st.error(f"❌ {str(e)}")
    elif st.session_state.gemini_client is not None:
        st.success("✅ Gemini connected")

    st.divider()

    if st.session_state.pdf_loaded:
        st.info(f"📄 {st.session_state.pdf_name}")
    else:
        st.info("No PDF loaded yet")

    if st.button("🗑️ Clear Everything", use_container_width=True):
        st.session_state.pdf_loaded = False
        st.session_state.pdf_name = None
        st.session_state.embedding_manager = None
        st.session_state.chat_history = []
        st.rerun()

# --- Main layout ---
col1, col2 = st.columns([1, 2])

# LEFT: PDF upload
with col1:
    st.subheader("📄 Upload PDF")

    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    # Only process when a NEW file is uploaded (filename changed)
    if uploaded_file is not None:
        new_file = (st.session_state.pdf_name != uploaded_file.name)
        if new_file:
            with st.spinner("Processing PDF..."):
                try:
                    temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Reuse existing EmbeddingManager if available — weights stay loaded
                    # Only create a new one on the very first PDF
                    if st.session_state.embedding_manager is None:
                        st.session_state.embedding_manager = EmbeddingManager()

                    chunks = load_pdf_chunks(temp_path)
                    st.write(f"🔄 Embedding {len(chunks)} chunks...")

                    # store_chunks deletes old collection before storing (already fixed)
                    with contextlib.redirect_stdout(io.StringIO()):
                        st.session_state.embedding_manager.store_chunks(
                            chunks,
                            metadata={"source": uploaded_file.name}
                        )

                    st.session_state.pdf_loaded = True
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chat_history = []

                    st.success(f"✅ Ready — {len(chunks)} chunks embedded")

                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

                except Exception as e:
                    st.error(f"❌ {str(e)}")
                    import traceback
                    with st.expander("Debug"):
                        st.code(traceback.format_exc())
        else:
            st.success(f"✅ {uploaded_file.name} already loaded")

# RIGHT: Chat
with col2:
    st.subheader("💬 Chat")

    if not st.session_state.pdf_loaded:
        st.info("👈 Upload a PDF first")
    elif not st.session_state.gemini_client:
        st.info("👈 Enter your Gemini API key in the sidebar")
    else:
        # Render existing chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and "context" in message:
                    with st.expander("📚 Sources"):
                        for i, ctx in enumerate(message["context"], 1):
                            st.caption(f"**Source {i}:**")
                            st.text(ctx["text"][:200] + "...")

        user_input = st.chat_input("Ask anything about your PDF...")

        if user_input:
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })

            try:
                with st.spinner("Thinking..."):
                    context_results = st.session_state.embedding_manager.search(
                        user_input, top_k=5
                    )
                    context = [{"text": r["text"]} for r in context_results]

                    response = st.session_state.gemini_client.ask_question(
                        user_input, context=context
                    )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "context": context
                })

                st.rerun()
            except Exception as e:
                st.error(f"❌ {str(e)}")
                import traceback
                with st.expander("Debug"):
                    st.code(traceback.format_exc())

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
            