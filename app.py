# import streamlit as st
# import os
# from retrieval import Retriever
# from generator import RAGGenerator
# import subprocess

# INDEX_PATH = "output"  # based on your folder

# def ensure_index_exists():
#     if not os.path.exists(INDEX_PATH) or len(os.listdir(INDEX_PATH)) == 0:
#         with st.spinner("Setting up knowledge base for first time... ⏳"):
#             subprocess.run(["python3", "ingest.py"], check=True)

# ensure_index_exists()
# # Initialize components
# @st.cache_resource
# def load_system():
#     # Only loads once
#     retriever = Retriever()
#     generator = RAGGenerator()
#     return retriever, generator

# st.set_page_config(page_title="RBS Student Life Assistant", page_icon="🛡️")

# st.title("Student Life Assistant for Rutgers Business School 🛡️")
# st.markdown("Ask questions about RBS contacts, events, majors, and student life! Powered by Hybrid Retrieval (FAISS + BM25) and OpenAI.")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Show previous messages
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])
#         if "sources" in msg and msg["sources"]:
#             with st.expander("View Retrieved Sources"):
#                 for src in msg["sources"]:
#                     st.caption(f"{src['metadata_prefix']} \n\n {src['text']}")

# # Chat Input
# query = st.chat_input("Ask a question (e.g. 'Who is the contact for MITA?')")

# if query:
#     # 1. User messages
#     st.session_state.messages.append({"role": "user", "content": query})
#     with st.chat_message("user"):
#         st.markdown(query)

#     # 2. Process query
#     with st.spinner("Searching specific knowledge base..."):
#         try:
#             retriever, generator = load_system()
#         except FileNotFoundError:
#             st.error("Error: Missing index files. Please run `python ingest.py` first to process documents.")
#             st.stop()
            
#         retrieved_chunks, intent = retriever.retrieve(query, top_k=5)
    
#     with st.spinner(f"Generating answer (Router detected intent: {intent})..."):
#         answer = generator.generate_answer(query, retrieved_chunks)

#     # 3. Bot response
#     with st.chat_message("assistant"):
#         st.markdown(answer)
#         with st.expander("View Retrieved Sources"):
#             for src in retrieved_chunks:
#                 st.caption(f"{src['metadata_prefix']} \n\n {src['text']}")
    
#     st.session_state.messages.append({"role": "assistant", "content": answer, "sources": retrieved_chunks})

# # Sidebar metrics
# with st.sidebar:
#     st.header("Pipeline Info")
#     st.markdown("- **Embeddings:** all-MiniLM-L6-v2")
#     st.markdown("- **Vector DB:** FAISS (Dense)")
#     st.markdown("- **Keyword:** BM25 (Sparse)")
#     st.markdown("- **Reranker:** Reciprocal Rank Fusion")
#     st.markdown("- **LLM:** gpt-4o-mini")
    
import streamlit as st
import os
import subprocess
from retrieval import Retriever
from generator import RAGGenerator

INDEX_FILE = "output/vector_index.faiss"

@st.cache_resource
def setup_index():
    if not os.path.exists(INDEX_FILE):
        with st.spinner("Running ingest.py... ⏳"):
            result = subprocess.run(
                ["python3", "ingest.py"],
                capture_output=True,
                text=True
            )
            
            # 🔥 SHOW LOGS
            st.text("STDOUT:\n" + result.stdout)
            st.text("STDERR:\n" + result.stderr)

            if result.returncode != 0:
                st.error("ingest.py FAILED ❌")
                st.stop()

setup_index()

# ---------- STEP 2: Load system ----------
@st.cache_resource
def load_system():
    retriever = Retriever()
    generator = RAGGenerator()
    return retriever, generator

# ---------- UI ----------
st.set_page_config(page_title="RBS Student Life Assistant", page_icon="🛡️")

st.title("Student Life Assistant for Rutgers Business School 🛡️")
st.markdown(
    "Ask questions about RBS contacts, events, majors, and student life! "
    "Powered by Hybrid Retrieval (FAISS + BM25) and OpenAI."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Retrieved Sources"):
                for src in msg["sources"]:
                    st.caption(f"{src['metadata_prefix']} \n\n {src['text']}")

# ---------- Chat ----------
query = st.chat_input("Ask a question (e.g. 'Who is the contact for MITA?')")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # Load system AFTER index is ready
    try:
        retriever, generator = load_system()
    except Exception as e:
        st.error(f"System failed to load: {e}")
        st.stop()

    # Retrieval
    with st.spinner("Searching knowledge base..."):
        retrieved_chunks, intent = retriever.retrieve(query, top_k=5)

    # Generation
    with st.spinner(f"Generating answer (Intent: {intent})..."):
        answer = generator.generate_answer(query, retrieved_chunks)

    # Response
    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("View Retrieved Sources"):
            for src in retrieved_chunks:
                st.caption(f"{src['metadata_prefix']} \n\n {src['text']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved_chunks
    })

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Pipeline Info")
    st.markdown("- **Embeddings:** all-MiniLM-L6-v2")
    st.markdown("- **Vector DB:** FAISS (Dense)")
    st.markdown("- **Keyword:** BM25 (Sparse)")
    st.markdown("- **Reranker:** Reciprocal Rank Fusion")
    st.markdown("- **LLM:** gpt-4o-mini")