import os
import uuid
import streamlit as st

# Fix HuggingFace tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import (
    PDF_DEFAULT,
    CHUNK_TOKENS,
    OVERLAP_TOKENS,
    TOKEN_CHAR_RATIO,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
)
from pdf_utils import extract_pages_with_lines, chunk_by_line_ranges
from chunking import chunk_turns
from embeddings import get_sentence_transformer
from chroma_utils import (
    create_chroma_client,
    upsert_chunks_to_chroma,
    query_negative_statement,
)
from claim_checker import check_claim_with_ollama

# -----------------------
# STREAMLIT APP
# -----------------------
st.set_page_config(page_title="Second Chair — POC", layout="wide")
st.title("Second Chair — POC (Transcript chunking, embedding, retrieval)")

st.sidebar.header("Inputs")
uploaded_file = st.sidebar.file_uploader("Upload deposition PDF", type=["pdf"])

chunk_tokens = st.sidebar.number_input(
    "chunk_tokens", value=CHUNK_TOKENS, min_value=50, max_value=2000, step=50
)
overlap_tokens = st.sidebar.number_input(
    "overlap_tokens", value=OVERLAP_TOKENS, min_value=0, max_value=1000, step=25
)

do_extract = st.sidebar.button("Extract, chunk & embed PDF")
run_query = st.sidebar.button("Run negative-statement query")

status = st.empty()
pdf_path = None
# if use_default:
#     if not os.path.exists(PDF_DEFAULT):
#         st.error(f"Default PDF not found at {PDF_DEFAULT}")
#     else:
#         pdf_path = PDF_DEFAULT
if uploaded_file:
    tmp_path = f"./uploaded_{uuid.uuid4().hex}.pdf"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())
    pdf_path = tmp_path

if "chroma_initialized" not in st.session_state:
    st.session_state["chroma_initialized"] = False

if do_extract and pdf_path:
    status.info("Extracting PDF...")
    pages = extract_pages_with_lines(pdf_path)

    status.info("Chunking by line ranges...")
    chunks = chunk_by_line_ranges(pages, lines_per_chunk=10)
    
    # Display chunk statistics
    st.subheader("📊 Chunking Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    chunk_lengths = [len(c.get("clean_text", "")) for c in chunks]
    avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    
    with col1:
        st.metric("Total Chunks", len(chunks))
    with col2:
        st.metric("Avg Characters", f"{avg_length:.0f}")
    with col3:
        st.metric("Min/Max Length", f"{min(chunk_lengths)}/{max(chunk_lengths)}")
    with col4:
        st.metric("Total Pages", len(pages))
    
    # Show chunk type distribution
    chunk_types = {}
    for c in chunks:
        chunk_type = c.get("type", "unknown")
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
    
    if chunk_types:
        st.write("**Chunk Types:**", chunk_types)
    
    # Sample chunks
    st.subheader("📄 Sample Chunks")
    with st.expander("Show first 3 chunks"):
        for i, chunk in enumerate(chunks[:3]):
            st.write(f"**Chunk {i+1}** (Page {chunk.get('page')}, Lines {chunk.get('start_line')}-{chunk.get('end_line')})")
            st.code(chunk.get("clean_text", "")[:300] + "..." if len(chunk.get("clean_text", "")) > 300 else chunk.get("clean_text", ""))

    # Save to session state
    st.session_state["chunks"] = chunks

    status.info("Loading embedding model...")
    embedder = get_sentence_transformer("all-MiniLM-L6-v2")
    st.session_state["embedder"] = embedder

    status.info("Creating Chroma client...")
    chroma_client = create_chroma_client(CHROMA_PERSIST_DIR)
    st.session_state["chroma_client"] = chroma_client

    status.info("Upserting chunks to Chroma...")
    upsert_chunks_to_chroma(chroma_client, COLLECTION_NAME, chunks, embedder, pdf_path)
    st.session_state["chroma_initialized"] = True

    status.success("Chunks embedded and stored in Chroma")


if run_query:
    if not st.session_state.get("chroma_initialized"):
        st.error("Chroma DB not initialized. Run extract step first.")
    else:
        query_text = st.sidebar.text_area(
            "Negative statement",
            value="He denied ever working at Metro-Atlantic in 1963.",
        )
        top_k = st.sidebar.number_input("Top K", value=5, min_value=1, max_value=50)
        hits = query_negative_statement(
            st.session_state["chroma_client"],
            COLLECTION_NAME,
            query_text,
            st.session_state["embedder"],
            top_k=top_k,
        )

        st.write(f"Found {len(hits)} results")
        for i, h in enumerate(hits):
            st.subheader(f"Rank {i+1} — distance {h['distance']:.4f}")
            st.write(h["metadata"])
            st.code(h["document"][:500] + ("..." if len(h["document"]) > 500 else ""))

        # ✅ Ollama Analysis
        ollama_model = st.sidebar.selectbox(
            "Ollama Model", 
            ["llama3.2", "llama2", "mistral", "codellama"],
            index=0,
            help="Choose which Ollama model to use for analysis"
        )
        
        if st.sidebar.button("Analyze with Ollama"):
            with st.spinner(f"Analyzing with {ollama_model}..."):
                verdict = check_claim_with_ollama(query_text, hits, model=ollama_model)
            
            st.header("🔍 Ollama Analysis Results")
            
            # Color-code verdict
            verdict_color = {
                "SUPPORT": "🟢",
                "REFUTE": "🔴", 
                "NOT_FOUND": "🟡",
                "ERROR": "⚫",
                "UNKNOWN": "⚪"
            }.get(verdict['verdict'], "⚪")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Verdict", f"{verdict_color} {verdict['verdict']}")
            with col2:
                st.metric("Confidence", f"{verdict['confidence']}%")
            
            st.subheader("Analysis")
            st.write(verdict['explanation'])
            
            st.subheader("Evidence Sent to Model")
            st.info(f"Analyzed {len(hits)} evidence chunks with {ollama_model}")