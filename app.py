# streamlit_app.py
import streamlit as st
from rag_pipeline import FaissRetriever, ChromaRetriever, GeminiClient, RAGChain
import os
import traceback

st.set_page_config(page_title="Medical RAG QA (FAISS + Gemini)", layout="wide")

st.title("Medical RAG QA — FAISS Retriever + Gemini (Vertex)")

with st.sidebar:
    st.markdown("### Settings")
    faiss_index_path = st.text_input("FAISS index path", value="/content/faiss_index.idx")
    faiss_meta_path = st.text_input("FAISS metadata path", value="/content/faiss_metadata.pkl")
    use_chroma = st.checkbox("Use Chroma Cloud retriever (instead of FAISS)", value=False)
    chroma_collection = st.text_input("Chroma collection name", value="medical_rag")
    gemini_model = st.text_input("Gemini model", value="gemini-2.5-flash")
    k = st.number_input("Top-k retrieved chunks", min_value=1, max_value=20, value=5)
    temperature = st.slider("Generation temperature", 0.0, 1.0, 0.0, 0.05)
    max_tokens = st.number_input("Max output tokens", min_value=64, max_value=2048, value=512, step=64)

# Load retriever (on button or runtime)
@st.cache_resource
def load_retriever(use_chroma, idx_path, meta_path, chroma_collection):
    if use_chroma:
        return ChromaRetriever(collection_name=chroma_collection)
    else:
        return FaissRetriever(index_path=idx_path, metadata_path=meta_path)

@st.cache_resource
def load_generator(model):
    return GeminiClient(model=model)

try:
    retriever = load_retriever(use_chroma, faiss_index_path, faiss_meta_path, chroma_collection)
except Exception as e:
    st.sidebar.error("Failed to initialize retriever: " + str(e))
    st.sidebar.text(traceback.format_exc())
    retriever = None

generator = None
try:
    generator = load_generator(gemini_model)
except Exception as e:
    st.sidebar.error("Failed to initialize Gemini client: " + str(e))

if retriever and generator:
    rag = RAGChain(retriever=retriever, generator=generator, k=int(k))

    st.markdown("### Ask a clinical question (short or long)")
    question = st.text_area("Question", height=120)

    col1, col2 = st.columns([2,1])
    with col1:
        if st.button("Generate answer"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Retrieving and generating answer..."):
                    try:
                        out = rag.run(question, temperature=float(temperature), max_output_tokens=int(max_tokens))
                        st.markdown("#### Answer")
                        st.write(out["answer"])
                        st.markdown("#### Retrieved contexts (top-k)")
                        for i, c in enumerate(out["contexts"]):
                            meta = c.get("metadata", {})
                            st.markdown(f"**Source {i+1}** — {meta}")
                            st.write(c["text"][:2000])
                        with st.expander("Show full prompt sent to Gemini"):
                            st.code(out["prompt"][:20000])
                    except Exception as e:
                        st.error("Error during RAG run: " + str(e))
                        st.text(traceback.format_exc())
    with col2:
        st.markdown("### Controls & Info")
        st.markdown(f"- Retriever: {'Chroma Cloud' if use_chroma else 'FAISS (local)'}")
        st.markdown(f"- Gemini model: `{gemini_model}`")
        st.markdown(f"- Top-k: {k}")
        st.markdown("**Security**: Keep your GEMINI API key and Chroma keys secret. Do not commit them to source control.")
else:
    st.error("Retriever or generator failed to initialize. Check sidebar messages.")
