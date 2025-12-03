import streamlit as st
from rag_chain import load_rag_chain
import os

st.set_page_config(page_title="Medical RAG Assistant", layout="wide")

@st.cache_resource
def load_chain():
    return load_rag_chain()

qa_chain = load_chain()

st.title("🩺 Medical RAG Assistant (Evidence-Based)")

user_query = st.text_input("Enter a medical question:")

if st.button("Search") and user_query:
    with st.spinner("Retrieving evidence..."):
        result = qa_chain({"query": user_query})
        answer = result["result"]
        sources = result["source_documents"]

    st.subheader("🔎 Answer:")
    st.write(answer)

    st.subheader("📚 Evidence Sources:")
    for s in sources:
        st.write("---")
        st.write("**Chunk ID:**", s.metadata["chunk_id"])
        st.write("**Specialty:**", s.metadata["medical_specialty"])
        st.write(s.page_content)
