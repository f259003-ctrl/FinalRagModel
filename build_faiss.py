import json
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

def build_faiss(chunks_path="data/chunks.json", output_dir="vectorstore/faiss_index"):
    with open(chunks_path) as f:
        chunks = json.load(f)

    docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(output_dir)

    print(f"FAISS index saved to {output_dir}")

if __name__ == "__main__":
    build_faiss()
