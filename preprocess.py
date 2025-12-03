import pandas as pd
import re
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

def clean_text(t):
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def preprocess_dataset(csv_path="data/medical_transcriptions.csv", output_path="data/chunks.json"):
    df = pd.read_csv(csv_path)
    df.dropna(subset=["transcription"], inplace=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks = []
    for idx, row in df.iterrows():
        text = clean_text(row["transcription"])
        chunks = splitter.split_text(text)

        for i, c in enumerate(chunks):
            all_chunks.append({
                "text": c,
                "metadata": {
                    "medical_specialty": row["medical_specialty"],
                    "description": row["description"],
                    "chunk_id": f"{idx}_{i}"
                }
            })

    with open(output_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Saved {len(all_chunks)} chunks to {output_path}")

if __name__ == "__main__":
    preprocess_dataset()
