# src/embed_store.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize Chroma client & collection
chroma_client = chromadb.PersistentClient(path="chroma_db")

collection = chroma_client.create_collection(name="transcripts")

# Load embedding model (MiniLM)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def store_transcript(file_path, transcript_id):
    with open(file_path, "r") as f:
        text = f.read()
    chunks = chunk_text(text)

    # Create embeddings
    embeddings = embedding_model.encode(chunks).tolist()

    # Store chunks in Chroma
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{transcript_id}_{i}" for i in range(len(chunks))]
    )
    print(f"✅ Stored transcript '{transcript_id}' with {len(chunks)} chunks.")

if __name__ == "__main__":
    store_transcript("data/sample.txt", "meeting1")
