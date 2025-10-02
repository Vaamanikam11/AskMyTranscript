# src/query_engine.py
import chromadb
import subprocess

# Connect to existing Chroma collection
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection("transcripts")

def query_transcripts(question, top_k=2):
    # Retrieve most relevant chunks
    results = collection.query(query_texts=[question], n_results=top_k)
    retrieved_chunks = results["documents"][0]

    # Build context
    context = "\n".join(retrieved_chunks)
    prompt = f"Answer the question using only this context:\n{context}\n\nQuestion: {question}\nAnswer:"

    # Send to Ollama (Mistral model must be pulled: ollama pull mistral)
    response = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode("utf-8"),
        capture_output=True
    )

    return response.stdout.decode("utf-8")

if __name__ == "__main__":
    while True:
        user_q = input("Ask a question (or 'exit'): ")
        if user_q.lower() == "exit":
            break
        print(query_transcripts(user_q))
