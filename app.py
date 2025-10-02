import os
import chromadb
import streamlit as st
import subprocess
from sentence_transformers import SentenceTransformer

# Silence HF tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Persistent storage
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="transcripts")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def store_transcript(file, transcript_id):
    raw_bytes = file.read()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")

    chunks = chunk_text(text)
    embeddings = embedding_model.encode(chunks).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{transcript_id}_{i}" for i in range(len(chunks))]
    )
    return len(chunks)

def query_transcripts(question, top_k=3):
    results = collection.query(query_texts=[question], n_results=top_k)
    retrieved_chunks = results["documents"][0]

    context = "\n".join(retrieved_chunks)
    prompt = f"Answer the question using only this context:\n{context}\n\nQuestion: {question}\nAnswer:"

    response = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return response.stdout.decode("utf-8"), retrieved_chunks


# ---------------- Streamlit UI ---------------- #
st.set_page_config(page_title="AI Transcript Query", layout="wide")

st.title("🎙️ AI-Powered Transcript Query System")
st.write("Upload transcripts and ask questions securely, fully offline.")

# Upload multiple transcripts
uploaded_files = st.file_uploader("Upload transcript(s) (.txt)", type="txt", accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        num_chunks = store_transcript(f, transcript_id=f.name)
        st.success(f"✅ Stored transcript `{f.name}` with {num_chunks} chunks.")

# Clear database button
if st.button("🗑️ Clear All Transcripts"):
    chroma_client.delete_collection("transcripts")
    collection = chroma_client.get_or_create_collection(name="transcripts")
    st.warning("All transcripts cleared. Start fresh!")

# Question input
question = st.text_input("Ask a question about your transcripts:")
if st.button("Search") and question:
    answer, sources = query_transcripts(question)
    st.subheader("📌 Answer")
    st.write(answer.strip())

    st.subheader("🔍 Sources (retrieved chunks)")
    for s in sources:
        st.markdown(f"> {s}")
