from fastapi import FastAPI, UploadFile, File
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader
from transformers import pipeline
import re

app = FastAPI()

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
#llm = pipeline("text-generation", model="distilgpt2")
llm = pipeline("text-generation", model="google/flan-t5-base")
# In-memory DB
documents = []
index = None

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces/newlines
    text = re.sub(r'CodeEvidence\d+', '', text)
    text = re.sub(r'BrowserEvidence\d+', '', text)
    return text.strip()

def split_into_chunks(text, chunk_size=300, overlap=50):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# 🔹 Step 1: Upload PDF
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global documents, index

    reader = PdfReader(file.file)
    text = ""

    for page in reader.pages:
        text += page.extract_text()
    
     # Clean full text
    text = clean_text(text)

    # Split into chunks
    chunks = split_into_chunks(text)
    documents = chunks

    # Create embeddings
    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return {"message": "PDF processed successfully", "chunks": len(chunks)}

# 🔹 Step 2: Ask question
@app.get("/ask")
def ask_question(query: str):
    global index, documents

    if index is None:
        return {"error": "Upload PDF first"}

    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=5)

    results = []
    for i, score in zip(I[0], D[0]):
        if score < 1.5:
            results.append(documents[i])

    if not results:
        results = [documents[i] for i in I[0][:3]]

    context = " ".join(results[:3])
    context = clean_text(context)

    prompt = f"""
You are a helpful assistant.

Answer ONLY from the context.
If the answer is not clearly present, say: Not found in document.

Context:
{context}

Question:
{query}

Answer in one clear sentence:
"""

    response = llm(prompt, max_length=120, do_sample=False)
    answer = response[0]["generated_text"].strip()

    return {
        "question": query,
        "answer": answer
    }
@app.get("/")
def home():
    return {"message": "PDF Chatbot Running 🚀"}