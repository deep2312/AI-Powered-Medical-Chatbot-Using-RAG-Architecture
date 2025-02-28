from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
import os
import numpy as np

# Paths
DATA_PATH = 'data/'  # Path to the directory containing PDF files
CSV_PATH = 'data/drug_conditions.csv'  # Path to the CSV file
DB_FAISS_PATH = 'vecstore/db_faiss'  # Path to save the FAISS database
EMBEDDINGS_PATH = 'vecstore/embeddings.npy'  # Path to save precomputed embeddings

# Load PubMedBERT for embeddings (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("NeuML/pubmedbert-base-embeddings")
model = AutoModel.from_pretrained("NeuML/pubmedbert-base-embeddings").to(device)

# Function to generate embeddings in batches
def generate_embeddings_batch(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

# Create vector database
def create_vector_db():
    # Step 1: Load and process PDF documents
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    pdf_documents = loader.load()

    # Step 2: Load and process CSV data
    csv_data = pd.read_csv(CSV_PATH)
    csv_documents = [
        Document(
            page_content=f"Drug: {row['drugName']}\nCondition: {row['condition']}",
            metadata={"source": "CSV"}
        )
        for _, row in csv_data.iterrows()
    ]

    # Combine PDF and CSV documents
    all_documents = pdf_documents + csv_documents

    # Step 3: Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = [doc.page_content for doc in text_splitter.split_documents(all_documents)]

    # Step 4: Generate embeddings
    if os.path.exists(EMBEDDINGS_PATH):
        print("Loading precomputed embeddings...")
        embeddings = np.load(EMBEDDINGS_PATH)
    else:
        print("Generating embeddings...")
        embeddings = generate_embeddings_batch(texts)
        np.save(EMBEDDINGS_PATH, embeddings)
        print(f"Embeddings saved to {EMBEDDINGS_PATH}")

    # Step 5: Create FAISS vector database
    print("Creating FAISS vector database...")
    db = FAISS.from_embeddings(embeddings, texts)

    # Step 6: Save FAISS vector database
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    print(f"FAISS vector database saved to {DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vector_db()
