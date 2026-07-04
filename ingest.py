import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


DATA_PATH = "data"
VECTORSTORE_PATH = "vectorstore"


# Load all PDFs
documents = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(DATA_PATH, file)

        print(f"Loading {file}...")

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        documents.extend(docs)

print(f"\nTotal pages loaded: {len(documents)}")


# Chunk documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")


# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create Chroma DB
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=VECTORSTORE_PATH
)

vectordb.persist()

print("\nVector database created successfully!")
print(f"Stored in: {VECTORSTORE_PATH}")