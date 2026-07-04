import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "data"
VECTORSTORE_PATH = "vectorstore"


def create_vectorstore():

    if os.path.exists(VECTORSTORE_PATH) and os.listdir(VECTORSTORE_PATH):
        print("Vector database already exists.")
        return

    documents = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DATA_PATH, file)

            print(f"Loading {file}...")

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            documents.extend(docs)

    print(f"Total pages loaded: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )

    vectordb.persist()

    print("Vector database created successfully!")