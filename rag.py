from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load API key from .env
load_dotenv()

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector database
vectordb = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings
)

# Create retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Load Groq LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"
)

print("🌾 Rural Scheme Advisor")
print("Type 'exit' to quit.")

while True:
    query = input("\nAsk: ")

    if query.lower() == "exit":
        break

    # Retrieve relevant documents
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a Rural Government Scheme Advisor.

Answer ONLY from the provided context.

If the answer is not present in the context, say:
"I could not find this information in the available documents."

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content)

    print("\n" + "="*50)
    print("SOURCES")
    print("="*50)

    for i, doc in enumerate(docs, start=1):
        print(f"\nSource {i}")
        print("File:", doc.metadata.get("source", "Unknown"))
        print("Page:", doc.metadata.get("page", "Unknown"))