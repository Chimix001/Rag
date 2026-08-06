#Installing dependencies
import os
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

#loading and extracting the document
def loading_document():
    print('loading documents.....')
    loader = CSVLoader(
        file_path=r"C:\Users\user\Desktop\Customer Agent\banking_knowledge_base_1000.csv",
        encoding="utf-8")
    documents = loader.load()

    return documents
documents = loading_document()
print(documents[0])

#-----creating embeddings

def create_vector_store(documents, persist_directory="db/chroma_db"):
    """creating persists chromaDB vectore store"""
    print('creating embeding ad storing in ChromaDB')

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

documents = loading_document()
vector_store = create_vector_store(documents)

query = "How do I open a savings account?"

results = vector_store.similarity_search(query)
for doc in results:
    print(doc.page_content)

#-----injestion pipeline

def main():
    """Main Ingeston pipeline"""
    print('RAG Ingestion pipeline')

#define path

    persistent_directory = "db/chroma_db"

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model, 
            collection_metadata={"hnsw:space": "cosine"}
        )
    print(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
    return vectorstore

print("Persistent directory does not exist. Initializing vector store...\n")

if __name__ == "__main__":
    main()
