from chunks import create_chunks

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


DB_PATH = "./vector_db"


def create_vector_database():

    # Get chunks
    chunks = create_chunks()

    # Create embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create Chroma vector database
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="research_papers",
        persist_directory=DB_PATH
    )

    print("\nVector database created successfully!")
    print("Stored chunks:", len(chunks))

    return vector_db


def load_vector_database():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma(
        collection_name="research_papers",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )

    return vector_db