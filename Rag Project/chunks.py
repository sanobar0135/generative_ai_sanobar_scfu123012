from loader import load_documents

from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks():

    # Load all PDFs
    documents = load_documents()

    print("\nTotal pages:", len(documents))

    # Create splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    # Create chunks
    chunks = splitter.split_documents(documents)

    print("Total chunks:", len(chunks))

    return chunks