from langchain_community.document_loaders import PyPDFLoader


PDF_FILES = [
    ("attention is all you need.pdf", 1, "Attention Is All You Need"),
    ("bert.pdf", 2, "BERT"),
    ("luong attention.pdf", 3, "Luong Attention")
]


def load_documents():

    all_documents = []

    for pdf_file, document_id, document_name in PDF_FILES:

        print(f"Loading: {pdf_file}")

        loader = PyPDFLoader(pdf_file)

        documents = loader.load()

        for document in documents:

            document.metadata["document_id"] = document_id
            document.metadata["document_name"] = document_name

        all_documents.extend(documents)

        print(f"Pages loaded: {len(documents)}")

    return all_documents