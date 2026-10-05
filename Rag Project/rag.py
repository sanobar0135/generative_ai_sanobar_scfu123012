from collections import Counter

from vector_db import load_vector_database

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os


load_dotenv()


# ---------------------------------------
# Load Vector Database
# ---------------------------------------

vector_db = load_vector_database()


# ---------------------------------------
# Load LLM
# ---------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)


# ---------------------------------------
# RAG FUNCTION
# ---------------------------------------

def answer_question(question):

    # -----------------------------------
    # STEP 1: Retrieve TOP 10 chunks
    # -----------------------------------

    results = vector_db.similarity_search(
        question,
        k=10
    )

    print("\n========== TOP 10 CHUNKS ==========")

    for i, result in enumerate(results):

        print(
            f"{i+1}. "
            f"{result.metadata['document_name']} "
            f"(Page {result.metadata['page'] + 1})"
        )


    # -----------------------------------
    # STEP 2: Count documents
    # -----------------------------------

    document_names = [
        result.metadata["document_name"]
        for result in results
    ]

    counts = Counter(document_names)


    print("\n========== DOCUMENT COUNT ==========")

    for document, count in counts.items():

        print(f"{document}: {count} chunks")


    # -----------------------------------
    # STEP 3: Find highest document
    # -----------------------------------

    highest_document = counts.most_common(1)[0][0]

    highest_count = counts.most_common(1)[0][1]


    print("\n========== SELECTED DOCUMENT ==========")

    print(
        f"{highest_document} "
        f"with {highest_count} chunks"
    )


    # -----------------------------------
    # STEP 4: Keep only chunks
    # from highest document
    # -----------------------------------

    selected_chunks = [
        result
        for result in results
        if result.metadata["document_name"] == highest_document
    ]


    # -----------------------------------
    # STEP 5: Create context
    # -----------------------------------

    context = "\n\n".join(
        [
            chunk.page_content
            for chunk in selected_chunks
        ]
    )


    # -----------------------------------
    # STEP 6: Prompt LLM
    # -----------------------------------

    prompt = ChatPromptTemplate.from_template(
        """
You are a research paper assistant.

Answer the question using ONLY the provided
research paper context.

If the answer is not available in the context,
say that the answer cannot be found.

Research Paper:
{document}

Context:
{context}

Question:
{question}

Give a clear and concise answer.
"""
    )


    # -----------------------------------
    # STEP 7: Send to LLM
    # -----------------------------------

    chain = prompt | llm

    response = chain.invoke(
        {
            "document": highest_document,
            "context": context,
            "question": question
        }
    )


    # -----------------------------------
    # RETURN EVERYTHING
    # -----------------------------------

    return {
        "answer": response.content,
        "selected_document": highest_document,
        "document_count": highest_count,
        "all_counts": counts,
        "top_10": results
    }