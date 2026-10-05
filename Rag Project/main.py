from vector_db import create_vector_database
from rag import answer_question


# ---------------------------------------
# STEP 1
# Create vector database
# ---------------------------------------

print("Creating vector database...\n")

create_vector_database()


# ---------------------------------------
# STEP 2
# Ask question
# ---------------------------------------

question = input(
    "\nEnter your research question: "
)


# ---------------------------------------
# STEP 3
# Run RAG
# ---------------------------------------

result = answer_question(question)


# ---------------------------------------
# STEP 4
# Print final result
# ---------------------------------------

print("\n")
print("=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(result["answer"])


print("\n")
print("=" * 60)
print("DOCUMENT STATISTICS")
print("=" * 60)

for document, count in result["all_counts"].items():

    print(
        f"{document}: {count} chunks"
    )


print("\n")
print("=" * 60)
print("DOCUMENT USED FOR ANSWER")
print("=" * 60)

print(
    result["selected_document"]
)