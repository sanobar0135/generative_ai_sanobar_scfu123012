# customer_review_to_support_ticket.py

from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import (
    PydanticOutputParser,
    StrOutputParser
)

from pydantic import BaseModel, Field
import os


# ============================================================
# MODEL
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)


# ============================================================
# STEP 1: STRUCTURED OUTPUT
# ============================================================

class ComplaintData(BaseModel):

    complaint: str = Field(
        description="The core complaint made by the customer"
    )

    product_or_feature: str = Field(
        description="The product or feature mentioned by the customer"
    )

    sentiment: str = Field(
        description="The customer's sentiment such as Positive, Negative, or Neutral"
    )


# Create Pydantic Output Parser
parser = PydanticOutputParser(
    pydantic_object=ComplaintData
)


# ============================================================
# STEP 1: PROMPT TEMPLATE
# ============================================================

input_prompt = PromptTemplate(
    template="""
You are a customer review analyzer.

Analyze the customer review below and extract:

1. The core complaint
2. The product or feature mentioned
3. The customer sentiment

Return the result only in the required structured format.

{format_instructions}

Customer Review:
{review_text}
""",

    input_variables=["review_text"],

    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)


# ============================================================
# STEP 1 PIPELINE
# ============================================================

step1_chain = input_prompt | llm | parser


# ============================================================
# STEP 2: SUPPORT TICKET WRITER
# ============================================================

ticket_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
You are a support ticket writer.

Your job is to create a concise and professional
support ticket summary for the internal support team.

Use ONLY the structured information provided from Step 1.

Do NOT use the original customer review.

Include:
- Complaint
- Product/Feature
- Sentiment
- Recommended Action
"""
    ),

    (
        "human",
        """
Structured Complaint Data:

Complaint: {complaint}

Product/Feature: {product_or_feature}

Sentiment: {sentiment}

Generate the support ticket summary.
"""
    )
])


# ============================================================
# STEP 2 PIPELINE
# ============================================================

step2_chain = ticket_prompt | llm | StrOutputParser()


# ============================================================
# INPUT
# ============================================================

review_text = """
I purchased the wireless headphones last week.
The sound quality is good, but the Bluetooth connection
keeps disconnecting every few minutes. It is very frustrating
because I cannot use the headphones properly.
"""


# ============================================================
# RUN STEP 1
# ============================================================

structured_data = step1_chain.invoke({
    "review_text": review_text
})


# ============================================================
# RUN STEP 2
# IMPORTANT:
# Only structured data from Step 1 is passed here.
# The original review_text is NOT passed.
# ============================================================

support_ticket = step2_chain.invoke({

    "complaint": structured_data.complaint,

    "product_or_feature": structured_data.product_or_feature,

    "sentiment": structured_data.sentiment
})


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n========== STEP 1: STRUCTURED COMPLAINT DATA ==========")

print("Complaint:", structured_data.complaint)

print("Product/Feature:", structured_data.product_or_feature)

print("Sentiment:", structured_data.sentiment)


print("\n========== STEP 2: SUPPORT TICKET SUMMARY ==========")

print(support_ticket)