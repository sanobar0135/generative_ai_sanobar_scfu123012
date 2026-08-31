### Exercise 2 — News Article to Fact Card Pipeline. Build a 2-step pipeline. Step 1 extracts the core claims from a news article as a list. 

# Step 2 generates a short "fact card" (headline, 3 bullet points, source confidence note) using only the extracted claims from Step 1.

##**Input:** `article_text`

##**Output:** the list of extracted claims from Step 1, and the fact card from Step 2


import os 
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)



#STEP-1

def extract_claims(article):

    prompt = f"""  
    """
    response = client.chat.completion.create(
        model = "nvidia/nemotron-nano-12b",
        messages = [{

        
            "role": "system",
            "content": "you are a facts claimer system thats extracts just important headlines and claims."
        }
        ]
    )