


# ==========================================
# 1. LOAD NVIDIA API KEY
# ==========================================

import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise ValueError("NVIDIA_API_KEY is not found")


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)


MODEL = "openai/gpt-oss-20b"


# ==========================================
# 2. STEP 1 - DISCUSSION SUMMARY
# ==========================================

def discussion_summary(transcript):

    prompt = f"""
You are a meeting discussion extractor.

Your task is to analyze the meeting transcript and extract the important topics,
decisions, problems, and discussions.

Do NOT identify action items yet.

Return the result in this format:

{{
  "topics_discussed": [
    {{
      "topic": "",
      "summary": "",
      "decision": ""
    }}
  ]
}}

Rules:
- Capture only information explicitly discussed in the transcript.
- Do not invent missing information.
- Separate different topics.
- Include decisions when one was made.
- If no decision was made, use "No decision made".
- Keep summaries concise.

Meeting Transcript:

{transcript}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# ==========================================
# 3. STEP 2 - ACTION ITEMS
# ==========================================

def extract_action_items(discussion):

    prompt = f"""
You are an action-item extraction agent.

You will receive structured information extracted from a meeting.

Identify all tasks or follow-up actions that need to be completed.

For every action item, extract:

- task
- owner
- deadline
- owner_missing
- deadline_missing
- status

Rules:

1. Only create action items that are supported by the discussion.
2. Do not invent an owner or deadline.
3. If the owner is not mentioned, set owner to "Unassigned".
4. If the deadline is not mentioned, set deadline to "Not specified".
5. Set owner_missing to true when no owner is identified.
6. Set deadline_missing to true when no deadline is identified.
7. Status should initially be "Pending".
8. Keep the task description concise.

Return valid JSON only.

Structured Meeting Discussion:

{discussion}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# ==========================================
# 4. STEP 3 - FORMAT TASK TABLE
# ==========================================

def format_task_table(action_items):

    prompt = f"""
You are a task-table formatter.

Convert the provided action items into a clean structured task table.

Use exactly these columns:

| # | Task | Owner | Deadline | Missing Information | Status |

Rules:

- Number tasks sequentially starting from 1.
- Use the exact task descriptions provided.
- Use "Unassigned" when the owner is missing.
- Use "Not specified" when the deadline is missing.
- If owner_missing is true, include "Owner" in Missing Information.
- If deadline_missing is true, include "Deadline" in Missing Information.
- If both are missing, write "Owner, Deadline".
- If nothing is missing, write "None".
- Preserve the status.
- Do not add information that does not exist in the input.

Action Items:

{action_items}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# ==========================================
# 5. INPUT TRANSCRIPT
# ==========================================

transcript_text = """
Meeting Transcript - Team Sync (Aug 31)

Alice: Welcome everyone. Let's start with the database migration.

Bob: I've prepared the migration scripts. I will deploy the database
updates by Friday end of day.

Alice: Great. Next, who is handling the API authentication bug?

Charlie: We discussed it, but we haven't assigned anyone to fix it yet.
We definitely need it fixed soon though.

Alice: Okay, we should assign someone in the next meeting.

What about updating the project documentation?

Bob: I can take documentation, but I am not sure when I can get to it.
No hard date was set.

Alice: Fair enough. Lastly, Charlie, please prepare the design slides
for the client pitch by Wednesday.

Charlie: Will do, I'll have them ready by Wednesday afternoon.
"""


# ==========================================
# 6. RUN PIPELINE
# ==========================================

print("========== STEP 1: Discussion Summary ==========")

discussion = discussion_summary(transcript_text)

print(discussion)


print("\n========== STEP 2: Action Items ==========")

action_items = extract_action_items(discussion)

print(action_items)


print("\n========== STEP 3: Formatted Task Table ==========")

task_table = format_task_table(action_items)

print(task_table)