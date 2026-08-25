import os

from dotenv import load_dotenv
from openai import OpenAI


# ==========================================
# NVIDIA API SETUP
# ==========================================

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)


# ==========================================
# STEP 1
# Job Posting → Structured Requirements
# ==========================================

def extract_job_requirements(job_posting_text):

    prompt = f"""
You are a job requirements extraction system.

Analyze the following job posting.

Extract the important hiring requirements.

Return the result in this structure:

{{
    "job_title": "",
    "required_skills": [],
    "preferred_skills": [],
    "experience_required": "",
    "education_required": "",
    "responsibilities": [],
    "key_qualifications": [],
    "location": "",
    "employment_type": ""
}}

Rules:
- Extract information only from the job posting.
- Do not invent information.
- If information is missing, use an empty string or empty list.
- Keep skills concise.
- Separate required skills from preferred skills.

Job Posting:

{job_posting_text}
"""

    response = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
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
# STEP 2
# Requirements + Candidate → Outreach
# ==========================================

def generate_outreach(
    structured_requirements,
    candidate_profile
):

    prompt = f"""
You are a professional recruiter.

Create a personalized candidate outreach message.

Use ONLY:
1. The structured job requirements.
2. The candidate profile.

Do NOT use or assume information from the original job posting.

Requirements:
- Mention the job title.
- Identify 1–3 relevant skills or experiences from the candidate.
- Explain why the candidate appears to be a good match.
- Do not invent candidate experience.
- Keep the message professional and friendly.
- Keep it under 150 words.
- End with a simple call to action.

Structured Job Requirements:

{structured_requirements}

Candidate Profile:

{candidate_profile}

Return only the outreach message.
"""

    response = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.5
    )

    return response.choices[0].message.content


# ==========================================
# INPUT
# ==========================================

job_posting = """
We are looking for a Software Engineer Intern.

Requirements:
- Strong knowledge of Python
- Experience with FastAPI
- Knowledge of SQL and Git
- Understanding of REST APIs
- Machine learning experience is preferred

Responsibilities:
- Develop backend APIs
- Work with the engineering team
- Write clean and maintainable code

Education:
B.Tech in Computer Science or related field.
"""


candidate_profile = """
Name: Rahul Sharma

Education:
B.Tech in Computer Engineering

Skills:
Python, FastAPI, SQL, Git, Machine Learning

Projects:
Built a REST API using FastAPI and Python.
Worked on a machine learning project for predicting house prices.

Experience:
6-month software development internship.
"""


# ==========================================
# RUN PIPELINE
# ==========================================

requirements = extract_job_requirements(job_posting)

print("\n========== STEP 1 ==========")
print(requirements)


outreach = generate_outreach(
    requirements,
    candidate_profile
)

print("\n========== STEP 2 ==========")
print(outreach)