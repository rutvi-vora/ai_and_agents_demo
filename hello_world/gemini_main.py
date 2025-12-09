from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": "You are an expert in Maths and only and only ans maths related questions."},
        {"role": "user", "content": "Hey there"},
    ],
)

print(response.choices[0].message.content)
