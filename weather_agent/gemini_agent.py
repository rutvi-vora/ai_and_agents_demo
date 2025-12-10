import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import requests
import json
from pydantic import BaseModel, Field
from typing import Optional


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------
# Tools
# ---------------------------

def run_command(cmd: str):
    return os.popen(cmd).read()

def get_weather(city):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    r = requests.get(url)
    return r.text if r.status_code == 200 else "Unable to fetch weather"

available_tools = {
    "get_weather": get_weather,
    "run_command": run_command
}

# ---------------------------
# Prompt
# ---------------------------

SYSTEM_PROMPT = """
You must output ONLY valid JSON in this format:
{"step": "START" | "PLAN" | "OUTPUT" | "TOOL", "content": "string", "tool": "string", "input": "string"}

Follow the START → PLAN → TOOL(optional) → OUTPUT workflow.
"""

class MyOutputModel(BaseModel):
    step: str
    content: Optional[str] = None
    tool: Optional[str] = None
    input: Optional[str] = None

# ---------------------------
# History buffer (Gemini 2.x requires Content list)
# ---------------------------

history = [
    types.Content(
        role="user",
        parts=[
                types.Part(text=SYSTEM_PROMPT)
            ]
    )
]

# ---------------------------
# Main loop
# ---------------------------

user_query = input("> ")
history.append(
    types.Content(role="user",
                  parts=[
                      types.Part(text=user_query)
                  ])
)

while True:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=history,
    )

    msg = response.text
    print("\nRAW:", msg)

    try:
        parsed = MyOutputModel.model_validate_json(msg)
    except:
        print("❌ Error: Model did not return valid JSON.")
        break

    # Store assistant message
    history.append(types.Content(role="model",
                                 parts=[
                                     types.Part(text=msg)
                                 ]))

    # STEP HANDLING
    if parsed.step == "PLAN":
        print("PLAN:", parsed.content)
        continue

    if parsed.step == "TOOL":
        tool_name = parsed.tool
        tool_input = parsed.input
        print(f"Calling tool: {tool_name}({tool_input})")

        output = available_tools[tool_name](tool_input)

        history.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(text=json.dumps({
                    "step": "OBSERVE",
                    "tool": tool_name,
                    "input": tool_input,
                    "output": output
                }))
                ]

            )
        )
        continue

    if parsed.step == "OUTPUT":
        print("OUTPUT:", parsed.content)
        break


# class GeminiAgent:
#     def __init__(self, model_name="gemini-pro"):
#         # Configure the genai library with an API key from environment variable
#         api_key = os.environ.get("GEMINI_API_KEY")
#         if not api_key:
#             raise ValueError("GEMINI_API_KEY environment variable not set.")
#         genai.configure(api_key=api_key)
#         self.model = genai.GenerativeModel(model_name)
#
#     def chat(self, messages):
#         system_instruction_content = None
#         gemini_formatted_messages = []
#
#         for msg in messages:
#             role = msg["role"]
#             content = msg["content"]
#
#             if role == "system":
#                 # Consolidate all system messages into a single system_instruction
#                 if system_instruction_content:
#                     system_instruction_content += "\n" + content
#                 else:
#                     system_instruction_content = content
#             elif role == "user":
#                 gemini_formatted_messages.append(genai_types.Content(role="user", parts=[content]))
#             elif role == "assistant":
#                 # Gemini's role for assistant responses is 'model'
#                 gemini_formatted_messages.append(genai_types.Content(role="model", parts=[content]))
#             else:
#                 # You might want to log or raise an error for unsupported roles
#                 print(f"Warning: Unsupported role '{role}' encountered. Skipping message.")
#
#         try:
#             # Call generate_content with the system_instruction and the chat history
#             response = self.model.generate_content(
#                 contents=gemini_formatted_messages,
#                 system_instruction=system_instruction_content, # Can be None
#                 generation_config={"temperature": 0.7, "top_p": 0.95, "top_k": 40} # Example config
#             )
#
#             # Extract the text from the response
#             # A `GenerateContentResponse` object has `candidates`
#             if response.candidates:
#                 # Assuming we care about the first candidate and its first part
#                 first_candidate_content = response.candidates[0].content
#                 if first_candidate_content and first_candidate_content.parts:
#                     return first_candidate_content.parts[0].text
#
#             # Fallback if no suitable text is found (e.g., if response is empty or error)
#             return f"No text content found in Gemini response. Full response: {str(response)}"
#
#         except Exception as e:
#             # Handle potential errors from the API call
#             print(f"Error calling Gemini API: {e}")
#             return f"Error: {e}"
