import  os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import asyncio
import speech_recognition as sr

from openai import AsyncOpenAI
from openai import OpenAI
from openai.helpers import LocalAudioPlayer

from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    # api_key=GEMINI_API_KEY,
    # base_url="https://generativelanguage.googleapis.com/v1beta/"
)
async_client = AsyncOpenAI()

async def tts(speech: str):
    async with async_client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            instructions="Always speak in cheerful manner with full of delight and happy",
            input=speech,
            response_format="pcm"
    ) as response:
        await LocalAudioPlayer().play(response)

def run_command(cmd: str):
    result = os.system(cmd)
    return result

def get_weather(city):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)
    if response.status_code == 200:
        return f"The current weather in {city} is: {response.text}"
    else:
        return "Sorry, I couldn't fetch the weather information right now."

available_tools = {
    "get_weather": get_weather,
    "run_command": run_command
}

SYSTEM_PROMPT = """You are an expert AI assistant in resolving user queries using chain of thought.
You work on START, PLAN and OUTPUT steps. You need to first PLAN what needs to be done.The PLAN can be multiple steps.
Once you think enough PLAN has been done, you need to OUTPUT the final answer. You can also call a tool if required from 
the list of available tools. for every tool call wait from the observe step which is the output from the called tool.

Rules:
- Strictly follow the given JSON output format.
- Only run one step at a time.
- The sequence of steps is START (Where user gives an input), PLAN (That can be multiple steps) and OUTPUT (Final answer).

Output JSON Format:
{"step": "START" | "PLAN" | "OUTPUT" | "TOOL", "content" : "string", "tool": "string", "input": "string"}

Available Tools:
- get_weather(city: str) -> str : Fetches the current weather for the specified city
- run_command(cmd: str) -> int : Takes a system linux command and executes the command on user's system and returns
the output from the command.

Example:
START: Hey, what is the weather of delhi?
PLAN: {"step": "PLAN", "content": "Seems like the user is interested in getting weather of delhi in India"}
PLAN: {"step": "PLAN", "content": "Let's see if we have any available tool from the list of available tools to get weather information"}
PLAN: {"step": "PLAN", "content": "Great, we have get_weather tool available for this query"}
PLAN: {"step": "PLAN", "content": "I need to call get_weather tool for delhi as inoput for city"}
PLAN: {"step": "TOOL", "tool": "get_weather", "input"": "delhi"}
PLAN: {"step": "OBSERVE", "tool": "get_weather", "output"": "the temperature of delhi is 35 degree celsius with sunny weather"}
PLAN: {"step": "PLAN", "content": "Great, I have got the weather information for delhi"}
OUTPUT: {"step": "OUTPUT", "content": "The current weather in delhi is 35 degree celsius with sunny weather"}

"""

print("\n\n\n")

class MyOutputModel(BaseModel):
    step: str = Field(..., description="ID of the step.One of START, PLAN, TOOL, OUTPUT")
    content: Optional[str] = Field(None, description="Content of the step")
    tool: Optional[str] = Field(None, description="ID of Toolto be called")
    input: Optional[str] = Field(None, description="Input params for the tool")

messages_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

r = sr.Recognizer()  # Speech to Text
with sr.Microphone() as source:  # Mic access
    r.adjust_for_ambient_noise(source)  # cutting off background noise
    r.pause_threshold = 2  # if user pauses for 2 seconds, start recognizing

    while True:
        print("Speak Something...")
        audio = r.listen(source)  # listen to the source
        print("Processing Audio...(STT)")
        user_query = r.recognize_google(audio)  # using google's stt
        messages_history.append({"role": "user", "content": user_query})

        while True:
            response = client.chat.completions.parse(
                model="gpt-4.1",
                response_format=MyOutputModel,
                messages=messages_history,
            )

            raw_result = response.choices[0].message.content
            messages_history.append({"role": "assistant", "content": raw_result})
            import json
            # parsed_result = json.loads(raw_result)
            parsed_result = response.choices[0].message.parsed

            if parsed_result.step == "START":
                print(parsed_result.get("content"))
                continue

            if parsed_result.step == "PLAN":
                print(f"PLAN: {parsed_result.content}")
                continue

            if parsed_result.step == "TOOL":
                tool_name = parsed_result.tool
                tool_input = parsed_result.input
                print(f"{tool_name} ({tool_input})")
                tool_response = available_tools[tool_name](tool_input)
                messages_history.append({
                    "role": "developer",
                    "content": json.dumps(
                        {
                            "step": "OBSERVE",
                            "tool": tool_name,
                            "input": tool_input,
                            "output": tool_response
                        }
                    )
                })
                continue


            if parsed_result.step == "OUTPUT":
                print(f"OUTPUT: {parsed_result.content}")
                asyncio.run(tts(parsed_result.content))
                break

