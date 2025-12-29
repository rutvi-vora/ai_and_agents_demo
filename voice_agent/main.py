import speech_recognition as sr
import asyncio

from openai import OpenAI
from dotenv import load_dotenv

from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

load_dotenv()

client = OpenAI()
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

def main():
    r = sr.Recognizer() # Speech to Text

    with sr.Microphone() as source: # Mic access
        r.adjust_for_ambient_noise(source) # cutting off background noise
        r.pause_threshold = 2 # if user pauses for 2 seconds, start recognizing

        SYSTEM_PROMPT = (
            "You are an expert voice agent. You are given the transcipt of what user has said. You need to output as"
            "if you are a voice agent and whatever you speack will be converted to audio using AI and played back to user. ")
        messages = [ {"role": "system", "content": SYSTEM_PROMPT}]

        while True:
            print("Speak Something...")
            audio = r.listen(source) # listen to the source
            print("Processing Audio...(STT)")
            stt = r.recognize_google(audio) # using google's stt
            print("You Said: ", stt)

            messages.append({"role": "user", "content": stt})

            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages
            )

            ai_response = response.choices[0].message.content
            print(ai_response)
            asyncio.run(tts(ai_response))

main()