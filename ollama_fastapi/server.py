from fastapi import FastAPI, Body
from ollama import Client

client = Client(
    host="http://localhost:11434"
)

app = FastAPI()

@app.post("/chat")
def chat(message: str = Body(..., description="The message")) -> str:
    response = client.chat(
        model="gemma:2b",
        messages=[
            {"role": "user", "content": message},
        ],
    )
    return response.message.content