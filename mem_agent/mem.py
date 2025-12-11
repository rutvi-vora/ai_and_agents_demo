import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from mem0 import Memory

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# config = {
#     "version": "v1.1",
#     "embedder": {
#         "provider": "openai",
#         "config": {
#             "api_key": OPENAI_API_KEY,
#             "model": "text-embedding-3-small"
#         }
#     },
#     "llm": {
#         "provider": "openai",
#         "config": {
#             "api_key": OPENAI_API_KEY,
#             "model": "gpt-4.1"
#         }
#     },
#     "vector_store": {
#         "provider": "qdrant",
#         "config": {
#             "host": "localhost",
#             "port": 6333
#         }
#     }
# }

# from langchain_community.embeddings import HuggingFaceEmbeddings
#
# embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
embedding_model_name = "BAAI/bge-small-en"

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": embedding_model_name
        }
    },
    "llm": {
        "provider": "gemini",
        "config": {
            "api_key": GEMINI_API_KEY,
            "model": "gemini-2.0-flash"   # or gemini-2.5-flash
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333
        }
    }
}

mem_client = Memory.from_config(config)

while True:
    user_query = input("Ask:")

    search_memory = mem_client.search(
        query=user_query,
        user_id="Garg"
    )

    memories  = [
        f"ID: {mem.get('id')}\nMemory:{mem.get('memory')}" for mem in search_memory.get("results")
    ]

    SYSTEM_PROMPT = f"""
        Here is the context about the user: 
        {json.dumps(memories)}
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_query
            }
        ]
    )
    ai_response = response.choices[0].message.content

    mem_client.add(
        user_id="Garg",
        messages=[
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": ai_response}
        ]
    )