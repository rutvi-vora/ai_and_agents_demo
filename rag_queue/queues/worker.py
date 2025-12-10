from dotenv import load_dotenv
load_dotenv()

import os

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/"
)

# embedding_model = OpenAIEmbeddings(
#     model="text-embedding-3-small",
# )
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_rag",
    embedding=embedding_model,
)

def process_query(query: str):
    print("Searching chunks", query)
    search_results = vector_db.similarity_search(query)

    context = "\n\n\n".join([f"Page Content: {result.page_content}\n"
                             f"Page Number: {result.metadata['page_label']}\n"
                             f"File Location: {result.metadata['source']}"
                             for result in search_results])

    SYSTEM_PROMPT = f"""
        You are an helpful assistant who answers user query based on the available context
        retrieved from a pdf file along with page_contents and page number.

        You should also ans the user based on the following context and navigate the user
        to open the right page number to know more.

        Context:
        {context}
    """

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
    )

    return response.choices[0].message.content
