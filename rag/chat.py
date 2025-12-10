from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()
client = OpenAI()

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
)
vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_rag",
    embedding=embedding_model,
)

user_query = input("Ask something:")

search_results = vector_db.similarity_search(user_query)

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
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
)

print(response.choices[0].message.content)