from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import ollama
import os

def load_config():
    """Load environment variables from .env"""
    load_dotenv()
    return {"HF_TOKEN": os.getenv("HF_TOKEN")}


def call_llm_offline(user_query):
    """Call the RAG-based system with an open-source LLM via Ollama (Offline)"""

    messages = [{"role": "user", "content": user_query}]

    response = ollama.chat(
        model="llama3.2",  # Change to "llama2" or another downloaded model if needed
        messages=messages
    )

    return response["message"]["content"]


# def call_llm(user_query):
#     """Call the RAG-based system with a user query"""
#     config = load_config()
#
#     client = InferenceClient(
#         provider="together",
#         api_key=config["HF_TOKEN"]
#     )
#
#     messages = [{"role": "user", "content": user_query}]
#
#     completion = client.chat.completions.create(
#         model="mistralai/Mistral-Small-24B-Instruct-2501",
#         messages=messages,
#         max_tokens=500,
#     )
#     return completion.choices[0].message

if __name__ == "__main__":
    query = "What is the capital of France?"
    response = call_llm_offline(query)
    print(response)
