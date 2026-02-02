import os
from langchain_openai import OpenAIEmbeddings

def get_embedding_model():
    """
    Returns the embedding model to use for vectorization.
    """
    # Ensure OPENAI_API_KEY is in env
    return OpenAIEmbeddings(model="text-embedding-3-small")
