import os
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    """
    Returns the embedding model to use for vectorization.
    Uses local HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2) 
    to be compatible with Groq (which doesn't have an embedding API).
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
