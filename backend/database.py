import os
import time
from typing import List
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

def get_index():
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME environment variable not set")
    
    existing_indexes = [i.name for i in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Index '{index_name}' does not exist. Creating...")
        pc.create_index(
            name=index_name,
            dimension=384, # sentence-transformers/all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        # Wait for index to be ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
            
    return pc.Index(index_name)

def upsert_documents(documents: List[Document]):
    """
    Upsert a list of LangChain Documents to Pinecone.
    Handles vector generation via the embedding model (external call or internal).
    Actually, LangChain's Pinecone integration is easier, but standard Pinecone client offers more control.
    We will use the standard client for clarity or LangChain's vectorstore.
    
    Let's use LangChain's Pinecone vectorstore for simplicity if we can, 
    but for "Giant System" connectivity, manual control over ID and metadata might be better.
    
    Implementation:
    Generate embeddings -> Batch Upsert to Pinecone.
    """
    from backend.rag.embeddings import get_embedding_model
    
    print(f"Upserting {len(documents)} documents to Pinecone...")
    
    embedding_model = get_embedding_model()
    index = get_index()
    
    # 1. Generate Embeddings
    texts = [doc.page_content for doc in documents]
    # Check for empty content
    valid_docs = [d for d in documents if d.page_content.strip()]
    if not valid_docs:
        print("No valid documents to upsert.")
        return

    texts = [d.page_content for d in valid_docs]
    ids = [f"{d.metadata.get('filename')}_{i}" for i, d in enumerate(valid_docs)]
    
    print("Generating embeddings...")
    embeddings = embedding_model.embed_documents(texts)
    
    # 2. Prepare Vectors
    vectors = []
    for i, doc in enumerate(valid_docs):
        # Clean metadata: Pinecone only allows str, int, float, bool, list[str]
        clean_metadata = {}
        for k, v in doc.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_metadata[k] = v
            elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                clean_metadata[k] = v
            else:
                clean_metadata[k] = str(v) # Fallback to stringified
        
        vectors.append({
            "id": ids[i],
            "values": embeddings[i],
            "metadata": {
                **clean_metadata,
                "text": doc.page_content # Metadata often needs the text for retrieval context
            }
        })
    
    # 3. Batch Upsert
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        print(f"Upserting batch {i} to {i+len(batch)}...")
        index.upsert(vectors=batch)
    
    print("Upsert complete.")
