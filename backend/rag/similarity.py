from pydantic import BaseModel
from typing import List, Optional
# from langchain_openai import ChatOpenAI (Unused)
# Replaced by ChatGroq in other files, but here it is unused.
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from backend.rag.prompts import MULTI_QUERY_PROMPT
from backend.rag.embeddings import get_embedding_model
from backend.rag.database import get_index

# ... (Existing imports typically) ... 

def find_related_papers(paper_text: str, top_n: int = 5) -> List[Document]:
    """
    Finds semantically similar papers/chunks using content-based filtering (KNN).
    
    Args:
        paper_text: The text content of the source paper/chunk.
        top_n: Number of related items to retrieve.
    """
    embedding_model = get_embedding_model()
    index = get_index()
    
    # Generate embedding for the query text
    query_vector = embedding_model.embed_query(paper_text)
    
    # Query Pinecone
    results = index.query(
        vector=query_vector,
        top_k=top_n + 1, # Fetch +1 because top 1 is likely the paper itself
        include_metadata=True
    )
    
    related_docs = []
    for match in results.matches:
        # Filter out self-matches if needed (based on exact score or ID)
        if match.score > 0.999: # Likely the same doc
             continue
             
        doc = Document(
            page_content=match.metadata.get("text", ""),
            metadata=match.metadata
        )
        related_docs.append(doc)
        
    return related_docs[:top_n]

# ... we should keep the existing retrieve functions, but this helper is additive.
# For modifying existing file, we will use multi_replace.
