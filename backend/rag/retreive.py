import os
from langchain_core.documents import Document
from typing import List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from backend.database import get_index
from backend.rag.embeddings import get_embedding_model

# Initialize the Cross-Encoder model for reranking
# ms-marco-MiniLM-L-6-v2 is a standard, lightweight reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

class MultiQuery(BaseModel):
    queries: List[str] = Field(description="List of 3 distinct research queries")

def get_multi_query_retriever(user_query: str):
    """
    Expands a single user query into 3 variations for better retrieval.
    """
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0,
        api_key=os.environ.get("GROQ_API_KEY")
    )

    # Advanced Research Prompting
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research assistant. Your task is to generate 3 different versions "
                   "of the user's query to retrieve relevant academic papers. Focus on "
                   "scientific terminology and different perspectives of the same topic."),
        ("user", "{question}")
    ])

    # Chain: Prompt -> LLM -> JSON Parser
    chain = prompt | llm.with_structured_output(MultiQuery)
    result = chain.invoke({"question": user_query})

    return result.queries

def retrieve_documents(queries: List[str], index_name: str, namespace: str = "research-papers"):
    """
    Implements the 'VectorDB' retrieval.
    Searches Pinecone for each expanded query and aggregates results.
    """
    embedding_model = get_embedding_model()
    index = get_index()

    all_docs = []
    for q in queries:
        # Generate embedding for the query
        query_vec = embedding_model.embed_query(q)
        
        # Retrieve top 5 per query (before reranking)
        try:
            results = index.query(
                vector=query_vec,
                top_k=5,
                namespace=namespace,
                include_metadata=True
            )
            
            for match in results.matches:
                # Reconstruct LangChain Document
                text_content = match.metadata.get("text", "")
                doc = Document(page_content=text_content, metadata=match.metadata)
                all_docs.append(doc)
                
        except Exception as e:
            print(f"Error querying Pinecone: {e}")

    # Deduplicate documents based on content
    seen = set()
    unique_docs = []
    for doc in all_docs:
        # Use page_content as unique key
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    return unique_docs


def rerank_documents(query: str, documents: List[Document], top_n: int = 5, alpha: float = 0.5):
    """
    Implements the 'Ranking' green block.
    Uses Local Cross-Encoder for semantic relevance + Knowledge Graph for authority.
    
    Formula: Final_Score = (alpha * Semantic_Norm) + ((1-alpha) * PageRank)
    """
    if not documents:
        return []

    # Prepare pairs for CrossEncoder [query, document_text]
    pairs = [[query, doc.page_content] for doc in documents]

    # Predict scores (returns a list of floats)
    # These scores are unbounded logits (e.g. -10 to +10)
    raw_scores = reranker.predict(pairs)
    
    # Normalize Semantic Scores to [0, 1] using Min-Max
    # Avoid division by zero
    if len(raw_scores) > 1:
        min_s = min(raw_scores)
        max_s = max(raw_scores)
        range_s = max_s - min_s
        if range_s == 0:
            norm_scores = [1.0] * len(raw_scores)
        else:
            norm_scores = [(s - min_s) / range_s for s in raw_scores]
    else:
        norm_scores = [1.0]

    # Combine with Graph Signals
    
    # 1. Calculate Community Mode (Dominant Topic)
    communities = [int(doc.metadata.get("community", -1)) for doc in documents]
    # Filter out -1 (no community) before finding mode, unless all are -1
    valid_comms = [c for c in communities if c != -1]
    
    dominant_community = -1
    if valid_comms:
        from collections import Counter
        counts = Counter(valid_comms)
        dominant_community = counts.most_common(1)[0][0] # Get the most frequent community ID

    for i, doc in enumerate(documents):
        # Fetch PageRank from metadata (pushed by analyze.py)
        # Default to 0.0 if not analyzed yet
        pagerank = float(doc.metadata.get("pagerank", 0.0))
        community = int(doc.metadata.get("community", -1))
        
        semantic_score = norm_scores[i]
        
        # Base Score (Alpha-Weighted)
        base_score = (alpha * semantic_score) + ((1 - alpha) * pagerank)
        
        # Community Multiplier
        if dominant_community != -1:
            if community == dominant_community:
                multiplier = 1.2 # Boost dominant topic
            else:
                multiplier = 0.8 # Penalize outliers
        else:
            multiplier = 1.0 # No consensus, no boost
            
        final_score = base_score * multiplier
        
        doc.metadata["relevance_score"] = final_score
        doc.metadata["semantic_raw"] = float(raw_scores[i])
        doc.metadata["pagerank"] = pagerank
        doc.metadata["community"] = community
        doc.metadata["community_boost"] = multiplier # Debug info

    # Sort by score descending
    documents.sort(key=lambda x: x.metadata["relevance_score"], reverse=True)

    return documents[:top_n]