import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from backend.rag.retreive import rerank_documents

def test_reranker():
    print("Testing Local Reranker (CrossEncoder)...")
    
    query = "What creates the magnetic field?"
    
    docs = [
        Document(page_content="The Earth's magnetic field is created by the movement of molten iron in the core.", metadata={"id": 1}),
        Document(page_content="Apples are red and delicious fruits aimed for eating.", metadata={"id": 2}),
        Document(page_content="Magnets have north and south poles.", metadata={"id": 3}),
    ]
    
    print(f"Query: {query}")
    print("Reranking 3 documents...")
    
    ranked = rerank_documents(query, docs, top_n=2)
    
    print("\nTop Results:")
    for i, doc in enumerate(ranked):
        print(f"{i+1}. Score: {doc.metadata['relevance_score']:.4f} - Content: {doc.page_content}")
        
    # Validation
    assert ranked[0].metadata["id"] == 1, "Top result should be about Earth's magnetic field"
    print("\nSUCCESS: Reranker correctly identified the most relevant document.")

if __name__ == "__main__":
    test_reranker()
