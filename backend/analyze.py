import os
import sys
from dotenv import load_dotenv

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.analysis.citations import CitationGraph
from backend.analysis.knowledge_graph import KnowledgeGraph
from backend.rag.database import get_index

def fetch_all_papers_metadata():
    """
    Fetches all papers from Pinecone to build the graph.
    Note: Pinecone doesn't support "dump all" efficiently. 
    We typically need to query or iterate. 
    For this prototype, we'll assume we can list them or have a local registry.
    
    Workaround: We'll list files in the `data/papers` (or whatever input dir) 
    and fetch their specific vectors/metadata by ID from Pinecone 
    OR just re-read the text locally for graph building if easier.
    
    Let's rely on local text reading for the GRAPH building, but update Pinecone metadata.
    """
    papers = []
    # Hack: Scan the local 'data/papers' if we stored them there, or just use arguments.
    # Ideally, we should have a `sqlite` or similar metadata store.
    # We will assume user passes a directory to analyze.
    return papers

def run_analysis(paper_dir: str):
    print(f"Analyzing papers in {paper_dir}...")
    
    # 1. Load Papers (Mocking the fetch from DB by reading files)
    papers = []
    for root, _, files in os.walk(paper_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                path = os.path.join(root, file)
                # We need text. We could re-parse or rely on text extraction.
                # For speed, let's use pypdf just to get text for analysis
                from pypdf import PdfReader
                try:
                    text = ""
                    reader = PdfReader(path)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    papers.append({"filename": file, "text": text})
                except:
                    pass

    if not papers:
        print("No papers found to analyze.")
        return

    # 2. Citation Graph & PageRank
    cg = CitationGraph()
    cg.build_graph(papers)
    scores = cg.calculate_pagerank()
    
    print("\n--- Research Intelligence Report ---\n")
    print("🏆 Top Influential Papers (PageRank):")
    for paper, score in cg.get_classic_papers():
        print(f"  - {paper}: {score:.4f}")

    # 3. Knowledge Graph & Communities
    kg = KnowledgeGraph()
    kg.build_term_graph(papers)
    communities = kg.detect_communities()
    
    print("\n🧩 Research Communities (Clusters):")
    for c_id, members in communities.items():
        print(f"  Community {c_id}: {members}")
        # Could analyze shared keywords here

if __name__ == "__main__":
    load_dotenv()
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        run_analysis(directory)
    else:
        print("Usage: python -m backend.analyze <directory_of_pdfs>")
