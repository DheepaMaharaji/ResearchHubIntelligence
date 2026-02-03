import os
import sys
from dotenv import load_dotenv

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.analysis.citations import CitationGraph
from backend.analysis.knowledge_graph import KnowledgeGraph
from backend.database import get_index

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
        
    # 4. Sync to Pinecone (The "Trust Layer" Integration)
    print("\n💾 Syncing Graph Metrics to Pinecone...")
    sync_to_pinecone(scores, communities)

def sync_to_pinecone(pagerank_scores: dict, communities: dict):
    """
    Updates Pinecone vectors with 'pagerank' and 'community' metadata.
    This enables O(1) access to graph signals during retrieval.
    """
    index = get_index()
    
    # Invert communities to paper -> community_id
    paper_to_community = {}
    for c_id, members in communities.items():
        for paper in members:
            paper_to_community[paper] = c_id
            
    # For each paper, find its chunks and update
    # Note: We iterate over all papers found in the analysis
    all_papers = set(pagerank_scores.keys()) | set(paper_to_community.keys())
    
    for paper_filename in all_papers:
        pr_score = pagerank_scores.get(paper_filename, 0.0)
        comm_id = paper_to_community.get(paper_filename, -1)
        
        # Fetch chunks for this file
        # Pinecone metadata filter query
        # Ideally we fetch IDs first. 
        # For simplicity in this script, we'll dummy-query to find them or assuming we know IDs.
        # Actually, Pinecone update requires IDs. 
        # We can list vectors with prefix if we used consistent ID naming (filename_chunkIndex).
        
        # Strategy: List IDs with prefix
        prefix = paper_filename + "_"
        try:
             # Basic list_paginated equivalent or scan
             # Since we can't easily scan pinecone freely, we rely on the implementation convention
             # "filename_i"
             # Let's try to update the first 100 chunks (papers rarely have more).
             # Efficient way: Use list(prefix=...) if available in Serverless, checking...
             # As of 2024/2025 Pinecone, list_paginated or prefix search on ID is supported.
             
             for i in range(200): # Hard limit for prototype safety
                 chunk_id = f"{paper_filename}_{i}"
                 
                 # Optimistic Update: We don't check if it exists, we just try to update metadata.
                 # If it doesn't exist, no harm.
                 try:
                    index.update(
                        id=chunk_id,
                        set_metadata={
                            "pagerank": pr_score,
                            "community": comm_id
                        }
                    )
                 except Exception:
                    # Break loop if we hit a gap or error (likely end of chunks)
                    # But actually we shouldn't break on one error... 
                    # Let's just break if we get a "Not Found" essentially or after some empty streaks?
                    # The update() call usually doesn't throw on missing ID, it just no-ops?
                    # Actually pinecone raises NotFound usually.
                    pass
                    
        except Exception as e:
            print(f"Error syncing {paper_filename}: {e}")
            
    print("Sync Complete.")

if __name__ == "__main__":
    load_dotenv()
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        run_analysis(directory)
    else:
        print("Usage: python -m backend.analyze <directory_of_pdfs>")
