import networkx as nx
import re
from typing import List, Dict, Any

class CitationGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph(self, papers: List[Dict[str, Any]]):
        """
        Builds a directed graph from a list of papers.
        Each paper is a node. Edges represent citations.
        
        Args:
            papers: List of dicts containing 'id' (filename) and 'text' (content).
        """
        print("Building Citation Graph...")
        for paper in papers:
            paper_id = paper.get("filename")
            self.graph.add_node(paper_id)
            
            # Extract citations (naive regex approach for demo)
            # In a real system, we'd use robust citation extraction or Groq
            citations = self._extract_citations(paper.get("text", ""), papers)
            
            for cited_paper_id in citations:
                self.graph.add_edge(paper_id, cited_paper_id)
                
    def _extract_citations(self, text: str, all_papers: List[Dict[str, Any]]) -> List[str]:
        """
        Heuristic: Look for mentions of other paper filenames or titles in the text.
        """
        citations = []
        for other_paper in all_papers:
            other_id = other_paper.get("filename")
            # Skip self
            if other_id == "unknown" or text == other_paper.get("text"): 
                continue
                
            # Check if filename (minus ext) appears in text
            # Or check for [1], [2] style if we had a mapping.
            # For this "Giant System" demo, we'll assume papers reference each other by Title or significant keywords?
            # Actually, let's look for exact matches of standard citation patterns or the filenames if available.
            # Simplified: Check if other paper's filename (without pdf) is mentioned.
            simple_name = other_id.replace(".pdf", "")
            if simple_name in text:
                citations.append(other_id)
                
        return citations

    def calculate_pagerank(self) -> Dict[str, float]:
        """
        Calculates PageRank for all nodes in the graph.
        Returns a dict of {paper_id: score}.
        """
        if self.graph.number_of_nodes() == 0:
            return {}
            
        try:
            pagerank_scores = nx.pagerank(self.graph, alpha=0.85)
            sorted_scores = dict(sorted(pagerank_scores.items(), key=lambda item: item[1], reverse=True))
            return sorted_scores
        except Exception as e:
            print(f"Error calculating PageRank: {e}")
            return {}

    def get_classic_papers(self, top_n=5) -> List[tuple]:
        scores = self.calculate_pagerank()
        return list(scores.items())[:top_n]
