import networkx as nx
import community.community_louvain as community_louvain 
# Note: python-louvain package imports as community
from typing import List, Dict, Any
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph() # Undirected graph for co-occurrence

    def extract_entities(self, text: str) -> List[str]:
        """
        Uses Groq to extract key scientific entities (materials, methods, concepts).
        """
        prompt = f"""
        Extract the top 5 key scientific entities (specific materials, algorithms, theories) from this text.
        Return ONLY a comma-separated list.
        
        Text: {text[:2000]}...
        """
        try:
            completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
            )
            entities_str = completion.choices[0].message.content
            # Clean up
            entities = [e.strip() for e in entities_str.split(",")]
            return entities
        except Exception as e:
            print(f"Entity extraction failed: {e}")
            return []

    def build_term_graph(self, papers: List[Dict[str, Any]]):
        """
        Builds a graph where Papers are specific nodes, and Entities are nodes.
        Edges connect Paper <-> Entity.
        Alternatively, for Community Detection on Papers, we can project to Paper <-> Paper based on shared entities.
        
        Let's do Paper <-> Paper (if they share entities).
        """
        print("Extracting entities and building Knowledge Graph...")
        
        paper_entities = {}
        
        for paper in papers:
            p_id = paper.get("filename")
            text = paper.get("text", "")
            entities = self.extract_entities(text)
            paper_entities[p_id] = set(entities)
            self.graph.add_node(p_id, type="paper")
            
        # Add edges based on Jaccard similarity of entities or just shared entities
        # Add edges based on Jaccard Similarity
        node_ids = list(paper_entities.keys())
        for i in range(len(node_ids)):
            for j in range(i+1, len(node_ids)):
                id1 = node_ids[i]
                id2 = node_ids[j]
                
                entities1 = paper_entities[id1]
                entities2 = paper_entities[id2]
                
                intersection = entities1.intersection(entities2)
                union = entities1.union(entities2)
                
                # Pruning Rule 1: Must share more than 1 entity to avoid noise
                if len(intersection) <= 1:
                    continue
                    
                # Calculate Jaccard Index
                if len(union) == 0:
                    jaccard = 0.0
                else:
                    jaccard = len(intersection) / len(union)
                    
                # Pruning Rule 2: Similarity Threshold
                if jaccard >= 0.2:
                    # Weight is the Jaccard Index (0.2 to 1.0)
                    self.graph.add_edge(id1, id2, weight=jaccard, entities=list(intersection))

    def detect_communities(self) -> Dict[int, List[str]]:
        """
        Detects communities using Louvain.
        Returns {community_id: [list of paper_ids]}
        """
        if self.graph.number_of_nodes() == 0:
            return {}
            
        try:
            partition = community_louvain.best_partition(self.graph)
            
            communities = {}
            for node, comm_id in partition.items():
                if comm_id not in communities:
                    communities[comm_id] = []
                communities[comm_id].append(node)
                
            return communities
        except Exception as e:
            print(f"Community detection failed: {e}")
            return {}
