The Research Intelligence Hub is an advanced Retrieval-Augmented Generation (RAG) platform designed to transform static PDF libraries into dynamic, interconnected knowledge graphs. Unlike standard RAG systems that rely solely on semantic similarity, this project introduces a Trust & Importance Layer—leveraging PageRank and community detection to surface not just "relevant" information, but the most authoritative insights in a field.


1. Chunking Upgrade: Semantic vs. Page-Level

Problem: PDF pages are arbitrary boundaries. A sentence starting on Page 1 and ending on Page 2 was split into two separate vectors, losing context. Headers and footers added noise.
Solution: Implemented Semantic Chunking using langchain-experimental.
How it works:
Extracts the entire text of the paper as a single stream.
Uses an embedding model helper to scan across the text.
Detects "semantic break points" (sudden changes in embedding similarity) which correspond to topic shifts (e.g., Introduction -> Methods).
Splits the text at these meaningful boundaries.
2. Model Migration: OpenAI -> Groq (Llama)

Change: Migrated to Groq for LLM and Vision tasks to leverage high-speed inference with top-tier Open Source models (Llama 3/4).
Vision Model: Upgraded to meta-llama/llama-4-scout-17b-16e-instruct for summarizing charts and images found in papers.
Text Model: Upgraded retrieval and query expansion to use llama-3.3-70b-versatile.
3. Embedding Migration: Local Embeddings

Problem: Groq does not offer an embedding API.
Solution: Switched to HuggingFace Local Embeddings (sentence-transformers/all-MiniLM-L6-v2).
Benefit:
Free: Runs on your local machine.
Standard: 384-dimensional vectors are industry standard for efficient retrieval.
4. Verification
We verified the pipeline by ingesting the famous paper "Attention Is All You Need".

Images: 3 images were extracted and summarized by Groq.
Text: The paper was semantically split into key concept chunks.
Database: The research-papers index was automatically created in Pinecone and populated with 403 vectors.
5. Reranker Migration: Open Source
Original State: Cohere Rerank API (Paid/External).

Change: Replaced with Local Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2).
Benefit:
Privacy: No data sent to third-party reranking API.
Free: Runs entirely on your local infrastructure.
Performance: High accuracy reranking comparable to commercial solutions.
6. Trust & Importance Layer (Knowledge Graph)
Feature: Graph-Enhanced Retrieval.

Offline: The backend.analyze module calculates PageRank (Authority) and Use Louvain Community Detection (Topic Clusters). These metrics are synced to Pinecone metadata.
Online: The retrieval engine uses a weighted scoring formula: Score = (α * Semantic) + ((1-alpha) * Authority) Additionally, a Community Multiplier is applied:
1.2x Boost: If the paper belongs to the dominant community (Mode) of the result set.
0.8x Penalty: If the paper is an outlier from a different community.
Adaptive: The alpha parameter adjusts based on user expertise:
Beginner: High Trust in Authority (PageRank).
Expert: High Trust in Specificity (Semantic Match). Graph Construction Pruning:
Metric: Jaccard Similarity (Intersection / Union).
Constraints: Edges are only formed if papers share >1 entity AND have a Score >= 0.2.
