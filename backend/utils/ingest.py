import os
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyMuPDFLoader


def ingest_research_paper(file_path: str):
    # 1. Load PDF accurately
    loader = PyMuPDFLoader(file_path)
    data = loader.load()

    # 2. Initialize Semantic Chunker
    # 'percentile' means it splits at the top 5% of largest semantic shifts
    embeddings = OpenAIEmbeddings()
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )

    # 3. Create Chunks
    # This automatically finds logical breaks in the academic text
    docs = text_splitter.split_documents(data)

    # 4. Enrich Metadata (Crucial for Interconnectedness)
    for doc in docs:
        doc.metadata["source_file"] = os.path.basename(file_path)

    return docs