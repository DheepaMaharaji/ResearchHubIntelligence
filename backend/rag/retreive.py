import os
import cohere
from langchain_core.documents import Document
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from backend.config import COHERE_API_KEY


def get_multi_query_retriever(user_query: str):
    """
    Expands a single user query into 3 variations for better retrieval.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)  # Low temperature for precision

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
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    all_docs = []
    for q in queries:
        # Retrieve top 5 per query (before reranking)
        docs = vectorstore.similarity_search(q, k=5, namespace=namespace)
        all_docs.extend(docs)

    # Deduplicate documents based on content or metadata ID
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    return unique_docs


co = cohere.Client(COHERE_API_KEY)


def rerank_documents(query: str, documents: List[Document], top_n: int = 5):
    """
    Implements the 'Ranking' green block.
    Uses Cohere to re-order docs based on true semantic relevance.
    """
    if not documents:
        return []

    # Prepare docs for Cohere (text only)
    doc_texts = [doc.page_content for doc in documents]

    # Rerank call
    results = co.rerank(
        query=query,
        documents=doc_texts,
        top_n=top_n,
        model="rerank-english-v3.0"
    )

    # Re-map results back to LangChain Document objects
    final_docs = []
    for res in results.results:
        original_doc = documents[res.index]
        original_doc.metadata["relevance_score"] = res.relevance_score
        final_docs.append(original_doc)

    return final_docs