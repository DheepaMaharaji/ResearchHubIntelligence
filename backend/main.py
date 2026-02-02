from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.rag.retreive import get_multi_query_retriever, retrieve_documents, rerank_documents
from backend.rag.prompts import SYNTHESIZER_PROMPT, GRADER_PROMPT
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os

app = FastAPI(title="Interconnected Research Intelligence Hub")


class ChatRequest(BaseModel):
    message: str


@app.post("/query")
async def research_query(request: ChatRequest):
    user_query = request.message
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 1. Query Translation (Multi-Query)
    # Block: Query Translation (Red) from your diagram
    expanded_queries = get_multi_query_retriever(user_query)

    # 2. Retrieval & Reranking
    # Block: Retrieval (Green) from your diagram
    initial_docs = retrieve_documents(expanded_queries, os.getenv("PINECONE_INDEX_NAME"))
    top_docs = rerank_documents(user_query, initial_docs, top_n=5)

    if not top_docs:
        return {"answer": "I couldn't find any relevant research papers on this topic.", "sources": []}

    # 3. Generation
    # Block: Generation (Purple) from your diagram
    context_text = "\n\n".join([doc.page_content for doc in top_docs])
    gen_prompt = ChatPromptTemplate.from_template(SYNTHESIZER_PROMPT)
    chain = gen_prompt | llm
    answer = chain.invoke({"context": context_text, "question": user_query}).content

    # 4. Hallucination Grading (Self-RAG)
    # Block: Refinement/CRAG (Green) from your diagram
    grade_prompt = ChatPromptTemplate.from_template(GRADER_PROMPT)
    grader_chain = grade_prompt | llm
    grade_result = grader_chain.invoke({"context": context_text, "answer": answer}).content

    # Logic: If the answer isn't grounded, we return a warning or re-try
    is_grounded = "yes" in grade_result.lower()

    return {
        "answer": answer if is_grounded else "The generated answer could not be verified against the sources.",
        "is_grounded": is_grounded,
        "sources": [{"content": d.page_content, "metadata": d.metadata} for d in top_docs]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)