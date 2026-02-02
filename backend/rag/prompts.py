from langchain.prompts import ChatPromptTemplate

# 1. The Research Synthesizer
SYNTHESIZER_PROMPT = """
You are a Research Intelligence AI. Using the papers provided below, answer the question.
If the papers disagree, highlight the controversy. If the answer isn't in the docs, say you don't know.

Context: {context}
Question: {question}
Answer:"""

# 2. The Hallucination Grader (The "Judge")
GRADER_PROMPT = """
You are an auditor. You will be given a RESEARCH PAPER CHUNK and an AI-GENERATED ANSWER.
Your goal is to give a binary score 'yes' or 'no'.
'yes' means the answer is strictly grounded in the chunk.
'no' means the answer contains information not found in the chunk (hallucination).

Chunk: {context}
Answer: {answer}
Grounded Score (yes/no):"""