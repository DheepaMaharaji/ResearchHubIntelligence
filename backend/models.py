from pydantic import BaseModel
from typing import List

class ChatRequest(BaseModel):
    message: str
    category: str
    locked: bool = False

class Citation(BaseModel):
    title: str
    source: str

class ChatResponse(BaseModel):
    answer: str
    confidence: str
    citations: List[Citation]
    off_topic: bool

# Define the structure for our multi-query output
class MultiQuery(BaseModel):
    queries: List[str] = Field(description="A list of 3 generated research-focused queries.")