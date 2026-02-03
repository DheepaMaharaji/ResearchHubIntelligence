import os
import base64
import io
from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document
from groq import Groq
from pypdf import PdfReader
from PIL import Image

# New Imports for Semantic Chunking
from langchain_experimental.text_splitter import SemanticChunker
from backend.rag.embeddings import get_embedding_model

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def process_paper(pdf_path: str) -> List[Document]:
    """
    Process a PDF using pypdf to extract text and images.
    
    Chunking Strategy:
    - Text: Uses Semantic Chunking (via OpenAI Embeddings) to split based on meaning.
    - Images: Extracted per page and summarized using Groq.
    """
    print(f"Processing {pdf_path} (Semantic Chunking enabled)...")
    
    reader = PdfReader(pdf_path)
    final_documents = []
    
    paper_filename = os.path.basename(pdf_path)
    
    # --- 1. Accumulate Full Text & Handle Images ---
    full_text = ""
    
    for i, page in enumerate(reader.pages):
        page_num = i + 1
        
        # Accumulate text
        text = page.extract_text()
        if text:
            # We add a cleaner separator if needed, but simple concatenation 
            # with newline is usually fine for flow.
            full_text += f"{text}\n"

        # Extract Images (Keep per-page logic as images are distinct)
        try:
            for image_file_object in page.images:
                image_name = image_file_object.name
                image_data = image_file_object.data
                
                # Convert to base64 for Groq
                encoded_string = base64.b64encode(image_data).decode('utf-8')
                
                # Generate Summary
                try:
                    summary = summarize_image(encoded_string)
                    print(f"  Generated summary for image {image_name} on page {page_num}")
                except Exception as e:
                    print(f"  Failed to summarize image {image_name}: {e}")
                    summary = "Image processing failed."

                doc = Document(
                    page_content=f"IMAGE SUMMARY: {summary}",
                    metadata={
                        "source": pdf_path,
                        "filename": paper_filename,
                        "page": page_num,
                        "type": "image",
                        "image_name": image_name
                    }
                )
                final_documents.append(doc)
        except Exception as e:
             # Just warn, don't crash
             print(f"  Warning: Image extraction failed on page {page_num}: {e}")

    # --- 2. Apply Semantic Chunking to Text ---
    if full_text.strip():
        print("  Applying Semantic Chunking to text...")
        try:
            # Initialize Semantic Chunker with the same embedding model used for storage
            embedding_model = get_embedding_model()
            text_splitter = SemanticChunker(
                embedding_model,
                breakpoint_threshold_type="percentile" # or "standard_deviation", "interquartile"
            )
            
            # Split the full text
            chunks = text_splitter.create_documents([full_text])
            
            # Add metadata to chunks
            for i, chunk in enumerate(chunks):
                chunk.metadata = {
                    "source": pdf_path,
                    "filename": paper_filename,
                    "chunk_index": i,
                    "type": "text",
                    "strategy": "semantic"
                }
                final_documents.extend(chunks)
                
            print(f"  Created {len(chunks)} semantic chunks from text.")
            
        except Exception as e:
            print(f"  Semantic Chunking failed: {e}. Fallback to whole-doc.")
            # Fallback: one big chunk (or you could revert to page-level if you kept the logic)
            final_documents.append(Document(
                page_content=full_text,
                metadata={
                    "source": pdf_path,
                    "filename": paper_filename,
                    "type": "text",
                    "strategy": "fallback_full"
                }
            ))
    
    return final_documents


def summarize_image(base64_image: str) -> str:
    """
    Summarize an image using Groq's Vision models.
    """
    # Note: Ensure the model supports vision.
    # LLaVA v1.5 or Llama 3.2 Vision if available on Groq.
    # As of early 2025, Groq supports Llama-3.2-11b-vision-preview or similar.
    # We will use a flexible model ID.
    model = "meta-llama/llama-4-scout-17b-16e-instruct" 
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image from a research paper. Describe any charts, graphs, or tables in detail."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model=model,
            temperature=0.1
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        # Fallback or re-raise
        raise e
