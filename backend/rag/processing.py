import os
import base64
import io
from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document
from groq import Groq
from pypdf import PdfReader
from PIL import Image

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def process_paper(pdf_path: str) -> List[Document]:
    """
    Process a PDF using pypdf (pure python) to avoid system dependency issues.
    Extracts text and images.
    Summarizes images using Groq.
    """
    print(f"Processing {pdf_path} with pypdf...")
    
    reader = PdfReader(pdf_path)
    documents = []
    
    paper_filename = os.path.basename(pdf_path)
    
    for i, page in enumerate(reader.pages):
        page_num = i + 1
        
        # 1. Extract Text
        text = page.extract_text()
        if text:
            doc = Document(
                page_content=text,
                metadata={
                    "source": pdf_path,
                    "filename": paper_filename,
                    "page": page_num,
                    "type": "text"
                }
            )
            documents.append(doc)
            
        # 2. Extract Images
        # pypdf >= 3.0.0
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
                documents.append(doc)
        except Exception as e:
             print(f"  Warning: Image extraction failed on page {page_num}: {e}")

    return documents


def summarize_image(base64_image: str) -> str:
    """
    Summarize an image using Groq's Vision models.
    """
    # Note: Ensure the model supports vision.
    # LLaVA v1.5 or Llama 3.2 Vision if available on Groq.
    # As of early 2025, Groq supports Llama-3.2-11b-vision-preview or similar.
    # We will use a flexible model ID.
    model = "llama-3.2-11b-vision-preview" 
    
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
