import os
import argparse
import sys
from dotenv import load_dotenv

# Add the project root to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag.processing import process_paper
from backend.rag.database import upsert_documents

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Ingest research papers into the RAG system.")
    parser.add_argument("--file", type=str, help="Path to a specific PDF file to ingest.")
    parser.add_argument("--dir", type=str, help="Directory containing PDF files to ingest.")
    
    args = parser.parse_args()
    
    files_to_process = []
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} does not exist.")
            return
        files_to_process.append(args.file)
        
    elif args.dir:
        if not os.path.exists(args.dir):
            print(f"Error: Directory {args.dir} does not exist.")
            return
        for root, _, files in os.walk(args.dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    files_to_process.append(os.path.join(root, file))
    else:
        print("Please provide --file or --dir argument.")
        return

    print(f"Found {len(files_to_process)} files to process.")
    
    for file_path in files_to_process:
        try:
            print(f"Starting ingestion for: {file_path}")
            documents = process_paper(file_path)
            if documents:
                upsert_documents(documents)
            else:
                print(f"No content extracted from {file_path}")
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
