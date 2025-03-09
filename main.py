# Entry point that runs the pipeline

from models import load_models
from pipeline import process_documents
from search import search_faiss, display_search_results
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Legal document processing and search")
    parser.add_argument("--no-chunking", action="store_true", help="Disable document chunking")
    parser.add_argument("--chunk-size", type=int, default=500, help="Size of document chunks (in words)")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap between chunks (in words)")
    parser.add_argument("--search-only", action="store_true", help="Skip processing and only run search")
    args = parser.parse_args()
    
    # Load models
    print("Loading models...")
    tokenizer, encoder, ner_model = load_models()
    
    if not args.search_only:
        # Process documents
        print("Starting document processing...")
        process_documents(
            tokenizer, 
            encoder, 
            ner_model,
            use_chunking=not args.no_chunking,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
    
    # Example of search functionality
    while True:
        search_query = input("\nEnter a search query (or press Enter to exit): ")
        if not search_query:
            break
            
        print(f"\nSearching for: '{search_query}'")
        search_results = search_faiss(tokenizer, encoder, search_query)
        display_search_results(search_results)


if __name__ == "__main__":
    main()