# Standalone script for running semantic searches

from models import load_models
from search import search_faiss, display_search_results
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Semantic search for legal documents")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--no-grouping", action="store_true", help="Don't group results by document")
    args = parser.parse_args()
    
    # Load models
    print("Loading models...")
    tokenizer, encoder, _ = load_models()  # We don't need the NER model for searching
    
    # Interactive search loop
    while True:
        query = input("\nEnter your search query (or 'q' to quit): ")
        if query.lower() == 'q':
            break
        
        # Perform search
        retrieved_docs = search_faiss(
            tokenizer, 
            encoder, 
            query, 
            top_k=args.top_k,
            group_by_document=not args.no_grouping
        )
        
        # Display results
        display_search_results(retrieved_docs)

if __name__ == "__main__":
    main()