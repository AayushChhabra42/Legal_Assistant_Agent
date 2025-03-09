# Semantic search functionality using FAISS

import faiss
import pickle
import json
import numpy as np
import os
from config import VECTOR_STORE_PATH, METADATA_PATH

def search_faiss(tokenizer, encoder, query, top_k=5, group_by_document=True):
    """
    Perform semantic search using FAISS vector index.
    
    Args:
        tokenizer: BERT tokenizer
        encoder: BERT model for encoding text
        query: String query to search for
        top_k: Number of top results to return
        group_by_document: Whether to group results by document and deduplicate
        
    Returns:
        List of document/chunk metadata for top matches
    """
    from embedding import embed_text
    
    # Load FAISS index
    index = faiss.read_index(VECTOR_STORE_PATH)
    
    # Load metadata
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    
    # Check if we're dealing with chunks
    is_chunked = "chunk_id" in metadata[0] if metadata else False
    
    # Embed query
    query_vector = embed_text(tokenizer, encoder, query).reshape(1, -1)
    
    # If using chunks and grouping by document, get more results initially
    search_k = top_k * 3 if is_chunked and group_by_document else top_k
    
    # Search for similar vectors
    distances, indices = index.search(query_vector, min(search_k, len(metadata)))
    
    # Get results
    results = [metadata[i] for i in indices[0] if i < len(metadata)]
    
    # Group by document if needed
    if is_chunked and group_by_document:
        # Group by parent document and take the best match from each
        grouped_results = {}
        for result in results:
            doc_id = result["parent_document"]
            if doc_id not in grouped_results or result["chunk_index"] < grouped_results[doc_id]["chunk_index"]:
                grouped_results[doc_id] = result
        
        # Convert back to list and limit to top_k
        results = list(grouped_results.values())[:top_k]
        
        # Try to load full document metadata if available
        doc_metadata_path = METADATA_PATH + ".documents"
        if os.path.exists(doc_metadata_path):
            with open(doc_metadata_path, "rb") as f:
                full_docs = pickle.load(f)
                
            # Create a mapping from filename to full document
            doc_map = {doc["filename"]: doc for doc in full_docs}
            
            # Enhance results with full document information
            for i, result in enumerate(results):
                doc_id = result["parent_document"]
                if doc_id in doc_map:
                    # Add full document info while preserving chunk reference
                    results[i] = {
                        **result,
                        "full_document": doc_map[doc_id]
                    }
    
    return results

def display_search_results(results):
    """Display search results in a readable format."""
    print(f"Found {len(results)} results:")
    
    # Check if we're dealing with chunks
    is_chunked = "chunk_id" in results[0] if results else False
    
    for i, item in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        
        if is_chunked:
            print(f"Document: {item['parent_document']}")
            print(f"Chunk: {item['chunk_index'] + 1} of {item['total_chunks']}")
            print(f"Chunk Text:{item['chunk_text']}")
            # print("\nChunk Preview:")
            # preview = item['chunk_text'][:200] + "..." if len(item['chunk_text']) > 200 else item['chunk_text']
            # print(preview)
            
            if 'full_document' in item:
                print(f"\nEntities in document: {json.dumps(item['full_document'].get('entities', []), indent=2)}")
            else:
                print(f"\nEntities in chunk: {json.dumps(item.get('entities', []), indent=2)}")
        else:
            print(f"Filename: {item['filename']}")
            print(f"Entities: {json.dumps(item.get('entities', []), indent=2)}")