# FAISS vector store operations

import faiss
import numpy as np
import pickle
import os
from config import VECTOR_STORE_PATH, METADATA_PATH

def store_documents_in_faiss(documents, chunks=None):
    """
    Stores document embeddings in FAISS vector store.
    
    Args:
        documents: List of document dictionaries with vectors
        chunks: List of chunk dictionaries with vectors (if using chunking)
    """
    # If using chunking, store chunks instead of full documents
    items_to_store = chunks if chunks is not None else documents
    
    if not items_to_store:
        print("No items to store in vector database")
        return
        
    dimension = len(items_to_store[0]["vector"])  # Get vector size
    index = faiss.IndexFlatL2(dimension)  # Create FAISS index
    metadata = []  # Stores metadata separately
    
    vectors = np.array([item["vector"] for item in items_to_store])
    index.add(vectors)  # Add all vectors to FAISS at once
    
    # Store metadata for each item (either document or chunk)
    for item in items_to_store:
        # Create a copy of the item without the vector to save space
        meta_item = {k: v for k, v in item.items() if k != "vector"}
        metadata.append(meta_item)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH) if os.path.dirname(VECTOR_STORE_PATH) else '.', exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, VECTOR_STORE_PATH)
    
    # Save metadata as a pickle file
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    
    # If using chunking, also store original documents separately
    if chunks is not None:
        with open(METADATA_PATH + ".documents", "wb") as f:
            pickle.dump([{k: v for k, v in doc.items() if k != "vector"} for doc in documents], f)
    
    print(f"Stored {len(items_to_store)} items in vector database")