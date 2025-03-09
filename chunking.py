# Text chunking functionality

import re
from typing import List, Dict, Any

def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into chunks of approximately chunk_size tokens with specified overlap.
    
    Args:
        text: The text to split into chunks
        chunk_size: Target size of each chunk in words
        overlap: Number of words to overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Simple splitting by paragraphs first
    paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        # Approximate paragraph size by counting words
        words = paragraph.split()
        paragraph_size = len(words)
        
        # If adding this paragraph would exceed chunk size, start a new chunk
        if current_size + paragraph_size > chunk_size and current_size > 0:
            # Add the current chunk to our list of chunks
            chunks.append(' '.join(current_chunk))
            
            # Start a new chunk, keeping the overlap from the previous chunk
            if overlap > 0 and current_size > overlap:
                # Calculate how many words to keep for overlap
                overlap_text = ' '.join(current_chunk[-overlap:])
                current_chunk = [overlap_text]
                current_size = overlap
            else:
                current_chunk = []
                current_size = 0
        
        # Add the paragraph to the current chunk
        current_chunk.append(paragraph)
        current_size += paragraph_size
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def create_chunk_metadata(doc_metadata: Dict[str, Any], chunk_index: int, total_chunks: int, chunk_text: str) -> Dict[str, Any]:
    """
    Create metadata for a chunk that references the original document.
    
    Args:
        doc_metadata: Original document metadata
        chunk_index: Index of this chunk
        total_chunks: Total number of chunks for this document
        chunk_text: The text content of this chunk
        
    Returns:
        Chunk metadata with references to original document
    """
    return {
        "chunk_id": f"{doc_metadata['filename']}_chunk_{chunk_index}",
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "chunk_text": chunk_text,
        "parent_document": doc_metadata['filename'],
        "entities": [e for e in str(doc_metadata.get('entities', [])) if e in chunk_text] if 'entities' in doc_metadata else []
    }