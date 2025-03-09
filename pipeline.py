#Main pipeline that orchestrates the process

import os
from config import PDF_DIR
from text_extraction import extract_text_from_pdf
from ner_processor import perform_ner
from embedding import embed_text
from chunking import split_into_chunks, create_chunk_metadata
from vector_store import store_documents_in_faiss

def process_documents(tokenizer, encoder, ner_model, use_chunking=True, chunk_size=500, chunk_overlap=50):
    """
    Process all PDF documents in the specified directory.
    
    Args:
        tokenizer: BERT tokenizer
        encoder: BERT encoder model
        ner_model: Named entity recognition model
        use_chunking: Whether to split documents into chunks
        chunk_size: Target size of each chunk in words 
        chunk_overlap: Number of words to overlap between chunks
    """
    documents = []
    all_chunks = []
    
    for pdf_file in os.listdir(PDF_DIR):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, pdf_file)
            print(f"Processing {pdf_file}...")
            
            text = extract_text_from_pdf(pdf_path)
            entities = perform_ner(ner_model, text)
            
            #Create document record
            document = {
                "filename": pdf_file,
                "text": text,
                "entities": entities,
            }
            
            if use_chunking:
                #Split document into chunks
                chunks = split_into_chunks(text, chunk_size, chunk_overlap)
                print(f"  - Split into {len(chunks)} chunks")
                
    
                document_chunks = []
                for i, chunk_text in enumerate(chunks):
                    
                    #Create vector embedding for this chunk
                    chunk_vector = embed_text(tokenizer, encoder, chunk_text)
                    
                    #Create chunk metadata
                    chunk_metadata = create_chunk_metadata(document, i, len(chunks), chunk_text)
                    chunk_metadata["vector"] = chunk_vector
                    document_chunks.append(chunk_metadata)
                
                #Add chunks to the global list
                all_chunks.extend(document_chunks)
                
                #Add document vector (embed full text or average chunk vectors)
                document["vector"] = embed_text(tokenizer, encoder, text)
            else:
                document["vector"] = embed_text(tokenizer, encoder, text)
            
            documents.append(document)
    
    print(f"Processed {len(documents)} documents")
    
    if documents:
        if use_chunking:
            store_documents_in_faiss(documents, all_chunks)
            print(f"Documents stored successfully in FAISS with {len(all_chunks)} chunks.")
        else:
            store_documents_in_faiss(documents)
            print("Documents stored successfully in FAISS.")
    else:
        print("No documents found to process.")
    
    return documents