# Text embedding functionality

import torch

def embed_text(tokenizer, encoder, text):
    """Encodes text into a vector using BERT."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = encoder(**inputs)
    vector = outputs.last_hidden_state.mean(dim=1).numpy().flatten()
    return vector