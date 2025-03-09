# Model loading and initialization

from transformers import BertModel, BertTokenizer
from gliner import GLiNER

def load_models():
    """Load and initialize all required models."""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoder = BertModel.from_pretrained("bert-base-uncased")  
    ner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    
    return tokenizer, encoder, ner_model