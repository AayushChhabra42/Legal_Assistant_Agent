# Named Entity Recognition processing

from config import LEGAL_ENTITIES

def perform_ner(ner_model, text):
    """Performs Named Entity Recognition using gLiner."""
    entities = ner_model.predict_entities(text, labels=LEGAL_ENTITIES)
    return entities