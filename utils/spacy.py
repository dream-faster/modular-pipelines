import spacy
from spacy.cli.download import download


def get_spacy() -> spacy.Language:
    try:
        loaded_spacy = spacy.load("en_core_web_lg")
    except:
        download("en_core_web_lg")
        loaded_spacy = spacy.load("en_core_web_lg")

    return loaded_spacy
