import spacy
from spacy.cli.download import download


def get_spacy(model="en_core_web_sm") -> spacy.Language:
    try:
        loaded_spacy = spacy.load(model)
    except:
        download(model)
        loaded_spacy = spacy.load(model)

    return loaded_spacy
