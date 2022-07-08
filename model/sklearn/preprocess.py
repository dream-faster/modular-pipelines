import spacy
from typing import Callable
import spacy


def create_preprocess(nlp: spacy) -> Callable:
    def preprocess(text: str) -> str:
        tokens = nlp(text.lower())
        return " ".join(
            [
                token.lemma_
                for token in tokens
                if not token.is_stop and not token.is_punct and token.lemma_ != " "
            ]
        )

    return preprocess
