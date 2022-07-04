import spacy

nlp = spacy.load("en_core_web_sm")
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS


def preprocess(text: str) -> str:
    tokens = nlp(text.lower())
    return " ".join(
        [
            token.lemma_
            for token in tokens
            if not token.is_stop and not token.is_punct and token.lemma_ != " "
        ]
    )
