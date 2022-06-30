
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.base import ClassifierMixin
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from utils.sklearn import pipelinize

df_train = pd.read_json('data/train.jsonl', lines=True)
df_dev = pd.read_json('data/dev.jsonl', lines=True)

nlp = spacy.load('en_core_web_sm')
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
def preprocess(text):
    tokens = nlp(text.lower())
    return " ".join([token.lemma_ for token in tokens if not token.is_stop and not token.is_punct and token.lemma_ != ' '])

X_train = df_train['text'].swifter.apply(preprocess)
y_train = df_train['label']
X_dev = df_dev['text'].swifter.apply(preprocess)
y_dev = df_dev['label']


def train(classifier: ClassifierMixin):
    pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(min_df=2, max_df=0.5, stop_words= spacy_stopwords, max_features=100000, ngram_range=(1, 3))),
                # ('feature_selection', SelectPercentile(chi2, percentile=50)),
                ('clf', OneVsRestClassifier(Pipeline([('sampling', RandomOverSampler()), ('svc', classifier)], verbose=True), n_jobs=4)),
            ], verbose=True)

    pipeline.fit(X_train, y_train)
    f1 = f1_score(y_dev,pipeline.predict(X_dev),average='weighted')
    print(f"{type(classifier)} f1: {f1}")

train(LinearSVC())
train(MultinomialNB())