from pathlib import Path

import pandas as pd
import spacy
from gensim.models import Word2Vec
from tqdm import tqdm


FN_DOCS_CSV = Path("docs/transcripts.csv")

PERSONS = ["Trump", "Biden"]


def get_subset_by_person(df, person):
    print(f"* filter dataset by person '{person}'")
    return df[df["Wer"] == person]


def train_model(df, nlp):
    sentences = list()
    print("* tokenize documents")
    for text in tqdm(df["Text"], desc="Tokenize"):
        doc = nlp(text)
        for sent in doc.sents:
            tokens = list(sent)
            # tokens = [tok for tok in tokens if tok.is_stop]
            sentences.append([tok.text for tok in tokens])
    print(f"-> got {len(sentences)} sentences in {len(df)} documents.")

    print("* train word2vec model")
    model = Word2Vec(
        sentences=sentences,
        size=100,
        window=7,
        min_count=1,
        workers=4,
        iter=30,
    )
    return model


def run():
    df = pd.read_csv(FN_DOCS_CSV)

    print("* load spacy model")
    nlp = spacy.load("en_core_web_lg")

    for person in PERSONS:
        df_person = get_subset_by_person(df, person)
        model = train_model(df_person, nlp)
        model.save(f"{person}.w2v.model")


if __name__ == "__main__":
    run()