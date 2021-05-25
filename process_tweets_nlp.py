from warnings import simplefilter

import pandas as pd
import spacy
import transformers
from tqdm import tqdm

simplefilter(action="ignore", category=FutureWarning)
tqdm.pandas()

#: email from 30.11.2020
FN_TWEETS_IN = "data/Tweets_R_TrumpBiden.xlsx"
FN_TWEETS_OUT = "data/Tweets_R_TrumpBiden_out.xlsx"

ONLY_ABOUT_OTHER = True


def do_work(df):

    print("* load models")
    nlp = spacy.load("en_core_web_lg")

    def run_spacy(row):
        doc = nlp(row["text"])

        row["spacy"] = doc

        row["text_stop"] = " ".join(tok.text for tok in doc if tok.is_stop)

        tokens = list(doc)
        tokens = [tok for tok in tokens if not tok.is_stop]
        tokens = [
            tok
            for tok in tokens
            if tok.pos_ not in ("PUNCT", "SYM", "CCONJ", "CONJ", "SCONJ")
        ]

        row["text_tokens"] = " ".join(tok.text for tok in tokens if not tok.is_stop)
        row["text_pos"] = " ".join(tok.pos_ for tok in tokens if not tok.is_stop)

        row["text_pronoun"] = " ".join(tok.text for tok in tokens if tok.pos_ == "PRON")
        row["text_noun"] = " ".join(tok.text for tok in tokens if tok.pos_ == "NOUN")
        row["text_proper_noun"] = " ".join(
            tok.text for tok in tokens if tok.pos_ == "PROPN"
        )
        row["text_adjective"] = " ".join(
            tok.text for tok in tokens if tok.pos_ == "ADJ"
        )
        row["text_adverb"] = " ".join(tok.text for tok in tokens if tok.pos_ == "ADV")
        row["text_verb"] = " ".join(tok.text for tok in tokens if tok.pos_ == "VERB")
        row["text_propn_adj"] = " ".join(
            tok.text for tok in tokens if tok.pos_ in ("ADJ", "PROPN")
        )

        return row

    def check_has_biden_tump(row):
        author = row["Who"]
        text = row["text_tokens"]

        row["about_other"] = False

        if author == "Biden":
            if " Trump " in f" {text} ":
                row["about_other"] = True
        elif author == "Trump":
            if " Biden " in f" {text} ":
                row["about_other"] = True

        return row

    def build_bigrams(row):
        text = row["text_tokens"].split(" ")

        pairs_trump_left, pairs_trump_right = list(), list()
        pairs_biden_left, pairs_biden_right = list(), list()
        for w_left, w_middle, w_right in zip(
            [None] + text[:-1], text, text[1:] + [None]
        ):
            if w_middle == "Trump":
                if w_left:
                    pairs_trump_left.append(f"{w_left}+{w_middle}")
                if w_right:
                    pairs_trump_right.append(f"{w_middle}+{w_right}")
            elif w_middle == "Biden":
                if w_left:
                    pairs_biden_left.append(f"{w_left}+{w_middle}")
                if w_right:
                    pairs_biden_right.append(f"{w_middle}+{w_right}")

        row["words_trump+left"] = " ".join(pairs_trump_left)
        row["words_trump+right"] = " ".join(pairs_trump_right)
        row["words_biden+left"] = " ".join(pairs_biden_left)
        row["words_biden+right"] = " ".join(pairs_biden_right)

        row["words_neighbors"] = None
        if row["about_other"]:
            if row["Who"] == "Trump":
                row["words_neighbors"] = " ".join(
                    [row["words_biden+left"], row["words_biden+right"]]
                )
            elif row["Who"] == "Biden":
                row["words_neighbors"] = " ".join(
                    [row["words_trump+left"], row["words_trump+right"]]
                )

        return row

    # --------------------------------

    print("* run spacy (tokenize, POS-tag, stopwords)")
    df = df.progress_apply(run_spacy, axis=1)

    print("* mark rows where one speaks about the other")
    df = df.progress_apply(check_has_biden_tump, axis=1)

    print("* search for neighborhood words")
    df = df.progress_apply(build_bigrams, axis=1)

    return df


def run():
    # load CSV data
    df: pd.DataFrame = pd.read_excel(FN_TWEETS_IN)

    # work: tokenize/pos
    df = do_work(df)

    # pipe = transformers.pipeline("ner")
    # pipe(df.iloc[0].text)
    # nlp = spacy.load("en_core_web_lg")

    if ONLY_ABOUT_OTHER:
        # remove all tweets that do not mention the other
        df = df[df["about_other"]]
        df.drop(columns="about_other", axis=1, inplace=True)

        # remove single pairs
        df.drop(
            columns=[
                "words_trump+left",
                "words_trump+right",
                "words_biden+left",
                "words_biden+right",
            ],
            axis=1,
            inplace=True,
        )

    df.drop(columns="spacy", axis=1, inplace=True)
    df.to_excel(FN_TWEETS_OUT, index=False)


if __name__ == "__main__":
    run()