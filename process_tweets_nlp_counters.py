from collections import Counter
from warnings import simplefilter

import pandas as pd
from tqdm import tqdm

simplefilter(action="ignore", category=FutureWarning)
tqdm.pandas()

FN_TWEETS_IN = "data/Tweets_R_TrumpBiden_out.xlsx"
FN_TWEETS_OUT = "data/Tweets_R_TrumpBiden_counters.xlsx"


def count_neighbors(dfp):
    npairs = [
        tuple(w.split("+"))
        for row in dfp["words_neighbors"]
        for w in str(row).strip().split(" ")
        if row
    ]
    cnt = Counter(npairs)

    dfpc = pd.DataFrame(
        [
            (n, left, right)
            for (left, right), n in sorted(
                cnt.items(), key=lambda x: x[1], reverse=True
            )
        ],
        columns=["amount", "left", "right"],
    )
    return dfpc


def count_words_in_column(dfp, colname):
    if colname not in dfp.columns:
        return None

    words = [w for row in dfp[colname] for w in str(row).strip().split(" ") if row]
    cnt = Counter(words)
    if "nan" in cnt:
        del cnt["nan"]

    dfpc = pd.DataFrame(
        [
            (n, word)
            for word, n in sorted(cnt.items(), key=lambda x: x[1], reverse=True)
        ],
        columns=["amount", "word"],
    )
    return dfpc


def run():
    # load CSV data
    df: pd.DataFrame = pd.read_excel(FN_TWEETS_IN)
    writer = pd.ExcelWriter(FN_TWEETS_OUT, engine="xlsxwriter")

    for person in ("Trump", "Biden"):
        dfp = df[df["Who"] == person]

        # neighbors
        dfpc = count_neighbors(dfp)
        dfpc.to_excel(writer, index=False, sheet_name=f"{person} Neighbors (counted)")

        for pos in (
            "verb",
            "adjective",
            "proper_noun",
            "noun",
            "pronoun",
            "adverb",
            "stop",
        ):
            colname = f"text_{pos}"
            dfppc = count_words_in_column(dfp, colname)
            if dfppc is not None:
                dfppc.to_excel(writer, index=False, sheet_name=f"{person} Word ({pos})")

    writer.save()


if __name__ == "__main__":
    run()