from pathlib import Path
from warnings import simplefilter

import pandas as pd
from tqdm import tqdm

simplefilter(action="ignore", category=FutureWarning)
tqdm.pandas()


FN_DATA_DIR = Path("db")
FN_DOCS_CSV = Path("docs/transcripts.csv")
FN_TWEETS_IN = Path("data/Tweets_R_TrumpBiden.xlsx")

LOWERCASE = False

FN_SOURCE_OUT = "eng_private-{type}-{person}_2020.source"
if LOWERCASE:
    FN_SOURCE_OUT = "eng_private-{type}-lower-{person}_2020.source"
SOURCE_HEADER = """<source><location>{source}</location><date>{date}</date><encoding>UTF-8</encoding><languages ls-version="2.0.0"><language confidence="1.00">eng</language></languages></source>"""

PERSONS = ["Trump", "Biden"]


def get_subset_by_person(df: pd.DataFrame, person: str, col_name: str = "Wer"):
    print(f"* filter dataset by person '{person}'")
    return df[df[col_name] == person]


def lowercase_text(df: pd.DataFrame, colname: str = "text") -> pd.DataFrame:
    def cleanup_text(row):
        row[colname] = row[colname].lower()
        return row

    print("* lowercase text")
    df = df.progress_apply(cleanup_text, axis=1)

    return df


def run():
    df_docs: pd.DataFrame = pd.read_csv(FN_DOCS_CSV)
    df_tweets: pd.DataFrame = pd.read_excel(FN_TWEETS_IN)

    for person in PERSONS:
        df_person = get_subset_by_person(df_docs, person, col_name="Wer")
        if LOWERCASE:
            df_person = lowercase_text(df_person, colname="Text")
        fn = FN_DATA_DIR / FN_SOURCE_OUT.format(type="transcripts", person=person)
        with open(fn, "w", encoding="utf-8") as fp:
            for _, row in tqdm(df_person.iterrows(), desc="Write source"):
                header = SOURCE_HEADER.format(source=row["Link"], date=row["Datum"])
                text = row["Text"]
                # row["Title"] ?
                content = f"{header}\n\n{text}\n\n"
                fp.write(content)

        df_person = get_subset_by_person(df_tweets, person, col_name="Who")
        if LOWERCASE:
            df_person = lowercase_text(df_person, colname="text")
        fn = FN_DATA_DIR / FN_SOURCE_OUT.format(type="tweets", person=person)
        with open(fn, "w", encoding="utf-8") as fp:
            for idx, row in tqdm(df_person.iterrows(), desc="Write source"):
                header = SOURCE_HEADER.format(
                    source=f"{row['Who']}-{idx}", date=row["timestamp"].split(" ", 1)[0]
                )
                text = row["text"]
                # hashtags = row["hashtags"]
                # "\n{hashtags}"
                content = f"{header}\n\n{text}\n\n"
                fp.write(content)


if __name__ == "__main__":
    run()