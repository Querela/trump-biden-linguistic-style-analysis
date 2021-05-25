import csv
import re
from collections import defaultdict
from collections import Counter
from io import StringIO
from warnings import simplefilter

import pandas as pd
from tqdm import tqdm

simplefilter(action="ignore", category=FutureWarning)
tqdm.pandas()


FN_TWEETS_RAW = "data/JoeBidenTweets.csv"
FN_TWEETS_OUT = "data/biden.xlsx"

PAT_WHITESPACES = re.compile(r"\s+", re.DOTALL | re.MULTILINE | re.DOTALL)
PAT_HASHTAG = re.compile(r"#[^ ]+", re.UNICODE | re.IGNORECASE | re.DOTALL)
PAT_MENTION = re.compile(r"@[\w_]+", re.UNICODE | re.IGNORECASE | re.DOTALL)
PAT_RETWEET = re.compile(
    r"^RT (" + PAT_MENTION.pattern + "):", re.UNICODE | re.IGNORECASE | re.DOTALL
)
PAT_URL = re.compile(
    r"""\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))""",
    re.UNICODE | re.IGNORECASE | re.DOTALL,
)
PAT_PUNCT = re.compile(r"""[!?"“”`´*~+/\\]""", re.UNICODE | re.IGNORECASE | re.DOTALL)
# —Joe ?
PAT_PUNCT2 = re.compile(r"(\W|['_—-])+\s", re.UNICODE | re.IGNORECASE | re.DOTALL)

# https://github.com/adonoho/TweetTokenizers/blob/master/PottsTweetTokenizer.py
# https://gist.github.com/gruber/8891611

# ---------------------------------------------------------------------------


def prepare(df, keep_original=False):
    print("* keep only timestamp, tweet --> rename to text")
    # just keep timestamp and content text
    df = df[["timestamp", "tweet"]]
    # rename columns
    df.columns = ["timestamp", "text"]
    if keep_original:
        print("* keep original text")
        # store original text
        df["text_original"] = df["text"]

    return df


def cleanup(df, keep_mentions_in_text=True):
    # cleanup text
    def cleanup_text(row):
        # row["text"] = row["text"].replace("\n", " ")
        row["text"] = PAT_WHITESPACES.sub(" ", row["text"]).strip()
        return row

    # extract hashtags into own column
    def extract_hashtags(row):
        hash_tags = PAT_HASHTAG.findall(row["text"])
        row["hashtags"] = " ".join(h.rstrip(""".!;:?"'""") for h in hash_tags)
        return row

    def clean_hashtags(row):
        row["text"] = PAT_WHITESPACES.sub(
            " ", PAT_HASHTAG.sub(" ", row["text"])
        ).strip()
        return row

    def extract_retweets(row):
        row["retweet"] = None

        match = PAT_RETWEET.search(row["text"])
        if match:
            row["retweet"] = match.group(1)
            row["text"] = row["text"][len(match.group(0)) :].lstrip()

        return row

    # extract mention into own column
    def extract_mention(row):
        mentions = PAT_MENTION.findall(row["text"])
        row["mentions"] = " ".join(mentions)
        return row

    def clean_mention(row):
        row["text"] = PAT_WHITESPACES.sub(
            " ", PAT_MENTION.sub(" ", row["text"])
        ).strip()
        return row

    def clean_urls(row):
        row["text"] = PAT_WHITESPACES.sub(" ", PAT_URL.sub(" ", row["text"])).strip()
        return row

    def clean_punctuation(row):
        # row["text"] = row["text"].replace("!", "")
        # remove punctuation that won't ever appear between words, like "stop-gap" should be left alone
        row["text"] = PAT_PUNCT.sub(" ", row["text"])
        # remove punctuations which are followed by space character
        row["text"] = PAT_PUNCT2.sub(" ", row["text"] + " ")
        # normalize whitespaces
        row["text"] = PAT_WHITESPACES.sub(" ", row["text"]).strip()
        return row

    # --------------------------------

    print("* cleanup whitespaces")
    df = df.progress_apply(cleanup_text, axis=1)

    print("* extract hashtags")
    df = df.progress_apply(extract_hashtags, axis=1)
    print("* remove hashtags from text")
    df = df.progress_apply(clean_hashtags, axis=1)

    print("* check re-tweets")
    df = df.progress_apply(extract_retweets, axis=1)

    print("* extract mention")
    df = df.progress_apply(extract_mention, axis=1)
    if not keep_mentions_in_text:
        print("* remove mention from text")
        df = df.progress_apply(clean_mention, axis=1)

    print("* remove URLs")
    df = df.progress_apply(clean_urls, axis=1)

    print("* remove punctuation marks")
    df = df.progress_apply(clean_punctuation, axis=1)

    return df


def trim_empty(df):
    len_before = len(df)

    mask_empty = df["text"] == ""
    df = df[~mask_empty]

    len_after = len(df)
    print(f"Trim empty tweets: {len_before} -> {len_after}")

    return df


# ---------------------------------------------------------------------------


def run():
    # load CSV data
    df = pd.read_csv(FN_TWEETS_RAW)
    df = prepare(df)

    # cleanup / transform data
    df = cleanup(df)
    df = trim_empty(df)

    df.to_excel(FN_TWEETS_OUT, index=False)
    # df.to_csv(FN_TWEETS_OUT, index=False, sep=";", encoding="utf-8-sig")


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    run()
