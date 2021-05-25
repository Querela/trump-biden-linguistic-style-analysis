import os
import re
from pathlib import Path
from warnings import simplefilter

import numpy as np
import pandas as pd
import requests
from cachecontrol import CacheControl
from cachecontrol.caches.file_cache import FileCache
from parsel import Selector
from tqdm import tqdm

from typing import List, NamedTuple


simplefilter(action="ignore", category=FutureWarning)
tqdm.pandas()


# ---------------------------------------------------------------------------

FN_DOCS_DIR = Path("docs")
FN_TXT_DIR = Path("txt")
FN_SHEET_INFO = Path("data/list-of-transcripts.tsv")
FN_DOCS_XLSX = FN_DOCS_DIR / "transcripts.xlsx"
FN_DOCS_CSV = FN_DOCS_DIR / "transcripts.csv"

OVERWRITE_EXISTING = True

PAT_BRACES = re.compile(r"\([^()]*?\)", re.DOTALL | re.UNICODE | re.IGNORECASE)
PAT_BRACKETS = re.compile(r"\[[^[\]]*?\]", re.DOTALL | re.UNICODE | re.IGNORECASE)

# ---------------------------------------------------------------------------


class TInfo(NamedTuple):
    id_: int
    who: str
    date: str
    title: str
    type_: str
    url: str
    rest: str


def load_sheet_info(fn_sheet: os.PathLike) -> List[TInfo]:
    tis = list()

    with open(fn_sheet, "r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.rstrip().split("\t")
            id_, who, date, title, type_, url, rest = parts

            id_ = int(id_)
            date = "-".join(reversed(date.split(".")))

            ti = TInfo(id_, who, date, title, type_, url, rest)
            tis.append(ti)

    return tis


def extract_text_blocks(content: str):
    sel = Selector(content)

    blocks = sel.css(".fl-callout-text > p").getall()
    test_block = blocks[2] if len(blocks) > 2 else blocks[0]
    if "<br>" in test_block:
        rows = list()
        for i, block in enumerate(blocks):
            block = block[3:-4].strip()
            if not block:
                continue
            if not "<br>" in block:
                print(f"! Empty speaker text block @{i}")
                continue

            header, text = block.split("<br>", 1)
            who = header.split(":", 1)[0].strip()
            text = text.strip()

            tb = {"speaker": who, "text": text}
            rows.append(tb)

    else:
        # multi line things?
        rows = list()
        for i in range(0, len(blocks), 2):
            header = blocks[i][3:-4]
            assert "<br>" not in header
            text = blocks[i + 1][3:-4]
            assert ":" in header

            who = header.split(":", 1)[0].strip()
            text = text.strip()

            tb = {"speaker": who, "text": text}
            rows.append(tb)

    df = pd.DataFrame.from_dict(rows)

    return df


def cleanup(df):
    # remove annotations / non-speech
    def remove_non_speech(row):
        # TODO: filter (...) | [...], non-greedy
        text = row["text"]

        while True:
            len_before = len(text)
            text = PAT_BRACES.sub("", text)
            if len(text) == len_before:
                # no change, then nothing more to replace, abort
                break

        while True:
            len_before = len(text)
            text = PAT_BRACKETS.sub("", text)
            if len(text) == len_before:
                break

        row["text"] = text
        return row

    # df.progress_apply(remove_non_speech, axis=1)
    df.apply(remove_non_speech, axis=1)

    return df


def process_one(
    sess,
    fn_name: os.PathLike,
    url: str,
    ti: TInfo,
    filter_speaker: bool = True,
    add_meta: bool = True,
    rename_german: bool = True,
    merge_one: bool = True,
    save_disk: bool = True,
    save_text: bool = True,
    fn_name_text: os.PathLike = None,
):
    req = sess.get(url)
    if not req.ok:
        print(f"Error with request: {fn_name.name} - {url}")
        return

    text = req.text

    df = extract_text_blocks(text)

    df = cleanup(df)

    if filter_speaker:
        # print(f"* Filter for speaker: {ti.who}")
        if ti.who == "Trump":
            mask_speaker = (
                (df["speaker"] == "Donald Trump")
                | (df["speaker"] == "President Trump")
                | (df["speaker"] == "Donald J Trump")
                | (df["speaker"] == "Donald J. Trump")
                | (df["speaker"] == "President Donald Trump")
                | (df["speaker"] == "President Donald J. Trump")
            )
        elif ti.who == "Biden":
            mask_speaker = (
                (df["speaker"] == "Joe Biden")
                | (df["speaker"] == "Vice President Joe Biden")
                | (df["speaker"] == "VIce President Biden")
            )
        else:
            # mask_speaker = np.one(len(df)).astype(bool)
            raise Exception("invalid speaker?")
        df = df[mask_speaker].copy()

        # in case we do not want the speaker in the output
        # because it is in the file name ...
        # del df["speaker"]

    if add_meta or rename_german:
        # print("* add meta columns")
        df["date"] = ti.date
        df["title"] = ti.title
        df["type"] = ti.type_
        df["rest"] = ti.rest
        df["url"] = ti.url
        df["who"] = ti.who

    if rename_german:
        # reorder and filter
        df = df[["text", "who", "date", "title", "url", "rest"]]
        # rename
        df.columns = ["Text", "Wer", "Datum", "Titel", "Link", "Sonstiges"]

    if save_disk:
        if str(fn_name).endswith("csv"):
            df.to_csv(fn_name, index=False) # , sep=";", encoding="utf-8-sig")
        elif str(fn_name).endswith("xlsx"):
            df.to_excel(fn_name, index=False)
        else:
            raise Exception("Invalid format!?")

    if save_text and fn_name_text:
        text = "\r\n\r\n".join(df["Text"].to_list())
        Path(fn_name_text).write_text(text, encoding="utf-8")

    if merge_one and rename_german and len(df) > 0:
        # group by all except text
        index_cols = df.columns.tolist()
        index_cols.remove("Text")

        # merge grouped result (text)
        # df = df.groupby(index_cols)["Text"].apply(list)
        df = df.groupby(index_cols)["Text"].apply(lambda x: " ".join(x.tolist()))
        df = df.reset_index()

        # reorder
        df = df[["Text", "Wer", "Datum", "Titel", "Link", "Sonstiges"]]

    return df


def run():
    if not FN_DOCS_DIR.exists():
        print(f"* create output dir: {FN_DOCS_DIR}")
        FN_DOCS_DIR.mkdir()
    if not FN_TXT_DIR.exists():
        print(f"* create output dir: {FN_TXT_DIR}")
        FN_TXT_DIR.mkdir()

    sess = CacheControl(requests.Session(), cache=FileCache(".web_cache"))

    tis = load_sheet_info(FN_SHEET_INFO)
    # tis = tis[:1]  # TESTING

    dfs = list()

    for ti in tqdm(tis, desc="Process transcriptions"):
        # output name (determines format (CSV or Excel))
        fn_name = FN_DOCS_DIR / f"{ti.date}-{ti.who}-{ti.id_}.csv"
        # fn_name = FN_DOCS_DIR / f"{ti.date}.{ti.who}.{ti.id_}.xlsx"

        fn_name_text = FN_TXT_DIR / f"{ti.date}-{ti.who}-{ti.id_}.txt"

        if not OVERWRITE_EXISTING and fn_name.exists():
            print(f"* {fn_name} exists. Skip.")
            continue

        df_one = process_one(sess, fn_name, ti.url, ti, fn_name_text=fn_name_text)
        # df_one = df_one.reset_index(drop=True)
        dfs.append(df_one)

    df = pd.concat(dfs, axis=0)
    df.to_excel(FN_DOCS_XLSX, index=False)
    df.to_csv(FN_DOCS_CSV, index=False) # , sep=";", encoding="utf-8-sig")


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    run()
