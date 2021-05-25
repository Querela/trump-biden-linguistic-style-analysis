# Data processing

## Setup

```bash
apt-get install antiword
```

```bash
python3.8 -m venv venv/
source venv/bin/activate

pip install -U pip setuptools wheel

pip install -r requirements.txt

python -m spacy download en_core_web_lg
```

Always enable the virtual environment beforehand:

```bash
source venv/bin/activate
```

## Tweets

```bash
python process_biden.py
```

```bash
python process_trump.py
```

### Data

Download tweets and store them in [`data`](data) folder. Sources below:

- [Biden @Kaggle](https://www.kaggle.com/rohanrao/joe-biden-tweets)
- [Trump](https://www.thetrumparchive.com/faq)

## Transcripts

Note, see TSV file ([`data/list-of-transcripts.tsv`](data/list-of-transcripts.tsv)) with the transcription links and meta data.
Run the following command to download and extract the text content from the web:

```bash
python download_all.py
```

## Workflows

### Webscrape transcripts

1. input file: [`data/list-of-transcripts.tsv`](data/list-of-transcripts.tsv) (list of links)
2. run: [`download_all.py`](download_all.py)
3. generates `docs/transcripts.xlsx` / `transcripts.csv` + other files in `docs/` / `txt`

### Process tweets

Automatic preprocessing of manually downloaded input files. Afterwards manual selection of required date range etc.

- Trump:

    1. see links above
    2. input file: `data/tweets_11-06-2020.csv`
    3. run: [`process_trump.py`](process_trump.py)
    4. generates: `data/trump.xlsx`

- Biden:

    1. see links above
    2. input file: `data/JoeBidenTweets.csv`
    3. run: [`process_biden.py`](process_biden.py)
    4. generates: `data/biden.xlsx`

Our manually curated and filtered set of tweets exists in the file: [`data/Tweets_R_TrumpBiden.xlsx`](data/Tweets_R_TrumpBiden.xlsx)

### Tweet Word Statistics

1. input file [`data/Tweets_R_TrumpBiden.xlsx`](data/Tweets_R_TrumpBiden.xlsx)
2. run: [`process_tweets_nlp.py`](process_tweets_nlp.py)
3. generates: `data/Tweets_R_TrumpBiden_out.xlsx`
4. run: [`process_tweets_nlp_counters.py`](process_tweets_nlp_counters.py)
5. generates: `data/Tweets_R_TrumpBiden_counters.xlsx`

### Word2Vec

1. input file `docs/transcripts.csv`
2. build models using [`make_w2v_model.py`](make_w2v_model.py)
3. query words using [`query_w2v_model.py`](query_w2v_model.py)
    - ex: `python query_w2v_model.py 'Trump;Biden;war;American;America;USA;homeless;wages;money;hunger;policies;politics;Europe'`

### Cooccurrence Analysis

1. input files: `docs/transcripts.csv`, `data/Tweets_R_TrumpBiden.xlsx`
2. export `source` files using [`export_toolchain_input.py`](export_toolchain_input.py)
3. run ASV Wortschatz toolchain (store corpora in DB) (corpus creation, cooccurrences)
4. run ASV GDEX, `sim_w_co` scripts (cooccurrences dice similarity)
5. run ASV pos-tagger (TreeTagger ENG)
6. backup/export databases

