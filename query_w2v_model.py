import sys
from io import StringIO
from itertools import zip_longest
from pathlib import Path

import pandas as pd
from gensim.models import Word2Vec


FN_DOCS_CSV = Path("docs/transcripts.csv")
TOP_K = 10


def run(query):
    query = query.split(";") if ";" in query else [query]

    outputs = list()

    for person in ["Trump", "Biden"]:
        fp = StringIO()
        model = Word2Vec.load(f"{person}.w2v.model")

        for qword in query:
            qword = qword.strip()

            print("-" * 40, file=fp)
            print(f"  {person.upper()}  - word: '{qword}'?", file=fp)
            print("-" * 40, file=fp)

            if qword not in model.wv:
                print("--> word not found!", file=fp)
                fp.write("\n" * (TOP_K - 1))

            else:
                sims = model.wv.most_similar(qword, topn=TOP_K)

                for word, score in sims:
                    print(f"{word:<30} [{score:.3f}]", file=fp)

            print("", file=fp)

        outputs.append(fp.getvalue())

    lines = zip_longest(*[s.split("\n") for s in outputs], fillvalue="")
    output = "\n".join(" | ".join([f"{l:<40}" for l in lp]) for lp in lines)
    print(output)


if __name__ == "__main__":
    word = sys.argv[1]
    run(word)