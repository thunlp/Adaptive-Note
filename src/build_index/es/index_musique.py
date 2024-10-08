from argparse import ArgumentParser
from elasticsearch import Elasticsearch
import html
import json
from tqdm import tqdm

from itertools import chain


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


INDEX_NAME = "musique"


def process_line(data):
    item = {
        "id": data["id"],
        "url": "empty",
        "title": data["title"],
        "title_unescape": html.unescape(data["title"]),
        "text": data["text"],
        "title_bigram": html.unescape(data["title"]),
        "title_unescape_bigram": html.unescape(data["title"]),
        "text_bigram": data["text"],
        "original_json": json.dumps(data),
    }
    return "{}\n{}".format(
        json.dumps({"index": {"_id": "wiki-{}".format(data["id"])}}), json.dumps(item)
    )


es = Elasticsearch(hosts="http://localhost:9200", timeout=50)


def index_chunk(chunk):
    res = es.bulk(index=INDEX_NAME, body="\n".join(chunk), timeout="100s")
    assert not res["errors"], res


def main(args):
    # make index
    if not args.dry:
        if es.indices.exists(index=INDEX_NAME):
            es.indices.delete(index=INDEX_NAME, ignore=[400, 403])
        es.indices.create(
            index=INDEX_NAME,
            ignore=400,
            body=json.dumps(
                {
                    "mappings": {
                        "doc": {
                            "properties": {
                                "id": {"type": "keyword"},
                                "url": {"type": "keyword"},
                                "title": {
                                    "type": "text",
                                    "analyzer": "simple",
                                    "copy_to": "title_all",
                                },
                                "title_unescape": {
                                    "type": "text",
                                    "analyzer": "simple",
                                    "copy_to": "title_all",
                                },
                                "text": {
                                    "type": "text",
                                    "analyzer": "my_english_analyzer",
                                },
                                "anchortext": {
                                    "type": "text",
                                    "analyzer": "my_english_analyzer",
                                },
                                "title_bigram": {
                                    "type": "text",
                                    "analyzer": "simple_bigram_analyzer",
                                    "copy_to": "title_all_bigram",
                                },
                                "title_unescape_bigram": {
                                    "type": "text",
                                    "analyzer": "simple_bigram_analyzer",
                                    "copy_to": "title_all_bigram",
                                },
                                "text_bigram": {
                                    "type": "text",
                                    "analyzer": "bigram_analyzer",
                                },
                                "anchortext_bigram": {
                                    "type": "text",
                                    "analyzer": "bigram_analyzer",
                                },
                                "original_json": {"type": "string"},
                            }
                        }
                    },
                    "settings": {
                        "analysis": {
                            "my_english_analyzer": {
                                "type": "standard",
                                "stopwords": "_english_",
                            },
                            "simple_bigram_analyzer": {
                                "tokenizer": "standard",
                                "filter": ["lowercase", "shingle", "asciifolding"],
                            },
                            "bigram_analyzer": {
                                "tokenizer": "standard",
                                "filter": [
                                    "lowercase",
                                    "stop",
                                    "shingle",
                                    "asciifolding",
                                ],
                            },
                        },
                    },
                }
            ),
        )

    train = [
        json.loads(line.strip())
        for line in open("../../../data/corpus/musique/musique_ans_v1.0_train.jsonl")
    ] + [
        json.loads(line.strip())
        for line in open("../../../data/corpus/musique/musique_full_v1.0_train.jsonl")
    ]
    dev = [
        json.loads(line.strip())
        for line in open("../../../data/corpus/musique/musique_ans_v1.0_dev.jsonl")
    ] + [
        json.loads(line.strip())
        for line in open("../../../data/corpus/musique/musique_full_v1.0_dev.jsonl")
    ]
    test = [
        json.loads(line.strip())
        for line in open("../../../data/corpus/musique/musique_ans_v1.0_test.jsonl")
    ] + [
        json.loads(line.strip())
        for line in open("../../../data/corpus/musique/musique_full_v1.0_test.jsonl")
    ]

    tot = 0
    wikipedia_data = []
    hist = set()
    for item in tqdm(chain(train, dev, test)):
        for p in item["paragraphs"]:
            stamp = p["title"] + " " + p["paragraph_text"]
            if not stamp in hist:
                wikipedia_data.append(
                    {"id": tot, "text": p["paragraph_text"], "title": p["title"]}
                )
                hist.add(stamp)
                tot += 1

    print("Making indexing queries...")
    all_queries = []
    for item in tqdm(wikipedia_data):
        all_queries.append(process_line(item))

    count = sum(len(queries.split("\n")) for queries in all_queries) // 2

    if not args.dry:
        print("Indexing...")
        chunksize = 100
        for chunk in tqdm(
            chunks(all_queries, chunksize),
            total=(len(all_queries) + chunksize - 1) // chunksize,
        ):
            res = es.bulk(index=INDEX_NAME, body="\n".join(chunk), timeout="100s")
            assert not res["errors"], res

    print(f"{count} documents indexed in total")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--reindex", action="store_true", help="Reindex everything")
    parser.add_argument("--dry", action="store_true", help="Dry run")

    args = parser.parse_args()

    main(args)
