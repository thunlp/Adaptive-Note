import json
from elasticsearch import Elasticsearch
import json
import re
import yaml

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

core_title_matcher = re.compile("([^()]+[^\s()])(?:\s*\(.+\))?")
core_title_filter = lambda x: (
    core_title_matcher.match(x).group(1) if core_title_matcher.match(x) else x
)


class ElasticSearch:
    def __init__(self, index_name):
        self.index_name = index_name
        self.client = Elasticsearch(config["es"]["url"])

    def _extract_one(self, item, lazy=False):
        res = {
            k: item["_source"][k]
            for k in ["id", "url", "title", "text", "title_unescape"]
        }
        res["_score"] = item["_score"]
        return res

    def rerank_with_query(self, query, results):
        def score_boost(item, query):
            score = item["_score"]
            core_title = core_title_filter(item["title_unescape"])
            if query.startswith("The ") or query.startswith("the "):
                query1 = query[4:]
            else:
                query1 = query
            if query == item["title_unescape"] or query1 == item["title_unescape"]:
                score *= 1.5
            elif (
                query.lower() == item["title_unescape"].lower()
                or query1.lower() == item["title_unescape"].lower()
            ):
                score *= 1.2
            elif item["title"].lower() in query:
                score *= 1.1
            elif query == core_title or query1 == core_title:
                score *= 1.2
            elif (
                query.lower() == core_title.lower()
                or query1.lower() == core_title.lower()
            ):
                score *= 1.1
            elif core_title.lower() in query.lower():
                score *= 1.05

            item["_score"] = score
            return item

        return list(
            sorted(
                [score_boost(item, query) for item in results],
                key=lambda item: -item["_score"],
            )
        )

    def single_text_query(self, query, topn=10, lazy=False, rerank_topn=50):

        constructed_query = {
            "multi_match": {
                "query": query,
                "fields": [
                    "title^1.25",
                    "title_unescape^1.25",
                    "text",
                    "title_bigram^1.25",
                    "title_unescape_bigram^1.25",
                    "text_bigram",
                ],
            }
        }
        res = self.client.search(
            index=self.index_name,
            body={"query": constructed_query, "timeout": "100s"},
            size=max(topn, rerank_topn),
            request_timeout=100,
        )

        res = [self._extract_one(x, lazy=lazy) for x in res["hits"]["hits"]]
        res = self.rerank_with_query(query, res)[:topn]
        res = [{"title": _["title"], "paragraph_text": _["text"]} for _ in res]
        return res

    def search(self, question, k=10):
        try:
            res = self.single_text_query(query=question, topn=k)
            return json.dumps(res, ensure_ascii=False)
        except Exception as err:
            print(Exception, err)
            raise


def retrieve(index_name, query, topk):
    ES = ElasticSearch(index_name)
    result = ES.search(query, topk)
    return json.loads(result)


if __name__ == "__main__":

    print(retrieve("musique", "Bilbo Baggins", 2))
