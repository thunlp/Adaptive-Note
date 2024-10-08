import time
import os
import json
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
import csv
from sentence_transformers import SentenceTransformer
import faiss
import argparse
import yaml

with open("../../../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    choices=["bge-base-zh-v1.5", "gtr-t5-xxl"],
    default="bge-base-zh-v1.5",
    help="Model to use",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="crud",
    choices=["crud", "asqa"],
    help="Dataset to use",
)
parser.add_argument("--chunk_size", type=int, default=512, help="chunk size")
parser.add_argument("--chunk_overlap", type=int, default=0, help="chunk overlap")
parser.add_argument("--device", type=str, default="cuda", help="Device to use")
args = parser.parse_args()


def split_text(docs_path):
    documents = SimpleDirectoryReader(docs_path).load_data()
    
    node_parser = SimpleNodeParser.from_defaults(
    chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
    contens=[d.text for d in nodes]
    return contens


def build_index(embeddings, vectorstore_path):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, vectorstore_path)


if __name__ == "__main__":

    model = SentenceTransformer(config["model"][args.model], device=args.device)
    vectorstore_path = f"../../../data/corpus/{args.dataset}/{args.dataset}.index"
    print("loading document ...")
    start = time.time()
    if args.dataset == "asqa":
        if os.path.exists("../../../data/corpus/asqa/gtr_wikipedia_index.pkl"):
            import pickle

            with open(
                "../../../data/corpus/asqa/gtr_wikipedia_index.pkl", "rb"
            ) as file:
                embeddings = pickle.load(file)
        else:
            with open(
                "../../../data/corpus/asqa/psgs_w100.tsv", "r", encoding="utf-8"
            ) as file:
                tsv_data = csv.DictReader(file, delimiter="\t")
                raw_data = [row["title"] + "\n" + row["text"] for row in tsv_data]
            print("dataset length", len(raw_data))
            embeddings = model.encode(raw_data, batch_size=100)
    elif args.dataset == "crud":
        contents=split_text("../../../data/corpus/crud/80000_docs")
        contents = list(set(contents))
        print("Generating embeddings ...")
        embeddings = model.encode(contents, batch_size=800)
    print("Building index ...")
    build_index(embeddings, vectorstore_path)
    end = time.time()
    with open(
        f"../../../data/corpus/{args.dataset}/chunk.json", "w", encoding="utf-8"
    ) as fout:
        json.dump(contents, fout, ensure_ascii=False)
