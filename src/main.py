import os
import json
import argparse
import logging
import datetime
import yaml
import faiss
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import backoff
import re
import time
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from eval import acc_score, F1_scorer, compute_exact, eval_asqa
from utils import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    choices=["gpt-3.5-turbo-0125", "llama3-8b-instruct", "qwen2-7b-instruct"],
    default="llama3-8b-instruct",
    help="Model to use",
)
parser.add_argument(
    "--max_step", type=int, default=2, help="Maximum number of update steps"
)
parser.add_argument(
    "--max_fail_step", type=int, default=2, help="Maximum number of failed steps"
)
parser.add_argument(
    "--MaxClients", type=int, default=1, help="Number of concurrent clients"
)
parser.add_argument(
    "--retrieve_top_k", type=int, default=5, help="Top K documents to retrieve"
)
parser.add_argument(
    "--max_top_k",
    type=int,
    default=15,
    help="Total maximum number of documents to retrieve",
)
parser.add_argument(
    "--dataset",
    type=str,
    choices=[
        "2wikimultihopqa",
        "hotpotqa",
        "musique",
        "asqa",
        "crud_1doc",
        "crud_2doc",
        "crud_3doc",
    ],
    default="hotpotqa",
    help="Dataset to use",
)
parser.add_argument(
    "--method",
    type=str,
    default="note",
    choices=["note", "base", "base_wo_retri"],
    help="Method to use",
)
parser.add_argument(
    "--resume_path",
    type=str,
    default="",
    help="Path to the file for resuming generation",
)
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
args = parser.parse_args()

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


@backoff.on_exception(backoff.expo, (Exception), max_time=100)
def call_gpt(prompt_file, variable_dict):
    with open(prompt_file, "r") as fin:
        prompt = fin.read().format(**variable_dict)
    res = gpt_gen(args.model, prompt)
    assert res is not None
    return res


def call_local(prompt_file, variable_dict):
    return local_gen(model, tokenizer, prompt_file, variable_dict)


if "gpt" in args.model:
    from utils import gpt_gen

    call_llm = call_gpt
elif "llama" in args.model:
    path = config["model"]["llama3-8b-instruct"]
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=args.device
    )
    from utils import llama_gen as local_gen

    call_llm = call_local
elif "qwen" in args.model:
    path = config["model"]["qwen2-7b-instruct"]
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    from utils import qwen_gen as local_gen

    call_llm = call_local


def get_context(data):

    if args.dataset == "asqa" or "crud" in args.dataset:
        text = "\n".join(data)
    else:
        text = ""
        for i in range(len(data)):
            text += f"Passage #{i+1} Title: {data[i]['title']}\nPassage #{i+1} Text: {data[i]['paragraph_text']} \n\n"
    return text


def save_log_to_file(logger, log_file="my_log", log_folder="logs"):
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    current_date = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
    log_file_name = f"{log_file}-{current_date}.log"
    file_handler = logging.FileHandler(os.path.join(log_folder, log_file_name))
    logger.addHandler(file_handler)


def is_directory_empty(directory_path: str) -> bool:
    try:
        return len(os.listdir(directory_path)) == 0
    except OSError:
        return False


def call_llm_template(template, variables):
    return call_llm(f"../prompts/{LAUGUAGE}/{template}", variables)


def first_gen_answer(query, refs):
    return call_llm_template("first_gen_answer", {"query": query, "refs": refs})


def gen_new_query(query, note, query_log):
    return call_llm_template(
        "gen_new_query", {"query": query, "note": note, "query_log": query_log}
    )


def refine_note(note, query, refs):
    return call_llm_template(
        "refine_note", {"note": note, "query": query, "refs": refs}
    )


def gen_answer(query, note_str):
    template = "gen_answer_asqa" if "asqa" in args.dataset else "gen_answer"
    return call_llm_template(template, {"query": query, "note": note_str})


def compare_note(query, best_note, new_note):
    response = call_llm_template(
        "compare", {"query": query, "best_note": best_note, "new_note": new_note}
    )
    try:
        status = json.loads(response)["status"]
        return status.lower() == "true"
    except:
        return "true" in response.lower()


def retrieve_note(doc_id, query, answer, top_k=2):
    ref_log, llm_times, note_log, query_log, query_list = [], 0, [], [], []
    refs = retrieve(args.dataset, query=query, topk=top_k)
    note = best_note = first_gen_answer(query, get_context(refs))
    ref_log.append({"refs": refs, "step": 0, "flag": "init_refs"})
    note_log.append({"note": note, "step": 0, "flag": "init_note"})
    llm_times += 1
    step, notes_status = 0, []

    while step < args.max_step:
        all_refs = {
            ref
            for d in ref_log
            for ref in (
                d["refs"]
                if "crud" in args.dataset or args.dataset == "asqa"
                else [d["title"] + d["paragraph_text"] for d in d["refs"]]
            )
        }
        if len(all_refs) >= args.max_top_k:
            break
        max_ref = (
            args.max_top_k - len(all_refs)
            if (len(all_refs) + top_k) > args.max_top_k
            else 0
        )

        new_query = gen_new_query(query, best_note, str(query_list))
        query_list.append(new_query)
        llm_times += 1

        refs = retrieve(
            args.dataset, query=(new_query + "\n" + query)[:500], topk=top_k
        )
        if max_ref > 0:
            refs = (
                [d for d in refs if d not in all_refs][:max_ref]
                if "crud" in args.dataset or args.dataset == "asqa"
                else [
                    d for d in refs if d["title"] + d["paragraph_text"] not in all_refs
                ][:max_ref]
            )

        note = refine_note(best_note, query, get_context(refs)).replace("\n", "")
        llm_times += 1
        status = compare_note(query, best_note, note)
        flag = "True" if status else "False"

        ref_log.append({"refs": refs, "step": step, "flag": flag})
        note_log.append({"note": note, "step": step, "flag": flag})
        query_log.append({"query": new_query, "step": step, "flag": flag})

        if status:
            best_note = note
        notes_status.append(status)
        if notes_status.count(False) >= args.max_fail_step:
            break
        step += 1

    llm_times += 1
    return {
        "id": doc_id,
        "question": query,
        "answer": answer,
        "output": gen_answer(query, best_note),
        "note": best_note,
        "query_log": query_log,
        "note_log": note_log,
        "ref_log": ref_log,
    }


def process_doc_cell(idx, doc_cell, args):
    id_new, query, answer = idx, doc_cell["question"], doc_cell["answer"]

    if args.method != "note":
        prompt_path = f"../prompts/{LAUGUAGE}"
        gen_name = (
            "base_wo_retri_asqa"
            if "wo_retri" in args.method and "asqa" in args.dataset
            else (
                "base_wo_retri"
                if "wo_retri" in args.method
                else "base_asqa" if "asqa" in args.dataset else "base"
            )
        )

        output = (
            call_llm(f"{prompt_path}/{gen_name}", {"query": query})
            if "wo_retri" in args.method
            else call_llm(
                f"{prompt_path}/{gen_name}",
                {
                    "query": query,
                    "refs": get_context(
                        retrieve(args.dataset, query=query, topk=args.retrieve_top_k)
                    ),
                },
            )
        )

        return {"id": id_new, "query": query, "answer": answer, "output": output}
    else:
        return retrieve_note(id_new, query, answer, top_k=args.retrieve_top_k)


if __name__ == "__main__":

    LAUGUAGE = "en"
    seed_everything(66)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    save_log_to_file(
        logger,
        log_file=f"{args.dataset}_{args.method}_{args.model}",
        log_folder="../log",
    )
    logger.info(f"{'*' * 30} CONFIGURATION {'*' * 30}")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (30 - len(key)))
        logger.info("%s -->   %s", keystr, val)
    dataset_name=args.dataset
    if "crud" in args.dataset:
        dataset_name = "crud"
    vector_path = f"../data/corpus/{dataset_name}/{dataset_name}.index"
    if args.dataset == "asqa":
        vector = faiss.read_index(vector_path)
        emb_model = SentenceTransformer(config["model"]["gtr-t5-xxl"], device="cpu")
        raw_data = pd.read_csv("../data/corpus/asqa/psgs_w100.tsv", sep="\t")

        def retrieve(_, query, topk):
            feature = emb_model.encode([query])
            _, match_id = vector.search(feature, topk)
            return [
                raw_data.iloc[i]["title"] + "\n" + raw_data.iloc[i]["text"]
                for i in match_id[0]
            ]

    elif "crud" in args.dataset:
        LAUGUAGE = "zh"
        vector = faiss.read_index(vector_path)
        emb_model = SentenceTransformer(
            config["model"]["bge-base-zh-v1.5"], device=args.device
        )
        with open(f"../data/corpus/crud/chunk.json", encoding="utf-8") as f:
            raw_data = json.load(f)

        def retrieve(_, query, topk):
            feature = emb_model.encode([query])
            _, match_id = vector.search(feature, topk)
            return [raw_data[i] for i in match_id[0]]

    else:
        from es_retrieve import retrieve

    formatted_time = time.strftime("%Y%m%d-%H%M%S")
    dataset = re.sub(r"_es$", "", args.dataset)

    with open(f"../data/eval/{dataset}/test.json", encoding="utf-8") as f:
        qa_data = json.load(f)

    save_path = f"../output/{args.dataset}/{args.method}/{args.model}"
    os.makedirs(save_path, exist_ok=True)

    all_result = []

    if args.resume_path:
        with open(args.resume_path, "r", encoding="utf-8") as fin:
            resume_data = [json.loads(i) for i in fin.readlines()]
            all_result = resume_data
            filepath = args.resume_path
    else:
        resume_data = []
        filepath = (
            f"{save_path}/topk-{args.retrieve_top_k}_{formatted_time}.jsonl"
            if args.method != "note"
            else f"{save_path}/topk-{args.retrieve_top_k}__max_step-{args.max_step}__max_fail_step-{args.max_fail_step}__{formatted_time}.jsonl"
        )
    logger.info(f"The predicted results will be saved in '{filepath}'.")
    last_id = len(resume_data)
    batch_size = args.MaxClients
    start = time.time()
    logger.info("start predicting ...")
    for i in tqdm(range(last_id, len(qa_data), batch_size)):
        pool = ThreadPool(processes=args.MaxClients)
        current_batch = qa_data[i : i + batch_size]
        tasks = [
            (idx + i, doc_cell, args) for idx, doc_cell in enumerate(current_batch)
        ]

        results = pool.starmap(process_doc_cell, tasks)
        pool.close()
        pool.join()

        for result in results:
            if result:
                all_result.append(result)
                with open(filepath, "a", buffering=1) as fout:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info("start evaluating ...")

    predictions = [data["output"] for data in all_result]
    answers = [data["answer"] for data in all_result]

    if "asqa" in args.dataset:
        eval_result = eval_asqa(all_result)
    elif "crud" in args.dataset:
        eval_result = None
    else:
        acc = acc_score(predictions, answers)
        f1 = F1_scorer(predictions, answers)
        em = compute_exact(predictions, answers)
        eval_result = {"Acc": acc, "F1": f1, "EM": em}

    if eval_result:
        with open(filepath, "a", buffering=1) as fout:
            fout.write(json.dumps(eval_result, ensure_ascii=False) + "\n")

    logger.info(f"eval result: {eval_result}")
    logger.info(f"Spend time: {time.time() - start}")
