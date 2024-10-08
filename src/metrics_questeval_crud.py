import os
import re
import json
import jieba
import numpy as np
from loguru import logger
from collections import Counter
import argparse
import os
import backoff
from multiprocessing.pool import ThreadPool
from utils import gpt_gen


@backoff.on_exception(backoff.expo, (Exception), max_time=100)
def call_gpt(prompt_file, variable_dict):
    with open(prompt_file, "r") as fin:
        prompt = fin.read().format(**variable_dict)
    res = gpt_gen(args.model, prompt)
    assert res is not None
    return res


def compute_f1(a_gold, a_pred):
    gold_toks = list(jieba.cut(a_gold))
    pred_toks = list(jieba.cut(a_pred))
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return (
            int(gold_toks == pred_toks),
            int(gold_toks == pred_toks),
            int(gold_toks == pred_toks),
        )
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def word_based_f1_score(a_gold_list, a_pred_list):
    f1_list, precision_list, recall_list = [], [], []
    for a_gold, a_pred in zip(a_gold_list, a_pred_list):
        f1, precision, recall = compute_f1(a_gold, a_pred)
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
    return np.mean(f1_list), np.mean(precision_list), np.mean(recall_list)


def question_answer(context, question):
    origin_answers = call_gpt(
        f"../prompts/crud/quest_eval_answer.txt",
        {"context": context, "questions": question},
    )
    answers = origin_answers
    pattern = r"<response>\n(.*?)\n</response>"
    real_answers = re.findall(pattern, answers, re.DOTALL)
    return real_answers


def get_QA_pair(data_point: dict, quest_gt_save: dict):
    generated_text = data_point["output"]
    questions_gt = quest_gt_save["question"]
    answers_gt4gt = quest_gt_save["answers"]
    answers_gm4gt = question_answer(generated_text, questions_gt)
    return questions_gt, answers_gt4gt, answers_gm4gt


def quest_eval(data_point: dict, quest_gt: dict):
    try:
        questions_gt, answers_gt4gt, answers_gm4gt = get_QA_pair(data_point, quest_gt)
        quest_eval_save = {
            "questions_gt": questions_gt,
            "answers_gt4gt": answers_gt4gt,
            "answers_gm4gt": answers_gm4gt,
        }

        indices = [i for i, x in enumerate(answers_gt4gt) if x != "无法推断"]
        answers_gm4gt = [answers_gm4gt[i] for i in indices]
        answers_gt4gt = [answers_gt4gt[i] for i in indices]

        if len(answers_gm4gt) == 0:
            return 0, 0, 0, 0, quest_eval_save

        undetermined_ratio = answers_gm4gt.count("无法推断") / len(answers_gm4gt)
        quest_recall = 1 - undetermined_ratio

        indices = [i for i, x in enumerate(answers_gm4gt) if x != "无法推断"]
        answers_gm4gt = [answers_gm4gt[i] for i in indices]
        answers_gt4gt = [answers_gt4gt[i] for i in indices]

        if answers_gm4gt == []:
            return 0, 0, 0, 0, quest_eval_save

        quest_avg_token_f1, quest_avg_token_precision, quest_avg_token_recall = (
            word_based_f1_score(answers_gt4gt, answers_gm4gt)
        )

    except Exception as e:
        logger.warning(repr(e))
        quest_eval_save = {}
        quest_eval_save["questions_gt"] = []
        quest_eval_save["answers_gt4gt"] = []
        quest_eval_save["answers_gm4gt"] = []
        return 0, 0, 0, 0, quest_eval_save

    return (
        quest_avg_token_f1,
        quest_avg_token_precision,
        quest_avg_token_recall,
        quest_recall,
        quest_eval_save,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt-3.5-turbo-0125"],
        default="gpt-3.5-turbo-0125",
        help="Model to use",
    )
    parser.add_argument(
        "--MaxClients", type=int, default=10, help="Number of concurrent clients"
    )
    parser.add_argument("--eval_path", type=str, help="eval path")
    args = parser.parse_args()
    save_path = os.path.join(os.path.dirname(args.eval_path), "metric_questeval")
    os.makedirs(save_path, exist_ok=True)
    with open(args.eval_path, "r") as fin:
        generate_answer = [json.loads(d) for d in fin]
    eval_name = args.eval_path.split("/")[-1].split(".")[0]
    pattern = r"crud_[123]doc"
    data_name = re.findall(pattern, args.eval_path)[0]
    with open(f"../data/eval/{data_name}/QuestAnswerQA_quest_gt.json", "r") as f:
        quest_gt = json.load(f)

    res_cell_list = []
    all_quest_avg_token_f1, all_quest_recall = [], []
    all_quest_avg_token_precision, all_quest_avg_token_recall = [], []
    pool = ThreadPool(processes=args.MaxClients)
    tasks = []
    ids = []
    for idx, generate_cell in enumerate(generate_answer):
        tasks.append((generate_cell, quest_gt[idx]))
    results = pool.starmap(quest_eval, tasks)
    pool.close()
    pool.join()

    for index, result in enumerate(results):
        (
            quest_avg_token_f1,
            quest_avg_token_precision,
            quest_avg_token_recall,
            quest_recall,
            quest_eval_save,
        ) = result
        all_quest_avg_token_f1.append(quest_avg_token_f1)
        all_quest_avg_token_precision.append(quest_avg_token_precision)
        all_quest_avg_token_recall.append(quest_avg_token_recall)
        all_quest_recall.append(quest_recall)

        res_cell = {
            "id": index,
            "quest_avg_token_f1": quest_avg_token_f1,
            "quest_avg_token_precision": quest_avg_token_precision,
            "quest_avg_token_recall": quest_avg_token_recall,
            "quest_recall": quest_recall,
            "quest_eval_save": quest_eval_save,
        }
        res_cell_list.append(res_cell)

    valid_qa_count = 0
    for res in res_cell_list:
        if res["quest_eval_save"]["questions_gt"]:
            valid_qa_count += 1
    avg_score = {
        "all_quest_avg_token_f1": sum(all_quest_avg_token_f1) / valid_qa_count,
        "all_quest_avg_token_precision": sum(all_quest_avg_token_precision)
        / valid_qa_count,
        "all_quest_avg_token_recall": sum(all_quest_avg_token_recall) / valid_qa_count,
        "all_quest_recall": sum(all_quest_recall) / valid_qa_count,
    }
    res_cell_list.append(avg_score)
    print("eval result", res_cell_list)
    with open(f"{save_path}/{eval_name}_questeval.json", "w", buffering=1) as f:
        f.write(json.dumps(res_cell_list, ensure_ascii=False, indent=2))
