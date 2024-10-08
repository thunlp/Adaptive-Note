import json
import string
from collections import Counter
import re
import numpy as np


def acc_score(predictions, answers):
    num_correct = 0
    for id, answer in enumerate(answers):
        pred = predictions[id]
        correctness = (
            "True" if any(ans.lower() in pred.lower() for ans in answer) else "False"
        )
        if correctness == "True":
            num_correct += 1
        else:
            pass
    acc = num_correct / len(answers)
    return acc


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def F1_scorer(predictions, answers):
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        for ground_truth in ground_truths:
            score = max(score, qa_f1_score(prediction, ground_truth))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def compute_exact(predictions, answers):
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        for ground_truth in ground_truths:
            score = max(
                score,
                int(normalize_answer(prediction) == normalize_answer(ground_truth)),
            )
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """

    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False


def compute_str_em(data):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """

    if "qa_pairs" not in data[0] or data[0]["qa_pairs"] is None:
        return 0, 0

    acc = []
    hit = []

    for item in data:
        loc_acc = []
        for qa_pair in item["qa_pairs"]:
            loc_acc.append(exact_presence(qa_pair["short_answers"], item["output"]))
        acc.append(np.mean(loc_acc))
        hit.append(int(np.mean(loc_acc) == 1))

    return 100 * np.mean(acc), 100 * np.mean(hit)


def eval_asqa(pred):
    raw_data_path = "../data/eval/asqa/asqa_eval_gtr_top100.json"
    with open(raw_data_path, "r") as f:
        raw_data = json.load(f)
    normalized_data = []
    for i in range(len(pred)):
        pred[i]["answer"] = pred[i]["output"].strip().split("\n")[0]
        pred[i]["answer"] = pred[i]["output"].replace("<|im_end|>", "")
    result = {}
    for id, data in enumerate(pred):
        normalized_data.append(
            {
                "question": raw_data[id]["question"],
                "answer": raw_data[id]["answer"],
                "qa_pairs": raw_data[id]["qa_pairs"],
                "output": data["answer"],
            }
        )
    result["str_em"], result["str_hit"] = compute_str_em(normalized_data)
    return result


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
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
    parser.add_argument("--pred_path", type=str, help="Location of the prediction file")
    args = parser.parse_args()

    outputs = []
    with open(args.pred_path, "r") as fin:
        for d in fin:
            pred = json.loads(d)
            if "id" in pred.keys():
                outputs.append(pred)
    if "asqa" in args.pred_path:
        result = eval_asqa(outputs)
        print("eval result:", result)
    else:
        with open(
            f"../data/eval/{args.dataset}/test.json", "r", encoding="utf-8"
        ) as f:  # dev_sample_10
            qa_data = json.load(f)

        predictions = [data["output"] for data in outputs]
        answers = [data["answer"] for data in outputs]
        acc = acc_score(predictions, answers)
        f1 = F1_scorer(predictions, answers)
        em = compute_exact(predictions, answers)

        print("ACC:", acc, "F1:", f1, "em:", em)
