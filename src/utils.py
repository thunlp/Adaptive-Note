import os, time
import torch
from openai import OpenAI
import random
import numpy as np
import yaml

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
os.environ["OPENAI_API_KEY"] = config["model"]["OPENAI_API_KEY"]
os.environ["OPENAI_BASE_URL"] = config["model"]["OPENAI_BASE_URL"]
client = OpenAI()


def gpt_gen(model, content):
    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.1,
            top_p=0.9,
            max_tokens=1280,
            messages=[{"role": "user", "content": content}],
        )
        return completion.choices[0].message.content

    except Exception as e:
        print(f"An error occurred: {e}")
        time.sleep(0.5)

    return None


def llama_gen(model, tokenizer, prompt_file, variable_dict):
    with open(prompt_file, "r") as fin:
        prompt = fin.read()
    prompt = prompt.format(**variable_dict)
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    outputs = model.generate(
        input_ids,
        eos_token_id=terminators,
        temperature=0.1,
        top_p=0.9,
        max_new_tokens=1280,
    )
    response = tokenizer.decode(
        outputs[0][input_ids.shape[-1] :], skip_special_tokens=True
    )
    return response


def qwen_gen(model, tokenizer, prompt_file, variable_dict):
    with open(prompt_file, "r") as fin:
        prompt = fin.read()
    prompt = prompt.format(**variable_dict)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        model_inputs.input_ids, temperature=0.1, top_p=0.9, max_new_tokens=1280
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
