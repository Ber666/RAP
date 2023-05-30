import json
import re
import os


def judge_answer_gsm8k(output, answer):
    output = re.findall('The answer is .*?([$ .0-9,\\-]+).*\\.', output)
    if len(output) == 0:
        output = ''
    else:
        output = output[-1].replace(',', '').replace('$', '').replace(' ', '')
    ret = output
    if '=' in output:
        output = output.split('=')[-1]
    try:
        output, answer = int(output), int(answer)
    except ValueError:
        try:
            output, answer = float(output), float(answer)
        except ValueError:
            pass
    return ret, output == answer


def get_gsm8k_dataset(split):
    with open(f'data/gsm8k/{split}.jsonl') as f:
        examples = [json.loads(l) for l in f]

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    print(f"{len(examples)} {split} examples")
    return examples
