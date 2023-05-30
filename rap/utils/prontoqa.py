import json


def get_prontoqa_dataset(file_name):
    with open(file_name) as f:
        examples = json.load(f)
    examples = [{
        'facts': v['test_example']['question'],
        'target': v['test_example']['query'],
        'answer': v['test_example']['answer'],
        'proof': v['test_example']['chain_of_thought'],
    } for v in examples.values()]
    return examples


def judge_prontoqa_answer(answer, output):
    return answer.lower() == output.lower()


def judge_prontoqa_proof(traj, ref_traj):
    if len(traj) != len(ref_traj):
        return False
    for si, sj in zip(traj[::2], ref_traj[::2]):
        if si != sj:
            return False
    return True
