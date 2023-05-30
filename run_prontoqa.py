import pickle
from datetime import datetime

from rap.prontoqa_mcts import reasoning_mcts_search
from rap.models import QueryLlama
from rap.utils.prontoqa import get_prontoqa_dataset, judge_prontoqa_answer, judge_prontoqa_proof

from typing import Tuple
import os
import sys
import torch
import torch.distributed
import torch.backends.cudnn
import fire
import time
import json
import random
import numpy as np
from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from tqdm import tqdm
from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int, max_batch_size: int) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    # print(checkpoints)
    assert (
            world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=2048, max_batch_size=max_batch_size, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args).cuda().half()
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main_mcts(llama_ckpt='llama-ckpts/13B',
              dataset_file_name='data/prontoqa/345hop_random_true.json',
              agent_prompts='data/prontoqa/prompts/next_step_examples.json',
              wm_transit_prompts='data/prontoqa/prompts/state_transit_examples.json',
              wm_output_prompts='data/prontoqa/prompts/output_examples.json',
              wm_finish_prompts='data/prontoqa/prompts/finish_examples.json',
              wm_valid_prompts='data/prontoqa/prompts/valid_examples.json',
              max_batch_size=3,
              max_response_length=200,
              n_sample_subquestion=9,
              mcts_rollouts=20,
              temperature=0.5,
              max_depth=7,
              w_exp=1,
              resume=0,
              log_dir=None):
    if log_dir is None:
        log_dir = f'logs/prontoqa_mcts_{llama_ckpt.split("/")[-1]}/{datetime.now().strftime("%Y-%m%d-%H%M")}'
    os.makedirs(log_dir, exist_ok=True)

    # set random seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    local_rank, world_size = setup_model_parallel()

    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')
        log_file = None
    else:
        log_file = None

    with open(agent_prompts) as f:
        agent_prompts = json.load(f)
    with open(wm_transit_prompts) as f:
        wm_transit_prompts = json.load(f)
    with open(wm_output_prompts) as f:
        wm_output_prompts = json.load(f)
    with open(wm_finish_prompts) as f:
        wm_finish_prompts = json.load(f)
    with open(wm_valid_prompts) as f:
        wm_valid_prompts = json.load(f)

    examples = get_prontoqa_dataset(dataset_file_name)

    os.makedirs(log_dir, exist_ok=True)

    # the parent directory of the checkpoint directory
    tokenizer_path = os.path.join(os.path.dirname(llama_ckpt), "tokenizer.model")
    llama = load(llama_ckpt, tokenizer_path, local_rank, world_size, max_batch_size)

    world_model = QueryLlama(llama, max_response_length=max_response_length, log_file=log_file)

    total_correct: list[int] = [0] * mcts_rollouts
    total_proof_correct = [0] * mcts_rollouts
    none_count = 0
    for i, example in enumerate((pbar := tqdm(examples, disable=local_rank > 0, position=1))):
        if i < resume:
            continue
        facts = example['facts']
        target = example['target']
        answer = example['answer']
        trajs, tree, trees, outputs = reasoning_mcts_search(facts, target,
                                                            agent_prompts, wm_transit_prompts, wm_output_prompts,
                                                            wm_finish_prompts, wm_valid_prompts,
                                                            world_model,
                                                            n_sample_subquestion=n_sample_subquestion,
                                                            mcts_rollouts=mcts_rollouts,
                                                            temperature=temperature,
                                                            max_depth=max_depth,
                                                            w_exp=w_exp,
                                                            eos_token_id=world_model.tokenizer.encode('\n', bos=False, eos=False)[-1],
                                                            logging=False)
        if local_rank == 0:
            json_logs = []
            for rollout, (output, traj) in enumerate(zip(outputs, trajs)):
                json_logs.append({
                    'rollout': rollout + 1,
                    'facts': facts,
                    'target': target,
                    'answer': answer,
                    'output': output,
                    'correct': (correct := judge_prontoqa_answer(answer, output)),
                    'proof_correct': (proof_correct := correct and judge_prontoqa_proof(traj, example['proof'])),
                    'traj': traj,
                    'ref_traj': example['proof'],
                })
                total_correct[rollout] += correct
                total_proof_correct[rollout] += proof_correct
            none_count += output == 'none'
            with open(os.path.join(log_dir, f'{i:04d}.json'), 'w') as f:
                json.dump(json_logs, f, indent=2)
            with open(os.path.join(log_dir, f'{i:04d}.tree'), 'w') as f:
                f.write(tree)
            with open(os.path.join(log_dir, f'{i:04d}.pkl'), 'wb') as f:
                pickle.dump(trees, f)
            tqdm.write(' '.join(f'{c/(i+1-resume):0.3f}' for c in total_correct))
            pbar.set_description(f'{total_correct[-1]}/{i+1-resume}={total_correct[-1]/(i+1-resume):.2f} {none_count} {total_proof_correct[-1]/(i+1-resume):}')


if __name__ == '__main__':
    fire.Fire(main_mcts)
