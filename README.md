# RAP: Reasoning via Planning
Source code for the paper [Reasoning with Language Model is Planning with World Model](https://arxiv.org/abs/2305.14992)
![Figure](assets/figure_full.png)

# Requirements

- Our experiments are conducted with LLaMA-33B, which takes at least 4 GPUs of 24GB memory each. The code also supports smaller LLaMA models, but other LLMs (e.g. those from Huggingface) are not tested.

- Please acquire the checkpoints of LLaMA from MetaAI following the [LLaMA official repo](https://github.com/facebookresearch/llama). Also make sure to install required pachages to run the code in the official repo. 

- We evaluate RAP on the Blocksworld with code from [GPT-Plan-Benchmark](https://github.com/karthikv792/gpt-plan-benchmark). If you would like to reproduce the results of method on Blocksworld, please make sure to install required pachages following their instruction first. 

# Commands

## Blocksworld

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --master_port 1034 --nproc_per_node 4 run_blocksworld.py --task mcts --model_name LLaMA --ckpt_path $LLAMA_CKPTS/30B --verbose True --data data/blocksworld/step_4.json --max_depth 4 --name run_4_May26_max_depth_4_alpha_05_rollouts_10 --rollouts 10 
```

## Other datasets
To be updated...