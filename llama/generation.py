# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.5,
        eos_token_id: int = -100,
    ) -> List[str]:
        bsz = len(prompts)
        
        # if "Question 3.2" in prompts[0]:
        #     print("=========")
        #     print(prompts[0])
        #     print("=========")
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        # print(prompts)
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        # if "Question 3.2" in prompts[0]:
        #     print("=========")
        #     print(len(prompt_tokens[0]))
        #     print("=========")

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t)[:params.max_seq_len].long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        # hs = []
        ends = torch.zeros(bsz).bool().cuda()
        
        # if "Question 3.2" in prompts[0]:
        #     print(tokens)

        for cur_pos in range(start_pos, total_len):

            logits, h = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            # hs.append(h)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            # if "Question 3.2" in prompts[0]:
            #     print(f"cur_pos: {cur_pos}, token: {next_token[0]}, ends: {ends}, eos: {eos_token_id}, status: {(next_token == eos_token_id)}")
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            ends = ends | (next_token == eos_token_id)
            # print(f"cur_pos: {cur_pos}, token: {next_token[0]}, ends: {ends}, eos: {eos_token_id}, status: {(next_token == eos_token_id)}")
            
            if ends.all():
                break
        # concat_h = torch.cat(hs, dim=1)
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[:params.max_seq_len]
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # replace pad with eos
            t = [x if x != self.tokenizer.pad_id else self.tokenizer.eos_id for x in t]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            try:
                t = t[: t[len(prompt_tokens[i]):].index(eos_token_id) + len(prompt_tokens[i]) + 1]
            except ValueError:
                pass
            # print(t)
            decoded.append(self.tokenizer.decode(t))
        return decoded# , concat_h


    @torch.no_grad()
    def encode(
        self,
        prompts: List[str],
        max_length: int,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        # total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
        if max_length != -1:
            assert max_length >= max_prompt_size
        else:
            max_length = max_prompt_size
        
        tokens = torch.full((bsz, max_length), self.tokenizer.eos_id).cuda().long()
        # this is to make sure that the index is legal
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        # print(tokens)
        logits, h = self.model.forward(tokens, prev_pos)
        return logits, h


    @torch.no_grad()
    def generate_with_prob(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        eos_token_id: int = -100,
        max_eos_cnt: int = 1,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        # print(prompts)
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t)[:params.max_seq_len].long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        # hs = []
        ends = torch.zeros(bsz).bool().cuda()
        
        # if "Question 3.2" in prompts[0]:
        #     print(tokens)

        eos_cnt = torch.zeros(bsz).long().cuda()
        return_probs = []
        for cur_pos in range(start_pos, total_len):

            logits, h = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            # hs.append(h)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
                # probs: [bsz, vocab_size]
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            
            # save the prob
            return_probs.append(probs[:, next_token].diag()) # [bsz]

            # print(f"cur_pos: {cur_pos}, token_id: {next_token[:]}, prob: {probs[:, next_token].diag().log()[:]}, token: {self.tokenizer.decode(next_token[:].tolist())}")


            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            
            eos_cnt = eos_cnt + (next_token == eos_token_id).long()
            # ends = ends | (next_token == eos_token_id)
            # print(f"cur_pos: {cur_pos}, token: {next_token[0]}, ends: {ends}, eos: {eos_token_id}, status: {(next_token == eos_token_id)}")
            
            # if every sample has reached max count of eos, break
            if (eos_cnt >= max_eos_cnt).all():
                break
            
        return_probs = torch.stack(return_probs, dim=1) # [bsz, max_gen_len]
        # concat_h = torch.cat(hs, dim=1)
        decoded = []

        log_prob = torch.log(return_probs)
        
        mask = torch.zeros_like(log_prob)
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[:params.max_seq_len]
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # replace pad with eos
            t = [x if x != self.tokenizer.pad_id else self.tokenizer.eos_id for x in t]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            try:
                first_eos = t[len(prompt_tokens[i]):].index(eos_token_id) + len(prompt_tokens[i]) + 1
                t = t[: t[first_eos:].index(eos_token_id, 1) + first_eos + 1]
            except ValueError:
                pass
            
            mask[i, :len(t)-start_pos] = 1
            decoded.append(self.tokenizer.decode(t))
        log_prob = log_prob * mask
        # print(log_prob.shape)
        # print(log_prob)
        return decoded, log_prob.sum(dim=1) / mask.sum(dim=1)


    @torch.no_grad()
    def get_ll(
        self,
        prefix: str,
        prompts: List[str],
    ) -> List[str]:
        params = self.model.params
        bsz = len(prompts)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        prefix_tokens = self.tokenizer.encode(prefix, bos=True, eos=False)
        # print(prompts)
        prompts_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        print("prefix length:", len(prefix_tokens))
        for prompt_tokens in prompts_tokens:
            print("prompt length:", len(prompt_tokens))
            assert prompt_tokens[: len(prefix_tokens)] == prefix_tokens


        min_prompt_size = min([len(t) for t in prompts_tokens])
        max_prompt_size = max([len(t) for t in prompts_tokens])

        total_len = max_prompt_size

        tokens = torch.full((bsz, total_len), self.tokenizer.eos_id).cuda().long()

        for k, t in enumerate(prompts_tokens):
            tokens[k, : len(t)] = torch.tensor(t)[:params.max_seq_len].long()

        _, h = self.model.forward(tokens[:, :], 0)
        logits = self.model.output(h)
        acc_probs = torch.zeros(bsz).cuda()
        for i in range(len(prefix_tokens), max_prompt_size):
            probs = torch.softmax(logits[:, i - 1, :], dim=-1)
            for j in range(bsz):
                if tokens[j, i] != self.tokenizer.eos_id:
                    acc_probs[j] += torch.log(probs[j, tokens[j, i]])

        return acc_probs.cpu().numpy()# , concat_h

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


