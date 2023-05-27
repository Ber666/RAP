# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math
from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import copy
import json
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)

import re

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 1024


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    # @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)  # (bsz, partial_seqlen, dim)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float(), h
    
class FunctionLM(nn.Module):
    def __init__(self, base_model, tokenizer, func_dict, load_path=None, inference_mode="func_embedding"):
        super().__init__()
        self.inference_mode = inference_mode
        self.model = base_model
        self.tokenizer = tokenizer
        self.func_dict = func_dict
        self.func_list = {v: k for k, v in func_dict.items()}
        # self.func_embed = ColumnParallelLinear(
        #     base_model.params.dim, len(func_list), bias=False, init_method=lambda x: x
        # )
        self.func_embed = nn.Linear(base_model.params.dim, len(func_dict), bias=False).to("cuda")
        if load_path is not None and load_path != "None": # load func_embed weights
            embedding = torch.load(load_path)
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.to("cuda")
                embedding = {"weight": embedding}
            self.func_embed.load_state_dict(embedding)
        
        # set the basemodel to eval mode and freeze the weights
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.logits_bias = 0

    def set_bias(self, logits_bias):
        self.logits_bias = logits_bias

    def get_loss(self, raw_inputs):
        
        assert len(raw_inputs) == 1
        raw_inputs = raw_inputs[0]

        # inputs: starts with <bos>, ends without <eos>, (bsz, seqlen)
        # labels: starts without <bos>, ends with <eos>, (bsz, seqlen)
        with torch.no_grad():
            # prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=True) for x in raw_inputs]
            raw_input_ids = torch.tensor(self.tokenizer.encode(raw_inputs["question"], bos=True, eos=True))[:]
            labels = torch.tensor(self.tokenizer.encode(raw_inputs["question"], bos=True, eos=True))[:]
            for s, t, eq in zip(raw_inputs["start_token_idx"], raw_inputs["end_token_idx"], raw_inputs["tar_eq"]):
                op = re.search(r"(<.*?>)", eq).group(1)
                # print(op)
                labels[s] = self.func_dict[op] + 32000
                labels[s+1: t] = -100
            # labels = labels[1:]

            inputs = raw_input_ids[:-1].expand(1, -1).to("cuda")
            labels = labels[1:].expand(1, -1).to("cuda")

            last_logits, h = self.model(inputs, 0) # h: (bsz, seqlen, dim)
            token_logits = self.model.output(h) # (bsz, seqlen, vocab_size)
            # print(h.device)
        
        func_logits = self.func_embed(h.float()) # (bsz, seqlen, len(func_list))
        
        concat_logits = torch.cat([token_logits, func_logits], dim=-1) # (bsz, seqlen, vocab_size + len(func_list))
        loss = F.cross_entropy(concat_logits.view(-1, concat_logits.shape[-1]), labels.view(-1), ignore_index=-100)
        # check p, r, f1 for each function
        pred = torch.argmax(concat_logits, dim=-1) # (bsz, seqlen)
        pred = pred.view(-1)
        labels = labels.view(-1)
        label_funcs = [labels == self.func_dict[op] + 32000 for op in self.func_dict.keys()]
        pred_funcs = [pred == self.func_dict[op] + 32000 for op in self.func_dict.keys()]
        label_funcs = torch.stack(label_funcs, dim=0)
        pred_funcs = torch.stack(pred_funcs, dim=0)
        # (len(func_list), seqlen)
        # true positive
        tp = torch.sum(label_funcs * pred_funcs, dim=-1).detach().cpu().numpy()
        pred_funcs = torch.sum(pred_funcs, dim=-1).detach().cpu().numpy()
        # strange bug: if I use `pred` as variable name, the returned results will be all zeros
        true = torch.sum(label_funcs, dim=-1).detach().cpu().numpy()
        results = {
            "tp": tp,
            "pred": pred_funcs,
            "true": true
        }


        return loss, results
    
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop_token: List[int] = [], # 29897: ), 3892: )=
        return_top: int = 0
    ) -> List[str]:
        bsz = len(prompts)

        assert bsz == 1
        
        generation_log = [] # (token, [(token, logits, prob)])
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        hs = []
        for cur_pos in range(start_pos, total_len):
            _, h = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            logits_token = self.model.output(h[:, -1, :]).float() # (bsz, vocab_size)
            logits_func = self.func_embed(h[:, -1, :].float()) # (bsz, len(func_list))
            if self.inference_mode != "func_embedding":
                logits_func = torch.zeros_like(logits_func) - 1e5

            logits_func += self.logits_bias
            logits = torch.cat([logits_token, logits_func], dim=-1) # (bsz, vocab_size + len(func_list))
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if the prompt is ended
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            if return_top > 0:
                generation_log.append(
                    (next_token[0].item(), [(i.item(), logits[0, i.item()].item()) for i in torch.argsort(logits[0, :], descending=True)[:return_top]])
                )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            if next_token[0] >= 32000 or next_token[0] in stop_token:
                break

        # concat_h = torch.cat(hs, dim=1)
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            if t[cur_pos] >= 32000:
                decoded.append(self.tokenizer.decode(t[:cur_pos]) + self.func_list[t[cur_pos] - 32000] + "(")
            else:
                decoded.append(self.tokenizer.decode(t[:cur_pos + 1]))
        if return_top > 0:
            return decoded, generation_log
        else:
            return decoded# , concat_h
    
def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token



class EmbedGenLM(nn.Module):
    def __init__(self, base_model, tokenizer, d_model, generation_args, feature, length_map, inference_mode="func_embedding", shuffle=False, gen_load_path=None):
        super().__init__()
        self.inference_mode = inference_mode
        self.model = base_model
        self.tokenizer = tokenizer
        self.attention_layers = torch.nn.ModuleList()
        self.generation_args = generation_args
        if generation_args["model"] == "linear":
            self.linear = nn.Linear(d_model, d_model).to("cuda")
            if gen_load_path is not None:
                self.linear.load_state_dict(torch.load(gen_load_path))
        elif generation_args["model"] == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(self.model.params.dim, nhead=generation_args["n_head"], batch_first=True).to("cuda")
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=generation_args["n_layer"]).to("cuda")
            if gen_load_path is not None:
                self.transformer_encoder.load_state_dict(torch.load(gen_load_path))

        elif generation_args["model"] == "embedding":
            self.embedding = nn.Linear(generation_args["n_apis"], d_model, bias=False).to("cuda")
            if gen_load_path is not None:
                self.embedding.load_state_dict(torch.load(gen_load_path))

        if feature != None:
            self.feature = feature.float().to("cuda")  # (n, l, d)
            # random permute the feature
            if shuffle:
                self.feature = self.feature[torch.randperm(self.feature.shape[0])]
                # sanity check

        if length_map != None:
            self.length_map = length_map
        # set the basemodel to eval mode and freeze the weights
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        print("model dim:", d_model)

    def save_gen_model(self, save_path):
        if self.generation_args["model"] == "linear":
            torch.save(self.linear.state_dict(), save_path)
        elif self.generation_args["model"] == "transformer":
            torch.save(self.transformer_encoder.state_dict(), save_path)
        elif self.generation_args["model"] == "embedding":
            torch.save(self.embedding.state_dict(), save_path)

    def load_weight(self, load_path):
        if self.generation_args["model"] == "linear":
            self.linear.load_state_dict(torch.load(load_path))
        elif self.generation_args["model"] == "transformer":
            self.transformer_encoder.load_state_dict(torch.load(load_path))
        elif self.generation_args["model"] == "embedding":
            self.embedding.load_state_dict(torch.load(load_path))

    def get_embedding(self, api_texts):
        # first encode the api text
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in api_texts]

        length = [len(x) for x in prompt_tokens]

        tokens = torch.full((len(prompt_tokens), max(length)), self.tokenizer.eos_id).cuda().long()

        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        
        print(tokens.shape)
        
        feature = []
        for i in range(len(api_texts)):
            logits, f = self.model.forward(tokens[i:i+1], 0)
            feature.append(f[0])

        feature = torch.stack(feature).float()
        
        if self.generation_args["model"] == "linear":
            encodings = self.linear(torch.stack([f[l-1] for f, l in zip(feature, length)]))
        
        elif self.generation_args["model"] == "transformer":
            # encoding_list = []
            # for f, l in zip(feature, length):
            #     f_ = self.transformer_encoder(f[:l].unsqueeze(0))
            #     encoding_list.append(f_[0, l-1, :])
            # encodings = torch.stack(encoding_list)

            # src_key_padding_mask (n, l)
            src_key_padding_mask = torch.zeros(feature.shape[0], feature.shape[1], dtype=torch.bool).to("cuda")
            for i, l in enumerate(length):
                src_key_padding_mask[i, :l] = True

            encodings = self.transformer_encoder(feature, src_key_padding_mask=src_key_padding_mask.logical_not())
            encodings = torch.stack([f[l-1] for f, l in zip(encodings, length)])

        elif self.generation_args["model"] == "embedding":
            raise NotImplementedError

        return encodings

    def get_loss(self, raw_inputs):
        
        assert len(raw_inputs) == 1
        raw_inputs = raw_inputs[0]
        neg_sample_idx = raw_inputs["neg_samples"]
        feature = self.feature[[raw_inputs["api_idx"]] + neg_sample_idx]  # (n, l, d)
        length = [self.length_map[raw_inputs["api_idx"]]] + [self.length_map[x] for x in neg_sample_idx]
        
        # mask = torch.zeros_like(feature).to("cuda")
        # for i, l in enumerate(length):
        #     mask[i, :l] = 1
        # encodings = self.function_encoder(feature, mask=mask)

        if self.generation_args["model"] == "linear":
            encodings = self.linear(torch.stack([f[l-1] for f, l in zip(feature, length)]))
        elif self.generation_args["model"] == "transformer":
            # encoding_list = []
            # for f, l in zip(feature, length):
            #     f_ = self.transformer_encoder(f[:l].unsqueeze(0))
            #     encoding_list.append(f_[0, l-1, :])
            # encodings = torch.stack(encoding_list)

            # src_key_padding_mask (n, l)
            src_key_padding_mask = torch.zeros(feature.shape[0], feature.shape[1], dtype=torch.bool).to("cuda")
            for i, l in enumerate(length):
                src_key_padding_mask[i, :l] = True

            encodings = self.transformer_encoder(feature, src_key_padding_mask=src_key_padding_mask.logical_not())
            encodings = torch.stack([f[l-1] for f, l in zip(encodings, length)])

        elif self.generation_args["model"] == "embedding":
            idx_list = torch.tensor([raw_inputs["api_idx"]] + neg_sample_idx).to("cuda")
            # (bsz, ) -> (bsz, n_apis) one-hot
            idx_list = F.one_hot(idx_list, num_classes=self.generation_args["n_apis"]).float()
            encodings = self.embedding(idx_list)
        
        # inputs: starts with <bos>, ends without <eos>, (bsz, seqlen)
        # labels: starts without <bos>, ends with <eos>, (bsz, seqlen)
        with torch.no_grad():
            # prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=True) for x in raw_inputs]
            raw_input_ids = torch.tensor(self.tokenizer.encode(raw_inputs["text"], bos=True, eos=True))[:]
            labels = torch.tensor(self.tokenizer.encode(raw_inputs["text"], bos=True, eos=True))[:]
            for s, t, eq in zip(raw_inputs["start_token_idx"], raw_inputs["end_token_idx"], raw_inputs["return"]):
                # print(op)
                labels[s] = 32000
                labels[s+1: t] = -100
            # labels = labels[1:]

            inputs = raw_input_ids[:-1].expand(1, -1).to("cuda")
            labels = labels[1:].expand(1, -1).to("cuda")

            last_logits, h = self.model(inputs, 0) # h: (bsz, seqlen, dim)
            token_logits = self.model.output(h) # (bsz, seqlen, vocab_size)
            # print(h.device)
        
        func_logits = h[0].float().mm(encodings.T)# seq_len, dim @ dim, n_apis
        concat_logits = torch.cat([token_logits, func_logits.unsqueeze(0)], dim=-1) # (bsz, seqlen, vocab_size + len(func_list))
        loss = F.cross_entropy(concat_logits.view(-1, concat_logits.shape[-1]), labels.view(-1), ignore_index=-100)
        # check p, r, f1 for each function
        pred = torch.argmax(concat_logits, dim=-1) # (bsz, seqlen)
        pred = pred.view(-1)
        labels = labels.view(-1)
        # label_funcs = [labels == self.func_dict[op] + 32000 for op in self.func_dict.keys()]
        # pred_funcs = [pred == self.func_dict[op] + 32000 for op in self.func_dict.keys()]
        # label_funcs = torch.stack(label_funcs, dim=0)
        # pred_funcs = torch.stack(pred_funcs, dim=0)
        # (len(func_list), seqlen)
        # true positive
        tp = torch.sum((labels == 32000) * (pred==32000), dim=-1).detach().cpu().numpy()
        pred_func0 = torch.sum(pred == 32000).detach().cpu().numpy()
        pred_funcs = torch.sum(pred >= 32000).detach().cpu().numpy()
        place_funcs = torch.sum((labels == 32000) * (pred >= 32000), dim=-1).detach().cpu().numpy()
        # strange bug: if I use `pred` as variable name, the returned results will be all zeros
        # true = torch.sum(label_funcs, dim=-1).detach().cpu().numpy()
        
        results = {
            "tp": tp,
            "place_funcs": place_funcs,
            "pred_func0": pred_func0,
            "pred_funcs": pred_funcs
        }


        return loss, results
    
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop_token: List[int] = [], # 29897: ), 3892: )=
    ) -> List[str]:
        bsz = len(prompts)
        assert bsz == 1
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        hs = []
        for cur_pos in range(start_pos, total_len):
            _, h = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            logits_token = self.model.output(h[:, -1, :]).float() # (bsz, vocab_size)
            logits_func = self.func_embed(h[:, -1, :].float()) # (bsz, len(func_list))
            if self.inference_mode != "func_embedding":
                logits_func = torch.zeros_like(logits_func) - 1e5

            logits = torch.cat([logits_token, logits_func], dim=-1) # (bsz, vocab_size + len(func_list))
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
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            if next_token[0] >= 32000 or next_token[0] in stop_token:
                break

        # concat_h = torch.cat(hs, dim=1)
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            if t[cur_pos] >= 32000:
                decoded.append(self.tokenizer.decode(t[:cur_pos]) + self.func_list[t[cur_pos] - 32000] + "(")
            else:
                decoded.append(self.tokenizer.decode(t[:cur_pos + 1]))
        return decoded# , concat_h
    
def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token