# This code is adapted from the repository: https://github.com/facebookresearch/three_bricks
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List

import torch
import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WmGenerator():
    def __init__(self, 
            model: LlamaForCausalLM, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
        ):
        # model config
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.model = model
        self.max_seq_len = model.config.max_sequence_length if 'max_sequence_length' in model.config.to_dict() else 2048
        self.pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else -1
        self.eos_id = model.config.eos_token_id
        # watermark config
        self.ngram = ngram
        self.salt_key = salt_key
        self.seed = seed
        self.hashtable = torch.randperm(1000003)
        self.seeding = seeding 
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Sane version, in the end we only need a small permutation table."""
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)] 
    
    def get_seed_rng(
        self, 
        input_ids: torch.LongTensor
    ) -> int:
        """Seed RNG with hash of input_ids."""
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.salt_key + i.item()) % (2 ** 64 - 1)
        elif self.seeding == 'additive':
            seed = self.salt_key * torch.sum(input_ids).item()
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.salt_key * input_ids[0].item()
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.salt_key * input_ids)
            seed = torch.min(seed).item()
        return seed

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        return_aux: bool = False,
    ) -> List[str]:
        
        bsz = len(prompts)
        prompt_tokens = [self.tokenizer.encode(x, add_special_tokens=False) for x in prompts]
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)
        tokens = torch.full((bsz, total_len), self.pad_id).to(device).long()
        if total_len < max_prompt_size:
            print("some prompts are bigger than max sequence length")
            prompt_tokens = [t[:total_len] for t in prompt_tokens]
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.pad_id

        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            outputs = self.model.forward(
                tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=outputs.past_key_values if prev_pos > 0 else None
            )
            aux = {
                'ngram_tokens': tokens[:, cur_pos-self.ngram:cur_pos],
                'cur_pos': cur_pos,
            }
            next_toks = self.sample_next(outputs.logits[:, -1, :], aux, temperature, top_p)
            tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks)
            prev_pos = cur_pos

        decoded = []
        tokens = tokens.tolist()
        aux = []
        for i, t in enumerate(tokens):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            finish_reason = 'length'
            try:
                t = t[: t.index(self.eos_id)]
                finish_reason = 'eos'
            except ValueError:
                pass
            aux.append({
                't': t, 
                'finish_reason': finish_reason,
                'n_toks_gen': len(t) - len(prompt_tokens[i]),
                'n_toks_tot': len(t),
            })
            decoded.append(self.tokenizer.decode(t))

        if return_aux:
            return decoded, aux
        return decoded
    
    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        aux: Dict, # ngram_tokens (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
    ):
        """Vanilla sampling with temperature and top p."""
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1) # one hot of next token, ordered by original probs
            next_token = torch.gather(probs_idx, -1, next_token) # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token


class OpenaiGenerator(WmGenerator):
    """
    Generate text using LLaMA and Aaronson's watermarking method.
    From ngram tokens, select the next token based on the following:
    - hash the ngram tokens and get a seed
    - use the seed to generate V random number r between [0,1]
    - select argmax ( r^(1/p) )
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        

    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        aux: Dict, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
    ):
        ngram_tokens = aux['ngram_tokens']
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            for ii in range(ngram_tokens.shape[0]): # batch of texts
                # seed with hash of ngram tokens
                seed = self.get_seed_rng(ngram_tokens[ii])
                self.rng.manual_seed(seed)
                # generate rs randomly between [0,1]
                rs = torch.rand(self.vocab_size, generator=self.rng) # n
                rs = torch.Tensor(rs).to(probs_sort.device)
                rs = rs[probs_idx[ii]] 
                # compute r^(1/p)
                probs_sort[ii] = torch.pow(rs, 1/probs_sort[ii])
            # select argmax ( r^(1/p) )
            next_token = torch.argmax(probs_sort, dim=-1, keepdim=True)
            next_token = torch.gather(probs_idx, -1, next_token)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token
    

class StanfordGenerator(WmGenerator):
    """
    Generate text using LLaMA and Stanford's watermarking method.
    - 
    - select argmax ( r^(1/p) )
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng.manual_seed(self.seed)
        self.xi = torch.rand((self.ngram, self.vocab_size), generator=self.rng)  # ngram, vocab_size

    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        aux: Dict, # contains cur_pos int
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
    ):
        cur_pos = aux['cur_pos']
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True) # bsz vocab_size
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            # generate rs from xi and cur_pos
            rs = self.xi[cur_pos%self.ngram, :] # vocab_size
            rs = torch.Tensor(rs).to(probs_sort.device) # vocab_size
            rs = rs[probs_idx]  # bsz vocab_size
            # compute r^(1/p)
            probs_sort = torch.pow(rs, 1/probs_sort)  # bsz vocab_size
            # select argmax ( r^(1/p) )
            next_token = torch.argmax(probs_sort, dim=-1, keepdim=True) # bsz 1
            next_token = torch.gather(probs_idx, -1, next_token)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token


class MarylandGenerator(WmGenerator):
    """
    Generate text using LLaMA and Maryland's watemrarking method.
    From ngram tokens, select the next token based on the following:
    - hash the ngram tokens and get a seed
    - use the seed to partition the vocabulary into greenlist (gamma*V words) and blacklist 
    - add delta to greenlist words' logits
    """
    def __init__(self, 
            *args, 
            gamma: float = 0.5,
            delta: float = 1.0,
            test_mul: float = 0,
            **kwargs
        ):
        super().__init__(*args, **kwargs)        
        self.gamma = gamma
        self.delta = delta
        self.test_mul = test_mul

    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        aux: Dict, # ngram_tokens (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
    ):
        ngram_tokens = aux['ngram_tokens']
        logits = self.logits_processor(logits, ngram_tokens)
        bsz, vocab_size = logits.shape
        if temperature > 0:
            if self.test_mul > 0:
                logits = torch.softmax(logits / temperature, dim=-1)
                probs_sort, probs_idx = torch.sort(logits, dim=-1, descending=True)
                probs_sum = torch.cumsum(probs_sort, dim=-1)
                mask = probs_sum - probs_sort > top_p
                probs_sort[mask] = 0.0
                probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
                for ii in range(ngram_tokens.shape[0]): # batch of texts
                    seed = self.get_seed_rng(ngram_tokens[ii])
                    self.rng.manual_seed(seed)
                    vocab_permutation = torch.randperm(vocab_size, generator=self.rng)
                    greenlist = vocab_permutation[:int(self.gamma * vocab_size)] # gamma * n
                    redlist = vocab_permutation[int(self.gamma * vocab_size):]
                    bias = torch.ones(vocab_size).to(logits.device)
                    bias[greenlist] = self.test_mul/self.gamma # 1 + self.test_mul/(2*self.gamma)
                    bias[redlist] = (1-self.test_mul)/(1-self.gamma) #1 - self.test_mul/(2*(1-self.gamma))  
                    probs_sort[ii] *= bias[probs_idx[ii]]
                probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                probs_sum = torch.cumsum(probs_sort, dim=-1)
                mask = probs_sum - probs_sort > top_p
                probs_sort[mask] = 0.0
                probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1) # one hot of next token, ordered by original probs
            next_token = torch.gather(probs_idx, -1, next_token) # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token

    def logits_processor(self, logits, ngram_tokens):
        """Process logits to mask out words in greenlist."""
        bsz, vocab_size = logits.shape
        logits = logits.clone()
        for ii in range(ngram_tokens.shape[0]): # batch of texts
            seed = self.get_seed_rng(ngram_tokens[ii])
            self.rng.manual_seed(seed)
            vocab_permutation = torch.randperm(vocab_size, generator=self.rng)
            greenlist = vocab_permutation[:int(self.gamma * vocab_size)] # gamma * n
            bias = torch.zeros(vocab_size).to(logits.device)
            bias[greenlist] = self.delta
            logits[ii] += bias # add bias to greenlist words
        return logits
