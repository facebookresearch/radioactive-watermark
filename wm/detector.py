# This code is adapted from the repository: https://github.com/facebookresearch/three_bricks
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import sys
import numpy as np
from scipy import special

import torch
from transformers import LlamaTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WmDetector():
    def __init__(self, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317
        ):
        # model config
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
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
    
    def get_seed_rng(self, input_ids: List[int]) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.salt_key + i) % (2 ** 64 - 1)
        elif self.seeding == 'additive':
            seed = self.salt_key * torch.sum(input_ids)
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.salt_key * input_ids[0]
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.salt_key * input_ids)
            seed = torch.min(seed)
        return seed

    def aggregate_scores(self, scores: List[List[np.array]], aggregation: str = 'mean') -> List[float]:
        """Aggregate scores along a text."""
        scores = np.asarray(scores)
        if aggregation == 'sum':
            return [ss.sum(axis=0) if ss.shape[0]!=0 else -1.0 for ss in scores]
        elif aggregation == 'mean':
            return [ss.mean(axis=0) if ss.shape[0]!=0 else -1.0 for ss in scores]
        elif aggregation == 'max':
            return [ss.max(axis=0) if ss.shape[0]!=0 else -1.0 for ss in scores]
        else:
            raise ValueError(f'Aggregation {aggregation} not supported.')

    def get_scores_by_t(
        self, 
        texts: List[str], 
        scoring_method: str="none",
        ntoks_max: int = None, 
        return_aux: bool = False
    ) -> List[List[float]]:
        """
        Get score increment for each token in list of texts.
        Args:
            texts: list of texts
            scoring_method: 
                'none': score all ngrams
                'v1': only score tokens for which wm window is unique
                'v2': only score unique {wm window+tok} is unique
            ntoks_max: maximum number of tokens
            return_aux: if True, return masks of scored tokens information
        Output:
            score_lists: list of [score increments for every token] for each text
            masks_lists (optional): list of [1 if token is scored, 0 otherwise] for each text
        """
        bsz = len(texts)
        tokens_id = [self.tokenizer.encode(x, add_special_tokens=False) for x in texts]
        if ntoks_max is not None:
            tokens_id = [x[:ntoks_max] for x in tokens_id]
        score_lists = []
        masks_lists = []
        for ii in range(bsz):
            total_len = len(tokens_id[ii])
            start_pos = self.ngram +1
            rts = [] # list of score increments for each token
            mask_scored = [] # stores 1 for token if scored, 0 otherwise
            seen_ntuples = set()
            for cur_pos in range(start_pos, total_len):
                ngram_tokens = tokens_id[ii][cur_pos-self.ngram:cur_pos] # h
                mask_scored += [0] # 0 by default
                if scoring_method == 'v1': # only score tokens for which wm window is unique
                    tup_for_unique = tuple(ngram_tokens)
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                elif scoring_method == 'v2': # only score unique {wm window+tok} is unique
                    tup_for_unique = tuple(ngram_tokens + tokens_id[ii][cur_pos:cur_pos+1])
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                mask_scored[-1] = 1 # 1 since we are scoring this token
                rt = self.score_tok(ngram_tokens, tokens_id[ii][cur_pos])
                rts.append(rt)
            score_lists.append(rts)
            masks_lists.append(mask_scored)
        if return_aux:
            return score_lists, masks_lists
        return score_lists

    def get_scores_by_t_chunked(
        self, 
        texts: List[str], 
        wm_inputs: List[str] = None,
        data_filter: set=None,
        scoring_method: str="none",
        ntoks_max: int = None, 
        return_aux: bool = False
    ) -> List[float]:
        """
        Get score increment for each token in list of texts, as if the texts were one big chunk.
        Args:
            texts: list of texts
            scoring_method: 
                'none': score all ngrams
                'v1': only score tokens for which wm window is unique
                'v2': only score unique {wm window+tok} is unique
            ntoks_max: maximum number of tokens
            return_aux: if True, return masks of scored tokens information
        Output:
            rts: list of score increments for every token
            mask_scored (optional): stores 1 for token if scored, 0 otherwise
        """
        bsz = len(texts)
        tokens_id = [self.tokenizer.encode(x, add_special_tokens=False) for x in texts]

        if wm_inputs is not None:
            assert len(wm_inputs)==bsz
            tokens_id_inputs = [self.tokenizer.encode(x, add_special_tokens=False) for x in wm_inputs]
            res = []
            for ii in range(bsz):
                seen_ntuples_inputs = set()
                total_len = len(tokens_id_inputs[ii])
                start_pos = self.ngram
                for cur_pos in range(start_pos, total_len):
                    ngram_tokens = tokens_id_inputs[ii][cur_pos-self.ngram:cur_pos]
                    seen_ntuples_inputs.add(tuple(ngram_tokens))
                res.append(seen_ntuples_inputs)
            assert len(res) == bsz
        else:
            res = [set() for i in range(bsz)]



        if ntoks_max is not None:
            tokens_id = [x[:ntoks_max] for x in tokens_id]
        rts = [] # list of score increments for each token
        mask_scored = [] # stores 1 for token if scored, 0 otherwise
        seen_ntuples = set()
        print("inside get_scores_by_t_chunked")
        for ii in range(bsz):
            total_len = len(tokens_id[ii])
            start_pos = self.ngram + 1
            for cur_pos in range(start_pos, total_len):
                ngram_tokens = tokens_id[ii][cur_pos-self.ngram:cur_pos] # h
                mask_scored += [0] # 0 by default
                if scoring_method == 'v1': # only score tokens for which wm window is unique
                    tup_for_unique = tuple(ngram_tokens)
                    if (tup_for_unique in seen_ntuples) or (tup_for_unique in res[ii]) or (data_filter is not None and tup_for_unique not in data_filter):
                        continue
                    seen_ntuples.add(tup_for_unique)
                elif scoring_method == 'v2': # only score unique {wm window+tok} is unique
                    tup_for_unique = tuple(ngram_tokens + tokens_id[ii][cur_pos:cur_pos+1])
                    if (tup_for_unique in seen_ntuples) or (tuple(ngram_tokens) in res[ii]) or (data_filter is not None and tuple(ngram_tokens) not in data_filter):
                        continue
                    seen_ntuples.add(tup_for_unique)
                mask_scored[-1] = 1 # 1 since we are scoring this token
                rt = self.score_tok(ngram_tokens, tokens_id[ii][cur_pos])
                rts.append(rt)
        if return_aux:
            return rts, mask_scored
        return rts

    def get_pvalues(
            self, 
            scores: List[List[float]], 
            eps: float=1e-200
        ) -> np.array:
        """
        Get p-value for each text.
        Args:
            score_lists: list of [list of score increments for each token] for each text
        Output:
            pvalues: np array of p-values for each text and payload
        """
        pvalues = []
        scores = np.asarray(scores) # bsz ntoks 
        for ss in scores:
            ntoks = ss.shape[0]
            final_score = ss.sum(axis=0) if ntoks!=0 else -1
            pval = self.get_pvalue(final_score, ntoks, eps=eps)
            pvalues.append(pval)
        return np.asarray(pvalues) # bsz

    def get_pvalues_by_t(
            self, 
            scores: List[float],
            eps: float=1e-200
        ) -> List[float]:
        """Get p-value for each text, at each scored token."""
        pvalues = []
        cum_score = 0
        cum_toks = 0
        for ss in scores:
            cum_score += ss
            cum_toks += 1
            pvalue = self.get_pvalue(cum_score, cum_toks, eps)
            pvalues.append(pvalue)
        return pvalues
    
    def score_tok(self, ngram_tokens: List[int], token_id: int):
        """ for each token in the text, compute the score increment """
        raise NotImplementedError
    
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ compute the p-value for a couple of score and number of tokens """
        raise NotImplementedError


class MarylandDetector(WmDetector):

    def __init__(self, 
            tokenizer: LlamaTokenizer,
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            gamma: float = 0.5, 
            delta: float = 1.0, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.gamma = gamma
        self.delta = delta
    
    def score_tok(self, ngram_tokens: List[int], token_id: int):
        """ rt = 1 if token_id in greenlist else 0 """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)] # gamma * n are in greenlist
        rt = 1 if token_id in greenlist else 0
        return rt
                
    def get_pvalue(self, score: int, ntoks: int, eps: float):
        """ from cdf of a binomial distribution """
        pvalue = special.betainc(score, 1 + ntoks - score, self.gamma)
        return max(pvalue, eps)

class MarylandDetectorZ(WmDetector):

    def __init__(self, 
            tokenizer: LlamaTokenizer,
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            gamma: float = 0.5, 
            delta: float = 1.0, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.gamma = gamma
        self.delta = delta
    
    def score_tok(self, ngram_tokens, token_id):
        """ rt = 1 if token_id in greenlist else 0 """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)] # gamma * n
        rt = 1.0 if token_id in greenlist else 0.0
        return rt
                
    def get_pvalue(self, score: int, ntoks: int, eps: float):
        """ from cdf of a binomial distribution """
        zscore = (score - self.gamma * ntoks) / np.sqrt(self.gamma * (1 - self.gamma) * ntoks)
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return max(pvalue, eps)
    

class OpenaiDetector(WmDetector):

    def __init__(self, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
    
    def score_tok(self, ngram_tokens, token_id):
        """ rt = -log(1 - rt[xt]]) """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        rs = torch.rand(self.vocab_size, generator=self.rng) # n
        rt = -(1 - rs).log()[token_id]
        return rt.item()
 
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ from cdf of a gamma distribution """
        pvalue = special.gammaincc(ntoks, score)
        return max(pvalue, eps)


class OpenaiDetectorZ(WmDetector):

    def __init__(self, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
    
    def score_tok(self, ngram_tokens, token_id):
        """ rt = -log(1 - rt[xt]]) """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        rs = torch.rand(self.vocab_size, generator=self.rng) # n
        rt = -(1 - rs).log()[token_id]
        return rt.item()
 
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ z value """
        mu0 = 1
        sigma0 = np.pi / np.sqrt(6)
        zscore = (score/ntoks - mu0) / (sigma0 / np.sqrt(ntoks))
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return max(pvalue, eps)


# additional imports
# import pyximport
# pyximport.install(reload_support=True, language_level=sys.version_info[0],
#                 setup_args={'include_dirs':np.get_include()})
# from .levenshtein import levenshtein

from numba import njit

@njit
def levenshtein(toks, xis, gamma=0.0):
    """
    Args:
        toks: list of token ids (ntoks)
        xis: list of watermark vectors (ngram, vocab_size) - ngram is the size of block
        gamma: cost of insertion/deletion
    """
    ngram, vocab_size = xis.shape
    ntoks = len(toks)
    A = np.zeros((ntoks+1, ntoks+1), dtype=np.float32)
    for ii in range(0, ntoks+1):
        for jj in range(0, ntoks+1):
            if ii == 0:
                A[ii, jj] = jj * gamma
            elif jj == 0:
                A[ii, jj] = ii * gamma
            else:
                cost = np.log(1 - xis[jj-1, toks[ii-1]])
                A[ii, jj] = A[ii-1, jj] + gamma
                if A[ii, jj-1] + gamma < A[ii, jj]:
                    A[ii, jj] = A[ii, jj-1] + gamma
                if A[ii-1, jj-1] + cost < A[ii, jj]:
                    A[ii, jj] = A[ii-1, jj-1] + cost

    return A[ntoks, ntoks]

class StanfordDetector(WmDetector):

    def __init__(self,
            tokenizer: LlamaTokenizer,
            ngram: int,
            seed: int,
            stanford_nruns: int,
            **kwargs):
        super().__init__(tokenizer, ngram, seed, **kwargs)
        self.nruns = stanford_nruns  # for p-value estimation
        self.rng.manual_seed(self.seed)
        self.xi = torch.rand((self.ngram, self.vocab_size), generator=self.rng)  # ngram, vocab_size

    def get_score(self, tokens, xi, max_nshifts=1):
        ntoks = len(tokens)
        nn = min(self.ngram, ntoks) # block size
        if ntoks >= nn:
            nshifts = min(max_nshifts, ntoks-nn+1)
        gamma = 0.0  # not used here
        A = np.empty((nshifts, self.ngram))
        for ii in range(nshifts):
            for jj in range(self.ngram):
                lev_jj = levenshtein(
                    np.array(tokens[ii:ii+nn]),
                    np.array(xi[(jj + np.arange(nn)) % self.ngram]), 
                    gamma
                )
                A[ii, jj] = lev_jj
        # print(A)
        return np.min(A)

    def get_scores(self, texts: List[str]):

        bsz = len(texts)
        tokens_id = [self.tokenizer.encode(x, add_special_tokens=False) for x in texts]

        return_dict = {'scores': [], 'num_tokens': [], 'pvalues': []}
        for ii in range(bsz):
            tokens = tokens_id[ii]
            score = self.get_score(tokens, xi=self.xi)
            if self.nruns == 0:
                pvalue = 0.5
            else:
                pvalue = 0
                for _ in range(self.nruns):
                    xi_alternative = np.array(torch.rand((self.ngram, self.vocab_size), generator=self.rng))
                    null_result = self.get_score(tokens, xi=xi_alternative)
                    pvalue += null_result <= score
                pvalue = (pvalue+1.0)/(self.nruns+1.0)

            return_dict['scores'].append(score)
            return_dict['num_tokens'].append(len(tokens))
            return_dict['pvalues'].append(pvalue)

        return return_dict