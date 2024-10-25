# This code is adapted from the repository: https://github.com/facebookresearch/three_bricks
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# This file is not to be used by its own; it is used in main_watermarl.py to concatenate prompts and score a large amount of tokens with adequate deduplication

import argparse
from typing import Any, Dict, Tuple, List
import os
import random
import json

import numpy as np
import pandas as pd
import tqdm

import torch
from transformers import AutoTokenizer

from wm import OpenaiDetector, MarylandDetector, MarylandDetectorZ, OpenaiDetectorZ
import utils

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)

    # paths parameters
    # parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--text_key', type=str, default='result')
    parser.add_argument('--model_name', type=str, default='llama-2-7b-chat-hf')

    # watermark parameters
    parser.add_argument('--method', type=str, default='none', help='Choose between: none (no watermarking), openai (Aaronson et al.), maryland (Kirchenbauer et al.)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeding', type=str, default='hash', help='seeding method for rng key generation as introduced in https://github.com/jwkirchenbauer/lm-watermarking')
    parser.add_argument('--ngram', type=int, default=4, help='n-gram size for rng key generation')
    parser.add_argument('--gamma', type=float, default=0.25, help='gamma for maryland: proportion of greenlist tokens')
    parser.add_argument('--delta', type=float, default=2.0, help='delta for maryland: bias to add to greenlist tokens')
    parser.add_argument('--hash_key', type=int, default=35317, help='hash key for rng key generation')
    parser.add_argument('--scoring_method', type=str, default='v2', help='method for scoring. choose between: none (score every tokens), v1 (score token when wm context is unique), v2 (score token when {wm context + current token} is unique')

    # filter parameters
    parser.add_argument('--keep_input_tokens', type=int, default=1, help='For each output, whether or not to keep the k grams from the inputs.')
    parser.add_argument('--filter', type=str, default=None, help='Path to a pickle that has a set of k-grams and their frequencies.')
    parser.add_argument('--filter_number', type=int, default=None, help='number of tokens to use from the filter')
    
    # expe parameters
    parser.add_argument('--nsamples', type=int, default=None)
    parser.add_argument('--nsplits', type=int, default=1)
    parser.add_argument('--log_freq', type=int, default=200)
    parser.add_argument('--output_dir', type=str, default='same')
    parser.add_argument('--fname', type=str, default='scores_chunked')

    return parser


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    if args.output_dir == "same":
        args.output_dir = os.path.dirname(args.json_path)
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    texts = utils.load_results(json_path=args.json_path, nsamples=args.nsamples, result_key=args.text_key)
    if args.keep_input_tokens == 0:
        print(f"args.keep_input_tokens is null. deduplicating input tokens.")
        inputs = utils.load_results(json_path=args.json_path, nsamples=args.nsamples, result_key="prompt")
    else:
        print("deduplicating input tokens")
        inputs = texts

    if (args.filter is not None and  args.filter!=""):
        print(f"using an additional filter for the kgrams:{args.filter}")
        import pickle
        # Replace 'file.pkl' with your .pkl file path
        with open(args.filter, 'rb') as f:
            data = pickle.load(f)[0]
            data_filter = set(data) if (args.filter_number is None or args.filter_number==-1) else set(data[:args.filter_number])
            print("using a filter of size:", len(data_filter))
    else:
        print("not using an additional filter")
        data_filter = None
    print(f"Loaded {len(texts)} texts from {args.json_path}")

    # build watermark detector
    if args.method == "openai":
        detector = OpenaiDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key)
    elif args.method == "maryland":
        detector = MarylandDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, delta=args.delta)

    print(f"Evaluating watermarks on continuous texts")
    # compute wm scores and pvalues
    log_stats = []
    with open(os.path.join(args.output_dir, f'{args.fname}.jsonl'), 'w') as f:
        # split results in nsplits chunks
        splits = np.array_split(texts, args.nsplits)
        splits = [split.tolist() for split in splits]
        splits_inputs = np.array_split(inputs, args.nsplits)
        splits_inputs = [split.tolist() for split in splits_inputs]
        for ii, (split, input) in enumerate(tqdm.tqdm(zip(splits, splits_inputs))):
            # get scores/pvalues as if the split was one big chunk
            scores, mask_scored = detector.get_scores_by_t_chunked(split, scoring_method=args.scoring_method, return_aux=True, wm_inputs = input if (args.keep_input_tokens ==0) else None, data_filter = data_filter) # (many tokens, )
            mean_r = np.mean(scores)
            pvalues = detector.get_pvalues_by_t(scores)
            num_tok = 0
            for jj, pvalue in enumerate(pvalues):
                # compute the number of total tokens seen (not necessarily scored)
                while (mask_scored[num_tok] != 1) and (num_tok < len(mask_scored)):
                    num_tok += 1
                if jj % args.log_freq == 0:
                    log_stat = {
                        'split': ii,
                        "mean_r":mean_r,
                        'num_token': num_tok,
                        'num_scored': jj,
                        'score': np.mean(scores[:jj]),
                        'pvalue': pvalue,
                        'log10_pvalue': np.log10(pvalue) if pvalue > 0 else -0.43,
                    }
                    f.write('\n' + json.dumps(log_stat))
                    log_stats.append(log_stat)
                num_tok += 1
    df = pd.DataFrame(log_stats)
    # print(f">>> Wm scores: \n{df.groupby('split').describe(percentiles=[])}")
    print(f">>> Wm scores for split 0: \n{df[df['split']==0]}")
    print(f">>> Saved scores to {os.path.join(args.output_dir, 'scores.jsonl')}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)