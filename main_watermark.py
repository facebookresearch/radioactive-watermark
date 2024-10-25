# This code is adapted from the repository: https://github.com/facebookresearch/three_bricks
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
python main_watermark.py \
    --model_name <your_model_path> \
    --prompt_path "data/used_maryland_ngram2_seed0.jsonl" \
    --method none --method_detect maryland \
    --ngram 2 --scoring_method v2 \
    --nsamples 10000 --batch_size 16 \
    --output_dir output_closed_supervised_0p05/ \
    --filter_path "data/used_maryland_ngram2_seed0_filter.pkl" 
"""


import argparse
import os
import time
import json

import tqdm
import pandas as pd
import numpy as np

import random

import torch
from utils import HiddenPrints 
with HiddenPrints():
    from peft import PeftModel    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer

from wm import (WmGenerator, OpenaiGenerator, MarylandGenerator, StanfordGenerator,
                WmDetector,  OpenaiDetector, MarylandDetector, StanfordDetector, 
                MarylandDetectorZ, OpenaiDetectorZ)
import utils
import subprocess


def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)

    # model parameters
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--adapters_name', type=str)

    # prompts parameters
    parser.add_argument('--prompt_path', type=str, default="")
    parser.add_argument('--prompt_type', type=str, default="none", help="used to specify system prompts for e.g. alpaca")
    parser.add_argument('--prompt', type=str, nargs='+', default=None)

    # generation parameters
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_gen_len', type=int, default=256)

    # watermark parameters
    parser.add_argument('--method', type=str, default='none', help='Choose between: none (no watermarking), openai (Aaronson et al.), maryland (Kirchenbauer et al.)')
    parser.add_argument('--method_detect', type=str, default='same', help='Statistical test to detect watermark. Choose between: same (same as method), openai and Maryland. We use the tests from the three bricks paper.', choices = ['same', 'openai', 'maryland'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeding', type=str, default='hash', help='seeding method for rng key generation as introduced in https://github.com/jwkirchenbauer/lm-watermarking')
    parser.add_argument('--ngram', type=int, default=4, help='n-gram size for rng key generation')
    parser.add_argument('--gamma', type=float, default=0.25, help='gamma for maryland: proportion of greenlist tokens')
    parser.add_argument('--delta', type=float, default=2.0, help='delta for maryland: bias to add to greenlist tokens')
    parser.add_argument('--test_mul', type=float, default=0, help='delta for maryland: bias to add to greenlist tokens')
    parser.add_argument('--hash_key', type=int, default=35317, help='hash key for rng key generation')
    parser.add_argument('--scoring_method', type=str, default='v2', help='method for scoring. choose between: none (score every tokens), v1 (score token when wm context is unique), v2 (score token when {wm context + token} is unique')

    # expe parameters
    parser.add_argument('--nsamples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--split', type=int, default=None)
    parser.add_argument('--nsplits', type=int, default=None)
    parser.add_argument('--shuffle', type=int, default=1)
    parser.add_argument('--fake_seed', type=int, default=0)

    # eval by chunk parameters
    parser.add_argument('--filter_path', type=str, default=None, help="path to a pickle file that contains k-grams and their frequencies")
    parser.add_argument('--filter_number', type=int, default=-1, help="number of top tokens to use from the filter, according to their frequency")
    parser.add_argument('--keep_input_tokens', type=int, default=0, help='For each output, whether or not score watermark windows already present in the inputs. If the input is a watermarked question, then putting the value to 1 with lead to wrong p-values.')

    # distributed parameters
    parser.add_argument('--ngpus', type=int, default=None)

    return parser


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # build model
    # Llama-2-7b-chat-hf
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    args.ngpus = torch.cuda.device_count() if args.ngpus is None else args.ngpus
    # breakpoint()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        max_memory={i: '32000MB' for i in range(args.ngpus)}, # automatically handles the number of gpus
        offload_folder="offload",
    )

    adapters_name = args.adapters_name
    if adapters_name is not None:
        print(f"Loading adapter {adapters_name}")
        model = PeftModel.from_pretrained(model, adapters_name)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Using {args.ngpus}/{torch.cuda.device_count()} GPUs - {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated per GPU")

    # build watermark generator
    if args.method == "none":
        generator = WmGenerator(model, tokenizer)
    elif args.method == "openai":
        generator = OpenaiGenerator(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key)
    elif args.method == "maryland":
        generator = MarylandGenerator(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, delta=args.delta, test_mul = args.test_mul)
    else:
        raise NotImplementedError("method {} not implemented".format(args.method))

    # load prompts
    if args.prompt is not None:
        prompts = args.prompt
        prompts = [{"instruction": prompt} for prompt in prompts]
    else:
        prompts = utils.load_prompts(json_path=args.prompt_path, prompt_type=args.prompt_type, nsamples=args.nsamples)
    if args.shuffle:
        random.Random(args.fake_seed).shuffle(prompts)

    # do splits
    if args.split is not None:
        nprompts = len(prompts)
        left = nprompts * args.split // args.nsplits 
        right = nprompts * (args.split + 1) // args.nsplits if (args.split != args.nsplits - 1) else nprompts
        prompts = prompts[left:right]
        print(f"Creating prompts from {left} to {right}")
    
    # start experiment
    os.makedirs(args.output_dir, exist_ok=True)
    # generate
    all_times = []

    # generate answers to prompts. If the file exists, we don't do anything
    if not os.path.exists(os.path.join(args.output_dir, f"results.jsonl")):
        with open(os.path.join(args.output_dir, f"results.jsonl"), "a") as f:
            for ii in range(0, len(prompts), args.batch_size):
                # generate chunk
                time0 = time.time()
                chunk_size = min(args.batch_size, len(prompts) - ii)
                results = generator.generate(
                    prompts[ii:ii+chunk_size], 
                    max_gen_len=args.max_gen_len, 
                    temperature=args.temperature, 
                    top_p=args.top_p
                )
                time1 = time.time()
                # time chunk
                speed = chunk_size / (time1 - time0)
                eta = (len(prompts) - ii) / speed
                eta = time.strftime("%Hh%Mm%Ss", time.gmtime(eta)) 
                all_times.append(time1 - time0)
                print(f"Generated {ii:5d} - {ii+chunk_size:5d} - Speed {speed:.2f} prompts/s - ETA {eta}")
                # log
                for prompt, result in zip(prompts[ii:ii+chunk_size], results):
                    f.write(json.dumps({
                        "prompt": prompt, 
                        "result": result[len(prompt):],
                        "speed": speed,
                        "eta": eta}) + "\n")
                    f.flush()
            print(f"Average time per prompt: {np.sum(all_times) / len(prompts) :.2f}")

    if args.method_detect == 'same': #
        args.method_detect = args.method
    
    # build watermark detector
    if args.method_detect == "openai":
        detector = OpenaiDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key)
    elif args.method_detect == "maryland":
        detector = MarylandDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, delta=args.delta)

    # build sbert model
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    cossim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    results_orig = utils.load_results(json_path=args.prompt_path, nsamples=args.nsamples, result_key="output")
    if args.split is not None:
        results_orig = results_orig[left:right]

    # Watermark detection in each answer
    results = utils.load_results(json_path=os.path.join(args.output_dir, f"results.jsonl"), nsamples=args.nsamples, result_key="result")
    log_stats = []
    if not os.path.exists(os.path.join(args.output_dir, f"scores.jsonl")):
        with open(os.path.join(args.output_dir, 'scores.jsonl'), 'w') as f:
            # for loop over texts, could be batched
            for text, text_orig in  tqdm.contrib.tzip(results, results_orig):
                # compute watermark score
                scores_no_aggreg = detector.get_scores_by_t([text], scoring_method=args.scoring_method)
                scores = detector.aggregate_scores(scores_no_aggreg) # 1
                pvalues = detector.get_pvalues(scores_no_aggreg)
                num_tokens = [len(score_no_aggreg) for score_no_aggreg in scores_no_aggreg]
                # compute sbert score
                xs = sbert_model.encode([text, text_orig], convert_to_tensor=True)
                score_sbert = cossim(xs[0], xs[1]).item()
                for ii in range(len(scores)):
                    log_stat = {
                        'text_index': ii,
                        'num_token': num_tokens[ii],
                        'score': scores[ii],
                        'pvalue': pvalues[ii], 
                        'log10_pvalue': np.log10(pvalues[ii]),
                        'score_sbert': score_sbert,
                    }
                    log_stats.append(log_stat)
                    f.write('\n' + json.dumps({k: float(v) for k, v in log_stat.items()}))
                    f.flush()
            df = pd.DataFrame(log_stats)
            df['log10_pvalue'] = np.log10(df['pvalue'])
            print(f">>> Scores: \n{df.describe(percentiles=[])}")
            print(f"Saved scores to {os.path.join(args.output_dir, 'scores.csv')}")

    ##### THIS IS WHERE CHUNKING EVALUATION STARTS #####

    print("evaluating the chunks")

    result_name = "result_chunked" if args.scoring_method == "v2" else "result_chunked_v1"
    filter_path = args.filter_path if args.filter_path is not None else "" # path to a pickle file that contains k-grams and their frequencies
    base = ["python", "main_eval_chunked.py", "--json_path", os.path.join(args.output_dir, f"results.jsonl"), "--model_name", args.model_name, "--method", args.method_detect, "--ngram", str(args.ngram), "--output_dir", "same", "--keep_input_tokens", str(args.keep_input_tokens), "--hash_key", str(args.hash_key), "--scoring_method", str(args.scoring_method), "--seed", str(args.seed)]
    eval_chunk_methods = [ "--fname", result_name, "--filter", filter_path, "--filter_number", str(args.filter_number)]

    command = base + eval_chunk_methods
    print(command)
    subprocess.call(command)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
