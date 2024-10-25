# This code is adapted from the repository: https://github.com/facebookresearch/three_bricks
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, List
import json
import io
import os
import sys
import socket
import subprocess

import torch
import torch.distributed as dist

# Experiment helpers

def bool_inst(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected in args')

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# Prompts and results loading

def format_prompts(prompts: List[Dict], prompt_type: str) -> List[str]:
    prompt_type = prompt_type.lower()
    if prompt_type=='alpaca':
        PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
            ),
        }
    elif prompt_type=='self_instruct':
        PROMPT_DICT = {
            "prompt_input": (
                "[INST] Write a response that appropriately completes the request given the context.\n## Instruction:\n{instruction}\n## Context:\n{input}\n[/INST]\n## Response:\n"
            ),
            "prompt_no_input": (
                "[INST] Write a response that appropriately completes the request.\n## Instruction:\n{instruction}\n[/INST]\n## Response:\n"
            )
        }
    elif prompt_type=='oasst':
        PROMPT_DICT = {
            "prompt_input": (
                "The following is a conversation between a human and an AI assistant named Llama. The assistant gives helpful, detailed, and polite answers to the user's questions. The human is indicated by 'Human:' and the assistant is indicated by 'Llama':\n\nHuman: {instruction}. The input is {input}.\nLLama:"
            ),
            "prompt_no_input": (
                "The following is a conversation between a human and an AI assistant named Llama. The assistant gives helpful, detailed, and polite answers to the user's questions. The human is indicated by 'Human:' and the assistant is indicated by 'Llama':\n\nHuman: {instruction}\nLLama:"
            ),
        }
    elif prompt_type=='guanaco':
        PROMPT_DICT = {
            "prompt_input": (
                "A chat between a curious human and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions.\n\n### Human: {instruction}\n\n### Input:\n{input}\n\n### Assistant:"
            ),
            "prompt_no_input": (
                "A chat between a curious human and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions.\n\n### Human: {instruction}\n\n### Assistant:"
            )
        }
    elif prompt_type=='llama-2':
        PROMPT_DICT = {
            "prompt_input": (
                "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n{instruction} The input for the previous task is '{input}'.[/INST]\n"
            ),
            "prompt_no_input": (
                "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n{instruction}[/INST]\n"
            )
        }
    elif prompt_type=='none':
        PROMPT_DICT = {
            "prompt_input": (
                "<s> {input}"
            ),
            "prompt_no_input": (
                "<s> {input} "
            )
        }
    else:
        raise ValueError(f"Unknown prompt type {prompt_type}")
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    prompts = [
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        for example in prompts
    ]
    return prompts

def load_prompts(json_path: str, prompt_type: str, nsamples: int=None) -> List[str]:
    with open(json_path, "r") as f:
        if json_path.endswith('.json'):
            prompts = json.loads(f.read())
        else:
            prompts = [json.loads(line) for line in f.readlines() if line.strip()] # load jsonl
    new_prompts = prompts
    new_prompts = new_prompts[:nsamples]
    print(f"Filtered {len(new_prompts)} prompts from {len(prompts)}")
    new_prompts = format_prompts(new_prompts, prompt_type)
    return new_prompts

def load_results(json_path: str, nsamples: int=None, result_key: str='result') -> List[str]:
    with open(json_path, "r") as f:
        if json_path.endswith('.json'):
            prompts = json.loads(f.read())
        else:
            prompts = [json.loads(line) for line in f.readlines() if line.strip()] # load jsonl
    new_prompts = [o[result_key] for o in prompts]
    new_prompts = new_prompts[:nsamples]
    return new_prompts

# alpaca utils

def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

# distrib utils

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print
    
    # remove logging when not master
    if not is_master:
        import logging
        import warnings
        logging.disable()
        warnings.filterwarnings("ignore")


def init_distributed_mode(master_port):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - local_rank: rank of the current GPU / node
        - world_size: number of GPUs / nodes
        - global_rank: unique id for each process (from 0 to world_size * num_gpus_per_node - 1)
        - n_nodes: number of nodes
        - node_id: id of the node
        - is_master: True if the current process is the main one, defined as the node with rank 0 and local rank 0
        - multi_node: True if we are in a multi-node setup
        - multi_gpu: True if we are in a multi-GPU setup
        - distributed: True for multi-GPU training
    """
    is_slurm_job = 'SLURM_JOB_ID' in os.environ
    print("SLURM job: %s" % str(is_slurm_job))

    # SLURM job
    if is_slurm_job:

        # assert training_args.local_rank == -1   # on the cluster, this is handled by SLURM

        SLURM_VARIABLES = [
            'SLURM_JOB_ID',
            'SLURM_JOB_NODELIST', 'SLURM_JOB_NUM_NODES', 'SLURM_NTASKS', 'SLURM_TASKS_PER_NODE',
            'SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU',
            'SLURM_NODEID', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_TASK_PID'
        ]

        PREFIX = "%i - " % int(os.environ['SLURM_PROCID'])
        for name in SLURM_VARIABLES:
            value = os.environ.get(name, None)
            print(PREFIX + "%s: %s" % (name, str(value)))

        # # job ID
        job_id = os.environ['SLURM_JOB_ID']

        # number of nodes / node ID
        n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        node_id = int(os.environ['SLURM_NODEID'])

        # local rank on the current node / global rank
        local_rank = int(os.environ['SLURM_LOCALID'])
        global_rank = int(os.environ['SLURM_PROCID'])

        # number of processes / GPUs per node
        world_size = int(os.environ['SLURM_NTASKS'])
        n_gpu_per_node = world_size // n_nodes

        # define master address and master port
        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        master_addr = hostnames.split()[0].decode('utf-8')
        if master_port==-1:
           master_port = 19500
        assert 10001 <= master_port <= 20000 or world_size == 1
        print(PREFIX + "Master address: %s" % master_addr)
        print(PREFIX + "Master port   : %i" % master_port)

        # set environment variables for 'env://'
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(global_rank)

        torch.distributed.init_process_group(
            init_method='env://',
            backend='nccl',
        )

        # set GPU device
        torch.cuda.set_device(local_rank)
        dist.barrier()

        # set new arg for local_rank
        sys.argv += ['--local_rank', str(local_rank)]

        is_master = (local_rank == 0) and (node_id == 0)
        print(PREFIX + "Is master      : %s" % str(is_master))
        setup_for_distributed(is_master)

    # multi-GPU job (local or multi-node) - jobs started with torchrun/ distributed.launch
    else:
        assert master_port == -1

        n_gpu_per_node = 2
        n_nodes = 1
        node_id = 0
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        is_master = (local_rank == 0) and (node_id == 0)
        setup_for_distributed(is_master)

    return local_rank, world_size

    # # sanity checks
    # assert params.n_nodes >= 1
    # assert 0 <= params.node_id < params.n_nodes
    # assert 0 <= params.local_rank <= params.global_rank < params.world_size
    # assert params.world_size == params.n_nodes * params.n_gpu_per_node

    # # define whether this is the master process / if we are in distributed mode
    # params.is_master = params.node_id == 0 and params.local_rank == 0
    # params.multi_node = params.n_nodes > 1
    # params.distributed = params.world_size > 1
    # params.distributed = True

    # # summary
    # PREFIX = "%i - " % params.global_rank
    # print(PREFIX + "Number of nodes: %i" % params.n_nodes)
    # print(PREFIX + "Node ID        : %i" % params.node_id)
    # print(PREFIX + "Local rank     : %i" % params.local_rank)
    # print(PREFIX + "Global rank    : %i" % params.global_rank)
    # print(PREFIX + "World size     : %i" % params.world_size)
    # print(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
    # print(PREFIX + "Master         : %s" % str(params.is_master))
    # print(PREFIX + "Multi-node     : %s" % str(params.multi_node))
    # print(PREFIX + "Multi-GPU      : %s" % str(params.distributed))
    # print(PREFIX + "Hostname       : %s" % socket.gethostname())

    # # initialize multi-GPU
    # # if params.distributed:

    #     # http://pytorch.apachecn.org/en/0.3.0/distributed.html#environment-variable-initialization
    #     # 'env://' will read these environment variables:
    #     # MASTER_PORT - required; has to be a free port on machine with rank 0
    #     # MASTER_ADDR - required (except for rank 0); address of rank 0 node
    #     # WORLD_SIZE - required; can be set either here, or in a call to init function
    #     # RANK - required; can be set either here, or in a call to init function

    #     print("Initializing PyTorch distributed ...")
    #     torch.distributed.init_process_group(
    #         init_method='env://',
    #         backend='nccl',
    #     )

    #     # set GPU device
    #     torch.cuda.set_device(params.local_rank)
    #     dist.barrier()

        # # initialize model parallel
        # initialize_model_parallel(params.world_size)
