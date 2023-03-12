import argparse
import flwr as fl
from flwr.common.typing import Scalar
import ray
import torch
import torchvision
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple, List
from myutils import fl_partition, initialise, train, test, logger 
from freeze import get_para_num, get_trainable_para_num, freeze_unfreeze_layers
import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)
import dill
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

#os.environ["TOKENIZERS_PARALLELISM"] = "false"

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
#RAY_ADDRESS="128.232.115.65:6379"

     
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser = argparse.ArgumentParser(description="Flower Simulation with bert")

    parser.add_argument("--num_client_cpus", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--do_noniid", type=bool, default=False)
    parser.add_argument("--do_freeze", type=bool, default=False)
    parser.add_argument("--num_freeze_layers", type=int, default=2)
    parser.add_argument("--alpha", type=int, default=10, help="increase to make dataset more non-iid")
    parser.add_argument("--beta", type=int, default=100, help="decrease to make dataset more non-iid")
    
    parser.add_argument(
        "--fed_dir_data",
        type=str,
        default=None,
        help="federated dataset dir.",
    )
    
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument("--cache_dir", type=str,
        default=None,
        help="Where do you want to store the pretrained models downloaded from huggingface.co"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    
    args = parser.parse_args()
    
    return args
            
# Flower client, adapted from Pytorch quickstart example
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir_data: str, args):
        self.cid = cid
        self.args = args
        self.args.output_dir = self.args.output_dir + str(int(cid)+1)
        self.fed_dir_data = fed_dir_data
              
        self.model = initialise(self.args)
        with open(self.fed_dir_data + args.task_name +'/train_dataloader.pkl','rb') as f:
            self.train_dataloader = dill.load(f)
        with open(self.fed_dir_data + args.task_name + '/eval_dataloader.pkl','rb') as f:
            self.eval_dataloader = dill.load(f)
        
        if args.do_noniid:
            self.train_range, self.eval_range = fl_partition(self.cid, len(self.train_dataloader), len(self.eval_dataloader), self.args.num_clients, alpha = args.alpha, beta = args.beta)
        
        else:
            train_step = len(self.train_dataloader) // args.num_clients
            self.train_range = [int(cid) * train_step, (int(cid)+1) * train_step]
            eval_step = len(self.eval_dataloader) // args.num_clients
            self.eval_range = [int(cid) * eval_step, (int(cid)+1) * eval_step]
            
            if int(cid) == (args.num_clients - 1):
                self.train_range = [int(cid) * train_step, len(self.train_dataloader)]
                self.eval_range = [int(cid) * eval_step, len(self.eval_dataloader)]
        
        if args.do_freeze:
            get_para_num(self.model)
            get_trainable_para_num(self.model)

            if args.num_freeze_layers == 1:
                freeze_unfreeze_layers(self.model, 0, unfreeze=False)
            else:
                freeze_unfreeze_layers(self.model, (0, args.num_freeze_layers-1), unfreeze=False)

            get_para_num(self.model)
            get_trainable_para_num(self.model)
                
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
       # num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        
        print("Training Started...")
        train(self.args, self.model.to(self.device), self.train_dataloader, self.device,self.train_range)
        print("Training Finished...")
        # Return local model and statistics
        return get_params(self.model), (self.train_range[1] - self.train_range[0]), {}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        #num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])     
    
        # Evaluate       
        loss, metric = test(self.args, self.model.to(self.device), self.eval_dataloader, self.device, self.eval_range)
        print("Evaluating finished...")
        # Return statistics
        return float(loss), (self.eval_range[1] - self.eval_range[0]), metric
    

def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 1,  # number of local epochs
        "batch_size": 8,
    }
    return config


def get_params(model: torch.nn.ModuleList) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model: torch.nn.ModuleList, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    print("###########setting#################")
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    print("###########finish################")

def aggregate_weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
    """Generic function to combine results from multiple clients
    following training or evaluation.

    Args:
        metrics (List[Tuple[int, dict]]): collected clients metrics

    Returns:
        dict: result dictionary containing the aggregate of the metrics passed.
    """
    average_dict: dict = defaultdict(list)
    total_examples: int = 0
    for num_examples, metrics_dict in metrics:
        for key, val in metrics_dict.items():
            if isinstance(val, numbers.Number):
                average_dict[key].append((num_examples, val))  # type:ignore
        total_examples += num_examples
    return {
        key: {
            "avg": float(
                sum([num_examples * metr for num_examples, metr in val])
                / float(total_examples)
            ),
            "all": val,
        }
        for key, val in average_dict.items()
    }
    
def get_evaluate_fn(
    testsets, args
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, float]]:
        """Use the HANS test set for evaluation."""

        # determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = initialise(args)
        
        metric = {}
        for testset in testsets:
            with open(testset + 'eval_dataloader.pkl','rb') as f:
                eval_dataloader = dill.load(f)

            set_params(model, parameters)
            model.to(device)
            loss, m = test(args, model, eval_dataloader, device, [0, len(eval_dataloader)])
            for k, v in m.items():
                metric[testset] =  v
        
        if args.output_dir is not None:
            out_dir = args.output_dir + "round" + str(server_round) + "/"
            folder = os.path.exists(out_dir)
            if not folder:
                os.makedirs(out_dir)
            all_results = {f"eval_{k}": v for k, v in metric.items()}
            with open(os.path.join(out_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f)

            model.save_pretrained(out_dir)
            
        # return statistics
        return loss, metric

    return evaluate


# Start simulation (a _default server_ will be created)

if __name__ == "__main__":

    # parse input arguments
    args = parse_args()
    
    pool_size = args.num_clients  # number of dataset partions (= number of total clients)
    client_resources = {
        "num_cpus": 7, # args.num_client_cpus,
        "num_gpus": 1
    }  
     
    testset = [args.fed_dir_data + "hans/", args.fed_dir_data + args.task_name + "/"]  
      
    # configure the strategy
    from fedavg import FedAvg
    strategy = FedAvg(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=pool_size,
        min_evaluate_clients=pool_size,
        min_available_clients=pool_size,  
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(testset, args),  # centralised evaluation of global model
    )

    def client_fn(cid: str):
        # create a single client instance
        return FlowerClient(cid, args.fed_dir_data, args)

    # (optional) specify Ray config
    ray_init_args = {"include_dashboard": False}

    from app import start_simulation
    # start simulation
    hist = start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
    )
    
    print(hist)