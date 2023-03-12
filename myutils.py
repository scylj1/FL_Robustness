import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import numpy as np

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import time
import transformers
import evaluate
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger(__name__)

def fl_partition(cid, train_len, eval_len, num_clients, alpha = 1, beta=100):
 
    def a_i(T, i):
        return int(T/num_clients + (2 * i - num_clients - 1) / 2 * np.log(alpha) * T / beta)

    cid = int(cid)
    t_1 = int(train_len / num_clients - (num_clients - 1) / 2 * np.log(alpha) * train_len / beta)
    e_1 = int(eval_len / num_clients - (num_clients - 1) / 2 * np.log(alpha) * eval_len / beta)
    train_range, eval_range = [0, max(t_1, 1)], [0, max(e_1, 1)]
    for n in range(num_clients-1):
        train_range.append(train_range[n+1] + a_i(train_len, n+2))
        eval_range.append(eval_range[n+1] + a_i(eval_len, n+2))
    train_range[-1] = train_len
    eval_range[-1] = eval_len

    print(train_range)
    print(eval_range)
    
    return train_range[cid:(cid+2)], eval_range[cid:(cid+2)]

def initialise(args):
               
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=3, finetuning_task=args.task_name)
    #tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )
          
    return model
  
def train(args, model, train_dataloader, device, train_range):
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
         
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
          
    # Train!
    total_batch_size = args.per_device_train_batch_size
    
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            
            model.load_state_dict(torch.load(args.resume_from_checkpoint))
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch
    start = time.perf_counter()
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            if step >= train_range[0] and step < train_range[1]:
                batch = {k: v.to(device) for k, v in batch.items()}
                # We need to skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        if step % args.gradient_accumulation_steps == 0:
                            progress_bar.update(1)
                            completed_steps += 1
                        continue
            
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps }"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        model.save_pretrained(output_dir)
                        #accelerator.save_state(output_dir)
                progress_bar.update(1)
                completed_steps += 1
                if step % 1000 == 0:
                    print(f"train_loss {loss}")
                if completed_steps >= args.max_train_steps:
                    break
            
        print(f"completed step {completed_steps}")
        print(f"train_loss {loss}")
        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"

    end = time.perf_counter()
    train_time = round(end-start)
    '''with open(os.path.join(args.output_dir, "train_results.json"), "a+") as f:
        json.dump({"train loss": float(loss) , "train time(s)": float(train_time)}, f)'''
    '''if args.output_dir is not None:
        model.save_pretrained(
            args.output_dir
        )'''
        
            
def test(args, model, eval_dataloader, device, eval_range):
    loss = 0   
    samples_seen = 0
    model.eval()
    losses = []
     # Get the metric function
    if args.task_name is not None:
        metric = evaluate.load("glue", args.task_name)
    else:
        metric = evaluate.load("accuracy")
    print("***** Running evaluating *****")
    print(f"  Num examples = {len(eval_dataloader)}")
    start = time.perf_counter()
    for step, batch in enumerate(eval_dataloader):
        if step >= eval_range[0] and step < eval_range[1]:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            loss += outputs.loss.item()
            if step % 1000 == 0:
                print("step: {}, eval loss: {}".format(step, loss))
            
            predictions = outputs.logits.argmax(dim=-1) 
            
            metric.add_batch(
                    predictions=predictions,
                    references=batch["labels"]
                )
      
    end = time.perf_counter()
    eval_time = round(end-start)

    eval_loss = loss / (eval_range[1] - eval_range[0])
    eval_metric = metric.compute()
    print(f"metric: {eval_metric}")
    
    '''if args.output_dir is not None:
        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f)'''
            
    return float(eval_loss), eval_metric

if __name__ == "__main__":
    print(1)
    fl_partition("0", 10000, 100, 5, alpha = 100000)

