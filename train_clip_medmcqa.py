"""
    Fine-tune CLIP-based text models on MedMCQA dataset
    posed as a multiple choice problem.
"""
import os 
import argparse
from pathlib import Path 
import shutil
import json

from data_collator_multiple_choice import DataCollatorForMultipleChoice
from modeling.modeling_clip_text_finetune import CLIPTextFinetunedModule, PMC_CLIPTextFinetunedModule, train
import wandb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # ***** Dataset *****
    parser.add_argument("--dataset", default="openlifescienceai/medmcqa", type=str)
    # ***** Model *****
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--base_model", type=str, choices=["clip", "pmc-clip"])
    # ***** PMC-CLIP *****
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--config", type=str, default="./modeling/pmc_clip/model_configs/RN50_fusion4.json")
    # ***** Trainer *****
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--report_to", type=str, nargs="+", default=["wandb"])
    # ***** Wandb ***** 
    parser.add_argument("--project", type=str, default="cs231n-medmcqa")
    args = parser.parse_args()

    # 1. Load and preprocess dataset
    dataset = load_dataset(args.dataset)
    print(dataset)
    
    if args.base_model == "clip":
        tokenizer = AutoTokenizer.from_pretrained(args.clip_model_name, use_fast=True)
        max_token_length=77
    elif args.base_model == "pmc-clip":
        model_config = json.load(open(args.config))
        tokenizer_name = model_config['text_cfg']['bert_model_name']
        assert tokenizer_name == 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', \
            "Please check [CLS]'s token id"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        max_token_length=512

    options = ["opa", "opb", "opc", "opd"]
    def filter_too_long(example):
        for option in options:
            if len(tokenizer.encode(f"{example['question']} {example[option]}", truncation=False)) > max_token_length:
                return False
            return True
    # Filter the dataset
    dataset["train"] = dataset["train"].filter(filter_too_long)
    dataset["validation"] = dataset["validation"].filter(filter_too_long)
    dataset["test"] = dataset["test"].filter(filter_too_long)
    print(dataset)

    """
    Inspired by https://huggingface.co/docs/transformers/en/tasks/multiple_choice
    """
    def preprocess_function(examples):
        # Repeat each first sentence four times to go with the four possibilities of second sentences.
        # first_sentences = [[context] * 4 for context in examples["question"]]
        first_sentences = [[f"{question} {examples[option][i]}" for option in options] for i, question in enumerate(examples["question"])]

        # Flatten everything
        first_sentences = sum(first_sentences, [])
        # second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = tokenizer(first_sentences, truncation=True, max_length=max_token_length, padding="max_length")
        # Un-flatten
        return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    dataset = dataset.map(preprocess_function, batched=True)
    dataset = dataset.rename_column("cop", "label")

    data_collator = DataCollatorForMultipleChoice(tokenizer)

    # Set the format to PyTorch tensors
    dataset.set_format(type='torch', columns=['label', 'input_ids', 'attention_mask'])

    # Create a DataLoader
    loader_train = DataLoader(dataset['train'], batch_size=args.batch_size, collate_fn=data_collator)
    loader_val = DataLoader(dataset['validation'], batch_size=args.batch_size, collate_fn=data_collator)

    # 2. Init and train model
    learning_rates = [5e-6]
    weight_decay = [1e-1]

    for lr in learning_rates:
        for wd in weight_decay:
            print(f"lr = {lr}, wd = {wd}")
            epoch = args.num_train_epochs
            # 🐝 initialise a wandb run
            wandb.init(
                project=args.project,
                name=f"lr = {lr}, wd = {wd}",
                config={
                    "epochs": epoch,
                    "batch_size": args.batch_size,
                    "lr": lr,
                    "wd": wd
                })
            if args.base_model == "clip":
                model = CLIPTextFinetunedModule(args.clip_model_name).to(device)
            elif args.base_model == "pmc-clip":
                model = PMC_CLIPTextFinetunedModule(args.checkpoint, 
                                    args.config, 
                                    device=device).to(device) 
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            train(args, model, loader_train, loader_val, optimizer, device, wandb, epochs=epoch, print_every=1000, lr=lr, wd=wd)
            wandb.finish()
