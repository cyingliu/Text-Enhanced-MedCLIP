"""
    Inference CLIP-based models on VQA-RAD dataset
    calculate test overall, closed/open, yesno/non-yesno accuracy
    outputs predictions of all test examples into a .csv file
"""
import os 
import argparse
from pathlib import Path 
import shutil
import pickle

from modeling.transformer_decoder import VqaTransformerDecoder
from modeling.transformer_encoder import VqaTransformerEncoder
import wandb
from tqdm import tqdm 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, CLIPImageProcessor
from transformers import TrainingArguments, Trainer
from transformers.data import DataCollatorWithPadding


from modeling import CLIPwithLinearFusion, PMC_CLIPforVQA
from transform import train_transform, test_transform
from data_collator import torch_images_and_label_data_collator

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    
    # ***** Dataset *****
    parser.add_argument("--dataset", default="cs231n-Medco/vqa-rad", type=str)
    parser.add_argument("--task", default="yesno", choices=["yesno", "all"])
    parser.add_argument("--seed", type=int, default=123, help="Seed for train/val split.")
    # ***** Model ***** 
    parser.add_argument("--base_model", type=str, choices=["clip", "pmc-clip"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pytorch_model.bin")
    parser.add_argument("--text_model_path", type=str)
    # ***** CLIP with Linear Fusion *****
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    # ***** PMC-CLIP for VQA *****
    parser.add_argument("--pmc_checkpoint", type=str)
    parser.add_argument("--config", type=str, default="./modeling/pmc_clip/model_configs/RN50_fusion4.json")
    parser.add_argument("--pool_type", type=str, default="average", choices=["average", "cls"])
    # **** CLIP with Transformer Fusion ****
    parser.add_argument("--transformer_fusion", type=str, choices=["decoder", "encoder", "none"], default="none")
    parser.add_argument("--transformer_num_layers", type=int, default=2)
    # ***** Other *****
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output", type=str, default="result.csv", help="Output path for predictions")
    parser.add_argument("--splits", type=str, nargs="+", help="Specify test-yesno, test-nonyesno, test-close, test-open "
                                                              "to evalaute on subset of questions. Only take effect when task is all.")
    args = parser.parse_args()

    if args.splits is None:
        args.splits = []
    for split in args.splits:
        assert split in ("test-yesno", "test-nonyesno", "test-closed", "test-open")
    assert args.dataset == "cs231n-Medco/vqa-rad" # the label matching is based on this dataset
    # 1. Preprocess dataset
    dataset = load_dataset(args.dataset)

    if args.task == "yesno":
        dataset = dataset.filter(lambda example: example["answer"].lower() in ("yes", "no"))

        def preprocess_labels_yesno(batch):
            batch["labels"] = [1 if answer.lower() == "yes" else 0 for answer in batch["answer"]]
            return batch 
        dataset = dataset.map(preprocess_labels_yesno, batched=True)
    
    elif args.task == "all":
        assert args.dataset == "cs231n-Medco/vqa-rad" # the dataset should be preprocessed from  https://github.com/aioz-ai/MICCAI21_MMQ?tab=readme-ov-file#vqa-rad-dataset-for-vqa-task
        dataset = dataset.sort("qid")
        
        train_target = pickle.load(open("./data/cache/train_target.pkl", "rb"))
        qid2label = {x['qid']: x['labels'][0] for x in train_target}
        assert dataset['train']['qid'] == list(qid2label.keys())
        dataset['train'] = dataset['train'].add_column("labels", list(qid2label.values()))

        test_target = pickle.load(open("./data/cache/test_target.pkl", "rb"))
        qid2label = {x['qid']: x['labels'][0] if len(x['labels']) > 0 else None for x in test_target}
        assert dataset['test']['qid'] == list(qid2label.keys())
        dataset['test'] = dataset['test'].add_column("labels", list(qid2label.values()))
        dataset['test'] = dataset['test'].filter(lambda example: example['labels'] is not None)

    train_val_dataset = dataset["train"].train_test_split(test_size=0.125, seed=args.seed)
    train_val_test_dataset = DatasetDict({'train': train_val_dataset['train'],
                                        'val': train_val_dataset['test'],
                                        'test': dataset['test']})

    image_processor = CLIPImageProcessor.from_pretrained(args.clip_model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.clip_model_name)
    def preprocess_clip(batch):
        image_features = image_processor(batch["image"])
        text_features = tokenizer(batch["question"])
        batch = {**image_features, **text_features, **batch}
        return batch
    
    def preprocess_pmcclip(batch):
        batch['bert_input'] = [question for question in batch['question']] # pmc-clip tokenize text inputs in the forward call
        batch['bert_label'] = [question for question in batch['question']]
        return batch
    
    if args.base_model == "clip":
        preprocess_fn = preprocess_clip
        select_columns = ['qid', 'labels', 'pixel_values', 'input_ids', 'attention_mask'] # add qid for mapping
    elif args.base_model == "pmc-clip":
        preprocess_fn = preprocess_pmcclip
        select_columns = ['qid', 'image', 'labels', 'bert_input', 'bert_label'] # add qid for mapping

    processed_dataset = train_val_test_dataset.map(preprocess_fn, batched=True)
    if args.task == 'all':
        if 'test-yesno' in args.splits:
            processed_dataset['test-yesno'] = processed_dataset['test'].filter(lambda example: example['answer'].strip().lower() == 'yes' or example['answer'].strip().lower() == 'no')
        if 'test-nonyesno' in args.splits:
            processed_dataset['test-nonyesno'] = processed_dataset['test'].filter(lambda example: example['answer'].strip().lower() != 'yes' and example['answer'].strip().lower() != 'no')
        if 'test-closed' in args.splits:
            processed_dataset['test-closed'] = processed_dataset['test'].filter(lambda example: example['answer_type'].strip().lower() == 'closed')
        if 'test-open' in args.splits:
            processed_dataset['test-open'] = processed_dataset['test'].filter(lambda example: example['answer_type'].strip().lower() == 'open')
    
    if args.base_model == "pmc-clip":
        processed_dataset['train'].set_transform(train_transform)
        processed_dataset['val'].set_transform(test_transform)
        processed_dataset['test'].set_transform(test_transform)
        if args.task == 'all':
            for split in processed_dataset:
                if split.startswith("test-"):
                    processed_dataset[split].set_transform(test_transform)
    
    processed_dataset = processed_dataset.select_columns(select_columns)

    # 2. Init model
    num_labels = 2 if args.task == "yesno" else 458 # num of labels in trainval_label2ans.pkl
    if args.base_model == "clip":
        if args.transformer_fusion == "none":
            model = CLIPwithLinearFusion(args.clip_model_name, 
                                    text_model_path=args.text_model_path,
                                    num_labels=num_labels,
                                    device=device).to(device)
        elif args.transformer_fusion == "decoder":
            model = VqaTransformerDecoder(args.clip_model_name, 
                                 text_model_path=args.text_model_path,
                                 num_labels=num_labels,
                                 num_layers=args.transformer_num_layers).to(device)
        elif args.transformer_fusion == "encoder":
            model = VqaTransformerEncoder(args.clip_model_name, 
                                 text_model_path=args.text_model_path,
                                 num_labels=num_labels,
                                 num_layers=args.transformer_num_layers).to(device)
    elif args.base_model == "pmc-clip":
        model = PMC_CLIPforVQA(args.pmc_checkpoint, 
                            args.config, 
                            text_model_path=args.text_model_path, 
                            num_labels=num_labels, 
                            pool_type=args.pool_type,
                            device=device).to(device) 

    model.load_state_dict(torch.load(args.checkpoint))
    print(f"Loaded weight from {args.checkpoint}")

    if args.base_model == "clip":
        data_collator = DataCollatorWithPadding(tokenizer)
    elif args.base_model == "pmc-clip":
        data_collator = torch_images_and_label_data_collator
    
    data_loader = DataLoader(processed_dataset['test'], batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)
    
    label2ans = pickle.load(open("./data/cache/trainval_label2ans.pkl", "rb"))

    model.eval()
    tot_correct = 0
    results = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            qids = batch['qid']
            batch.pop('qid')
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            logits = output.logits
            if args.task == "yesno":
                # binary classification
                predictions = logits > 0.5
            else:
                # multi class classification
                predictions = torch.argmax(logits, dim=1)
            correct = (predictions == batch["labels"]).sum()
            tot_correct += correct 
            for i in range(len(batch['labels'])):
                results.append({"qid": qids[i].item(),
                                "gt": batch['labels'][i].item(),
                                "pred_label": predictions[i].item(),
                                "pred_sent": label2ans[predictions[i].item()]})
    
    print(f"Test accuracy: {tot_correct / len(processed_dataset['test'])}")
    df = pd.DataFrame.from_records(results)
    df.to_csv(args.output, index=False)   
    
    for split in args.splits:
        data_loader = DataLoader(processed_dataset[split], batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)
        tot_correct = 0
        with torch.no_grad():
            for batch in tqdm(data_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                print("qid:", batch['qid'])
                output = model(**batch)
                logits = output.logits
                if args.task == "yesno":
                    # binary classification
                    predictions = logits > 0.5
                else:
                    # multi class classification
                    predictions = torch.argmax(logits, dim=1)
                correct = (predictions == batch["labels"]).sum()
                tot_correct += correct 
        
        print(f"{split} accuracy: {tot_correct / len(processed_dataset[split])}")

