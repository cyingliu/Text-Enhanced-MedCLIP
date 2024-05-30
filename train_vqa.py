"""
    Fine-tune CLIP-based models on VQA-RAD dataset
    posed as a binary/ mutli-class classification problem.
"""
import os 
import argparse
from pathlib import Path 
import shutil
import pickle

import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, CLIPImageProcessor
from transformers import TrainingArguments, Trainer


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
    parser.add_argument("--text_model_path", type=str)
    # ***** CLIP with Linear Fusion *****#
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    # ***** PMC-CLIP for VQA
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--config", type=str, default="./modeling/pmc_clip/model_configs/RN50_fusion4.json")
    parser.add_argument("--pool_type", type=str, default="average", choices=["average", "cls"])
    # ***** Trainer *****
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument("--evaluation_strategy", type=str, choices=["no", "steps", "epoch"], default="steps")
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--save_strategy", type=str, choices=["no", "steps", "epoch"], default="steps")
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--metric_for_best_model", type=str, choices=["accuracy", "loss"], default="accuracy")
    parser.add_argument("--report_to", type=str, nargs="+", default=["wandb"])
    # ***** Wandb ***** 
    parser.add_argument("--project", type=str, default="cs231n-medclip")
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = f"{args.clip_model_name.split[1]}_{args.task}"

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
        select_columns = ['labels', 'pixel_values', 'input_ids', 'attention_mask']
    elif args.base_model == "pmc-clip":
        preprocess_fn = preprocess_pmcclip
        select_columns = ['image', 'labels', 'bert_input', 'bert_label']

    processed_dataset = train_val_test_dataset.map(preprocess_fn, batched=True)
    if args.task == 'all':
        processed_dataset['test-yesno'] = processed_dataset['test'].filter(lambda example: example['answer'].strip().lower() == 'yes' or example['answer'].strip().lower() == 'no')
        processed_dataset['test-nonyesno'] = processed_dataset['test'].filter(lambda example: example['answer'].strip().lower() != 'yes' and example['answer'].strip().lower() != 'no')
        processed_dataset['test-closed'] = processed_dataset['test'].filter(lambda example: example['answer_type'].strip().lower() == 'closed')
        processed_dataset['test-open'] = processed_dataset['test'].filter(lambda example: example['answer_type'].strip().lower() == 'open')
    
    if args.base_model == "pmc-clip":
        processed_dataset['train'].set_transform(train_transform)
        processed_dataset['val'].set_transform(test_transform)
        processed_dataset['test'].set_transform(test_transform)
        if args.task == 'all':
            processed_dataset['test-yesno'].set_transform(test_transform)
            processed_dataset['test-nonyesno'].set_transform(test_transform)
            processed_dataset['test-closed'].set_transform(test_transform)
            processed_dataset['test-open'].set_transform(test_transform)

    processed_dataset = processed_dataset.select_columns(select_columns)

    # 2. Init model
    num_labels = 2 if args.task == "yesno" else 458 # num of labels in trainval_label2ans.pkl
    if args.base_model == "clip":
        model = CLIPwithLinearFusion(args.clip_model_name, 
                                 text_model_path=args.text_model_path,
                                 num_labels=num_labels,
                                 device=device).to(device)
    elif args.base_model == "pmc-clip":
        model = PMC_CLIPforVQA(args.checkpoint, 
                            args.config, 
                            text_model_path=args.text_model_path, 
                            num_labels=num_labels, 
                            pool_type=args.pool_type,
                            device=device).to(device) 

    # 3. Train model
    training_args = TrainingArguments(output_dir=args.output_dir,
                                      evaluation_strategy=args.evaluation_strategy,
                                      per_device_train_batch_size=args.batch_size,
                                      per_device_eval_batch_size=args.batch_size,
                                      learning_rate=args.learning_rate,
                                      num_train_epochs=args.num_train_epochs,
                                      logging_steps=args.logging_steps,
                                      save_steps=args.save_steps,
                                      save_strategy=args.save_strategy,
                                      save_total_limit=args.save_total_limit,
                                      load_best_model_at_end=args.load_best_model_at_end,
                                      metric_for_best_model=args.metric_for_best_model,
                                      report_to=args.report_to,
                                      save_safetensors=False,
                                      label_names=["labels"])

    if "wandb" in args.report_to:
        wandb.init(project=args.project)

    def compute_accuracy(eval_pred):
        logits = eval_pred.predictions # ndarray
        labels = eval_pred.label_ids # ndarray
        if len(logits.shape) == 1:
            # binary classification
            predictions = logits > 0.5
        else:
            # multi class classification
            predictions = np.argmax(logits, axis=1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}
    
    base_trainer_args = dict(model=model,
                            args=training_args,
                            train_dataset=processed_dataset["train"],
                            eval_dataset=processed_dataset["val"],
                            compute_metrics=compute_accuracy)
    clip_trainer_args = {"tokenizer": tokenizer, **base_trainer_args}
    pmcclip_trainer_args = {"data_collator": torch_images_and_label_data_collator, **base_trainer_args}

    if args.base_model == "clip":
        trainer_args = clip_trainer_args 
    elif args.base_model == "pmc-clip":
        trainer_args = pmcclip_trainer_args

    trainer = Trainer(**trainer_args)

    print("Training ...")
    trainer.train()

    for path in Path(args.output_dir).glob("checkpoint-*"):
        shutil.rmtree(path)
    
    trainer.save_model(f"{args.output_dir}/checkpoint-best")

    # 4. Evaluate
    print("Evaluating on val dataset ... ")
    val_results = trainer.evaluate()
    print(f"Val accuracy: {val_results['eval_accuracy']}")

    print("Evaluating on test dataset ...")
    test_results = trainer.evaluate(processed_dataset["test"])
    print(f"Test accuracy: {test_results['eval_accuracy']}")

    if args.task == 'all':
        splits = ['test-yesno', 'test-nonyesno', 'test-closed', 'test-open']
        for split in splits:
            test_results = trainer.evaluate(processed_dataset[split])
            print(f"{split} accuracy: {test_results['eval_accuracy']}")

    wandb.finish()
