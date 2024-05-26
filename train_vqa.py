"""
    Fine-tune CLIP-based models on VQA-RAD dataset
    posed as a binary/ mutli-class classification problem.
"""
import os 
import argparse
from pathlib import Path 
import shutil

import wandb
import matplotlib.pyplot as plt
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
    parser.add_argument("--dataset", default="flaviagiammarino/vqa-rad", type=str)
    parser.add_argument("--task", default="binary", choices=["binary", "multiclass"])
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
    train_val_dataset = dataset["train"].train_test_split(test_size=0.125, seed=args.seed)
    train_val_test_dataset = DatasetDict({'train': train_val_dataset['train'],
                                        'val': train_val_dataset['test'],
                                        'test': dataset['test']})
    if args.task == "binary":
        train_val_test_dataset = train_val_test_dataset.filter(lambda example: example["answer"].lower() in ("yes", "no"))

    def preprocess_labels(batch):
        if args.task == "binary":
            batch["labels"] = [1 if answer.lower() == "yes" else 0 for answer in batch["answer"]]
        else: # "multiclass"
            raise NotImplementedError
        return batch 
    train_val_test_dataset = train_val_test_dataset.map(preprocess_labels, batch=True)

    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
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
        remove_columns = ["image", "question", "answer"]
    elif args.base_model == "pmc-clip":
        preprocess_fn = preprocess_pmcclip
        remove_columns = ["question", "answer"]

    processed_dataset = train_val_test_dataset.map(preprocess_fn, batched=True)
    processed_dataset = processed_dataset.remove_columns(remove_columns)
        
    if args.base_model == "pmc-clip":
        processed_dataset['train'].set_transform(train_transform)
        processed_dataset['val'].set_transform(test_transform)
        processed_dataset['test'].set_transform(test_transform)
    
    def compute_accuracy(eval_pred):
        logits = eval_pred.predictions # ndarray
        labels = eval_pred.label_ids # ndarray
        if len(logits.shape) == 1:
            # binary classification
            predictions = logits > 0.5
        else:
            # multi class classification
            predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).mean()

        return {"accuracy": accuracy}
    
    # 2. Init model
    num_labels = 2 if args.task == "binary" else 458 # num of distinct answers in the train set
    
    if args.base_model == "clip":
        model = CLIPwithLinearFusion(args.clip_model_name, 
                                 text_model_path=args.text_model_path,
                                 num_labels=num_labels).to(device)
    elif args.base_model == "pmc-clip":
        model = PMC_CLIPforVQA(args.checkpoint, 
                            args.config, 
                            text_model_path=args.text_model_path, 
                            num_labels=num_labels, 
                            pool_type=args.pool_type).to(device) 

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
                                      save_safetensors=False)

    if "wandb" in args.report_to:
        wandb.init(project=args.project)

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

    wandb.finish()
