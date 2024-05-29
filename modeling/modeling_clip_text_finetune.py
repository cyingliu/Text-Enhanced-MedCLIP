from transformers import AutoModel

from argparse import Namespace
from typing import Any, Optional, Tuple, Union, List
import torch
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json

from .pmc_clip.model import PMC_CLIP 

class CLIPTextFinetunedModule(nn.Module):
    def __init__(self, clip_model_name, num_choices=4):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(clip_model_name)
        projection_dim = self.base_model.projection_dim
        self.MLP = nn.Sequential(nn.Linear(projection_dim, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 1))
        self.num_choices = num_choices

    def forward(
            self, 
            input_ids: Optional[torch.LongTensor] = None, 
            attention_mask: Optional[torch.Tensor] = None,
            ):
        batch_size = input_ids.shape[0] // self.num_choices
        text_features = self.base_model.text_model(input_ids, attention_mask).pooler_output
        logits = self.MLP(text_features).squeeze()
        logits = logits.view(batch_size, self.num_choices)
        return logits

class PMC_CLIPTextFinetunedModule(nn.Module):
    """
        
        Inputs:
            checkpoint_path: download checkpoint from https://huggingface.co/datasets/axiong/pmc_oa_beta/blob/main/checkpoint.pt
            config_path: RN50_fusion4.json for the provided checkpoint, download configs from PMC-CLIP repo
            
    """
    def __init__(self, checkpoint_path, config_path, num_choices=4, device="cuda"):
        super().__init__()
        
        model_config = json.load(open(config_path))
        args = dict(bert_model_name=model_config['text_cfg']['bert_model_name'],
                    device=device,
                    mlm=True)
        args = Namespace(**args)
        model_config["args"] = args
        model_config.pop("clip_model")
        self.base_model = PMC_CLIP(**model_config)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["state_dict"]
        sd = {k[len('module.'):]: v for k, v in state_dict.items()}
        if "text_encoder.embeddings.position_ids" in sd:
            del sd["text_encoder.embeddings.position_ids"]
        self.base_model.load_state_dict(sd)

        text_embed_dim = self.base_model.transformer_width # 768
        self.MLP = nn.Sequential(nn.Linear(text_embed_dim, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 1))
        self.num_choices = num_choices
        
        projection_dim = self.base_model.transformer_width # 768
        self.MLP = nn.Sequential(nn.Linear(projection_dim, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 1))
                
    def forward(self,
            input_ids: Optional[torch.LongTensor] = None, 
            attention_mask: Optional[torch.Tensor] = None,
    ):
        batch_size = input_ids.shape[0] // self.num_choices
        text_features = self.base_model.text_encoder(input_ids, attention_mask).pooler_output
        logits = self.MLP(text_features).squeeze()
        logits = logits.view(batch_size, self.num_choices)
        return logits

"""
Modified from CS231N assignment code.
"""    
def validate_model(loader, data_type, model, device, batch_count=None):
    # Start the timer
    start_time = time.time()
    if data_type == "validation":
        print('Checking accuracy on validation set')
    elif data_type == "train":
        print('Checking accuracy on train set')
    elif data_type == "test":
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    loss = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch_count is not None and i >= batch_count:
                break
            input_ids = batch['input_ids']
            input_ids = input_ids.to(device=device, dtype=torch.long)
            attention_mask = batch['attention_mask']
            attention_mask = attention_mask.to(device=device, dtype=torch.long)
            y = batch['labels']
            y = y.to(device=device, dtype=torch.long)
            logits = model(input_ids, attention_mask)
            loss += F.cross_entropy(logits, y) * y.size(0)
            _, preds = logits.max(1)
            preds = preds.squeeze()
            num_correct += (preds == y).sum()
            num_samples += y.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    # Stop the timer
    end_time = time.time()

    # Print the elapsed time
    print(f"Elapsed time: {end_time - start_time} seconds")
    return loss / num_samples, acc

"""
Modified from CS231N assignment code.
"""
def train(args, model, loader_train, loader_val, optimizer, device, wandb, epochs=1, print_every=200, lr=None, wd=None):
    example_ct = 0

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for epoch in range(epochs):
        # Start the timer
        start_time = time.time()
        for step, batch in enumerate(loader_train):
            model.train()  # put model to training mode
            input_ids = batch['input_ids']
            input_ids = input_ids.to(device=device, dtype=torch.long)
            attention_mask = batch['attention_mask']
            attention_mask = attention_mask.to(device=device, dtype=torch.long)
            y = batch['labels']
            y = y.to(device=device, dtype=torch.long)

            logits = model(input_ids, attention_mask) # (N, num_choice, 1)
            train_loss = F.cross_entropy(logits, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            train_loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            ### Wandb ###
            example_ct += y.size(0)
            metrics = {"train/per_batch_train_loss": train_loss,
                       "train/example_ct": example_ct}

            # 🐝 Log train metrics to wandb
            wandb.log(metrics)

            if step % print_every == 0:
                print('====== Epoch %d Step %d ======' % (epoch, step))

                val_loss, val_accuracy = validate_model(loader_val, "validation", model, device)
                train_loss, train_accuracy = validate_model(loader_train, "train", model, device, batch_count=300)

                # 🐝 Log train and validation metrics to wandb
                val_metrics = {"val/val_loss": val_loss,
                              "val/val_accuracy": val_accuracy,
                              "train/train_loss": train_loss,
                              "train/train_accuracy": train_accuracy,}
                wandb.log({**val_metrics})

                print(f"Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.2f}, Valid Loss: {val_loss:3f}, Valid Accuracy: {val_accuracy:.2f}")

        # 🐝 Log train and validation metrics to wandb
        val_metrics = {"val/per_epoch_val_loss": val_loss,
                      "val/per_epoch_val_accuracy": val_accuracy,
                       "train/per_epoch_train_loss": train_loss,
                       "train/per_epoch_train_accuracy": train_accuracy,}
        wandb.log({**val_metrics})

        print(f"Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.2f}, Valid Loss: {val_loss:3f}, Valid Accuracy: {val_accuracy:.2f}")
        # Stop the timer
        end_time = time.time()

        # Print the elapsed time
        print(f"Epoch {epoch}. Elapsed time: {end_time - start_time} seconds")

        torch.save(model.state_dict(), f"./{args.output_dir}_lr-{lr}-wd-{wd}-epoch-{epoch+1}-shuffled.pth")