import numpy as np
import copy
from typing import Any, Optional

import torch
import torch.nn as nn

from modeling.transformer_layers import *
from transformers import AutoModel
from transformers.modeling_outputs import ImageClassifierOutput

"""
Based on some code (with modification) from CS231N assigment and https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py
"""
class VqaTransformerEncoder(nn.Module):
    def __init__(self, clip_model_name, text_model_path=None, num_labels=2, num_heads=4,
                 num_layers=2):
        super().__init__()

        self.base_model = AutoModel.from_pretrained(clip_model_name)

        if text_model_path:
            print("Load text model weight from:", text_model_path)
            text_model_dict = torch.load(text_model_path, map_location="cpu")
            text_model_dict = {k: v for k, v in text_model_dict.items() if k.startswith("text_model")}
            base_model_dict = self.base_model.state_dict()
            base_model_dict.update(text_model_dict)
            self.base_model.load_state_dict(base_model_dict)

        self.num_labels = num_labels
        projection_dim = self.base_model.projection_dim
        text_embed_dim = self.base_model.text_embed_dim
        vision_embed_dim = self.base_model.vision_embed_dim
        self.eos_token_id = self.base_model.text_model.eos_token_id
        output_dim = 1 if num_labels == 2 else num_labels # scalar output for binary classification

        encoder_layer = TransformerEncoderLayer(input_dim=projection_dim, num_heads=num_heads)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._init_weights(self.transformer)

        self.text_projection = nn.Linear(text_embed_dim, projection_dim)
        self.vision_projection = nn.Linear(vision_embed_dim, projection_dim)
        self.output = nn.Linear(projection_dim, output_dim)

    def _init_weights(self, module):
        """
        Initialize the weights of the network.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, 
                input_ids: Optional[torch.LongTensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                labels: Optional[torch.Tensor] = None,):
        
        text_features = self.base_model.text_model(input_ids=input_ids,
                                                          attention_mask=attention_mask,
                                                          position_ids=position_ids).last_hidden_state # (N, S, 512)
        text_features = self.text_projection(text_features)
        image_features = self.base_model.vision_model(pixel_values=pixel_values).last_hidden_state # (N, T, 512)
        image_features = self.vision_projection(image_features)

        transformer_input = torch.cat((text_features, image_features), dim=1) # shape: (N, S+T, W)

        transformer_output = self.transformer(transformer_input, None) # shape: (N, S+T, W)

        # Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py
        if self.eos_token_id == 2:
            pooled_output = transformer_output[
                torch.arange(transformer_output.shape[0], device=transformer_output.device),
                input_ids.to(dtype=torch.int, device=transformer_output.device).argmax(dim=-1),
            ]
        else:
            pooled_output = transformer_output[
                torch.arange(transformer_output.shape[0], device=transformer_output.device),
                (input_ids.to(dtype=torch.int, device=transformer_output.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]

        logits = self.output(pooled_output).squeeze() # shape: (N, T, V)
        
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.num_labels == 2:
                # binary classification
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze().float())
            else:
                # multi-class classification
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of a Transformer encoder, to be used with TransformerEncoder.
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        Construct a TransformerEncoderLayer instance.

        Inputs:
         - input_dim: Number of expected features in the input.
         - num_heads: Number of attention heads
         - dim_feedforward: Dimension of the feedforward network model.
         - dropout: The dropout value.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()


    def forward(self, tgt, tgt_mask=None):
        """
        Pass the inputs (and mask) through the encoder layer.

        Inputs:
        - tgt: the sequence to the encoder layer, of shape (N, T, W)
        - tgt_mask: the parts of the target sequence to mask, of shape (T, T)

        Returns:
        - out: the Transformer features, of shape (N, T, W)
        """
        # Perform self-attention on the target sequence (along with dropout and
        # layer norm).
        tgt2 = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Pass
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, tgt_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(output, tgt_mask=tgt_mask)

        return output
