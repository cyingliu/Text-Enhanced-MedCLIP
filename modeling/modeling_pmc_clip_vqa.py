from argparse import Namespace
from typing import Any, Optional, Tuple, Union, List
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from transformers.modeling_outputs import ImageClassifierOutput

from .pmc_clip.model import PMC_CLIP 

class PMC_CLIPforVQA(nn.Module):
    """
        Apply a fully connected network on the pooled_output of the fusion module
        Predict a scalar score for binary/multi-class classification
        
        Inputs:
            checkpoint_path: download checkpoint from https://huggingface.co/datasets/axiong/pmc_oa_beta/blob/main/checkpoint.pt
            config_path: RN50_fusion4.json for the provided checkpoint, download configs from PMC-CLIP repo
            text_model_path
            num_labels
            pool_type: "average" or "cls"
            
    """
    def __init__(self, checkpoint_path, config_path, text_model_path=None, num_labels=2, pool_type="average", device="cuda"):
        super().__init__()
        
        model_config = json.load(open(config_path))
        args = dict(bert_model_name=model_config['text_cfg']['bert_model_name'],
                    device=device,
                    mlm=True)
        args = Namespace(**args)
        model_config["args"] = args
        model_config.pop("clip_model")
        self.base_model = PMC_CLIP(**model_config)
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint["state_dict"]
        sd = {k[len('module.'):]: v for k, v in state_dict.items()}
        if "text_encoder.embeddings.position_ids" in sd:
            del sd["text_encoder.embeddings.position_ids"]
        self.base_model.load_state_dict(sd)
        self.cls_id = self.base_model.tokenizer.cls_token_id
        
        if text_model_path:
            print("Load text model weight from:", text_model_path)
            text_model_dict = torch.load(text_model_path)
            text_model_dict = {k: v for k, v in text_model_dict.items() if k.startswith("text_model")}
            base_model_dict = self.base_model.state_dict()
            base_model_dict.update(text_model_dict)
            self.base_model.load_state_dict(base_model_dict)
        
        self.num_labels = num_labels
        projection_dim = self.base_model.transformer_width # 768
        output_dim = 1 if num_labels == 2 else num_labels # scalar output for binary classification
        self.MLP = nn.Sequential(nn.Linear(projection_dim, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, output_dim))
        
        self.pool_type = pool_type
        
    def forward(self,
                bert_input: Optional[List[str]] = None,
                bert_label: Optional[List[str]] = None,
                image: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        image_features = self.base_model.encode_image(image)
        image_features = F.normalize(image_features['image_features'], dim=-1)  # (bs, 768)

        batch = dict(bert_input=bert_input, bert_label=bert_label)
        text_output = self.base_model.encode_text(batch, image_features)

        fusion_features = text_output["fusion_features"] # (bs, 79, 768)

        if self.pool_type == "average":
            pooled_feature = fusion_features.mean(dim=1) # (bs, 768)
        elif self.pool_type == "cls":
            last_token_index = torch.nonzero((text_output["encoded_input"]['input_ids'] == self.cls_id).squeeze())
            pooled_feature = fusion_features[torch.arange(fusion_features.shape[0]), last_token_index[:, 1]] # the 0-index of each row

        logits = self.MLP(pooled_feature).squeeze() # (N,) or (N, C)

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