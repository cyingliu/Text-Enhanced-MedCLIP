from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from transformers.modeling_outputs import ImageClassifierOutput

class CLIPwithLinearFusion(nn.Module):
    """
        Apply a fully connected network on concatenated text and image features
        Predict a scalar score for binary/multi-class classification
    """
    def __init__(self, clip_model_name, text_model_path=None, num_labels=2):
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
        output_dim = 1 if num_labels == 2 else num_labels # scalar output for binary classification
        self.MLP = nn.Sequential(nn.Linear(2 * projection_dim, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, output_dim))
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        
        text_features = self.base_model.get_text_features(input_ids=input_ids,
                                                          attention_mask=attention_mask,
                                                          position_ids=position_ids) # (N, 512)
        image_features = self.base_model.get_image_features(pixel_values=pixel_values) # (N, 512)
        concat_features = torch.cat([text_features, image_features], dim=1) # (N, 1024)
        logits = self.MLP(concat_features).squeeze() # (N,) or (N, C)
        
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