from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch 

def torch_images_and_label_data_collator(features: List[Any]) -> Dict[str, Any]:
    """
        Collate images and label into tensors,
        leave bert_input and bert_label as list of strings,
        which will be tokenized and collated in PMC-CLIP
    """
    first = features[0]
    for k, v in first.items():
        if k not in ("bert_input", "bert_label") and v is not None:
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        else:
            batch[k] = [f[k] for f in features]
    return batch