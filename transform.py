"""
    transforms for pmc-clip
"""
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, CenterCrop

image_size = 224 # follow pmc-clip, model.visual.image_size
crop_scale = 0.9 # follow pmc-clip pre-training
mean = (0.48145466, 0.4578275, 0.40821073)  # OpenAI dataset mean
std = (0.26862954, 0.26130258, 0.27577711)  # OpenAI dataset std

def _convert_to_rgb(image):
    return image.convert('RGB')

train_image_transform =  Compose([
                                RandomResizedCrop(image_size, scale=(crop_scale, 1.0), interpolation=InterpolationMode.BICUBIC),
                                _convert_to_rgb,
                                ToTensor(),
                                Normalize(mean=mean, std=std),
                            ])
test_image_transform = Compose([
                            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                            CenterCrop(image_size),
                            _convert_to_rgb,
                            ToTensor(),
                            Normalize(mean=mean, std=std)
                        ])

def train_transform(batch):
    batch['image'] = [train_image_transform(img) for img in batch['image']]
    return batch

def test_transform(batch):
    batch['image'] = [test_image_transform(img) for img in batch['image']]
    return batch