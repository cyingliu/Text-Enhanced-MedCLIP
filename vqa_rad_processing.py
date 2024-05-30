"""This script de-duplicates the data provided by the VQA-RAD authors,
creates an "imagefolder" dataset and pushes it to the Hugging Face Hub.

Modified from https://huggingface.co/datasets/flaviagiammarino/vqa-rad/blob/main/scripts/processing.py.
"""

import re
import os
import shutil
import datasets
import pandas as pd
from datasets import load_dataset

# load the data
# train.jsonl is in JSON lines format, converted from trainset.json which is in Multiline JSON format
# test.jsonl is in JSON lines format, converted from testset.json which is in Multiline JSON format
train_data = pd.read_json("train.jsonl", lines=True)
test_data = pd.read_json("test.jsonl", lines=True)

# drop the duplicate image-question-answer triplets
train_data = train_data.drop_duplicates(ignore_index=True)
test_data = test_data.drop_duplicates(ignore_index=True)

# drop the common image-question-answer triplets
train_data = train_data[~train_data.apply(tuple, 1).isin(test_data.apply(tuple, 1))]
train_data = train_data.reset_index(drop=True)

# perform some basic data cleaning/normalization
f = lambda x: re.sub(' +', ' ', str(x).lower()).replace(" ?", "?").strip()
train_data["question"] = train_data["question"].apply(f)
test_data["question"] = test_data["question"].apply(f)
train_data["answer"] = train_data["answer"].apply(f)
test_data["answer"] = test_data["answer"].apply(f)

# copy the images using unique file names
os.makedirs(f"data/train/", exist_ok=True)
train_data.insert(0, "file_name", "")
for i, row in train_data.iterrows():
    file_name = f"img_{i}.jpg"
    train_data["file_name"].iloc[i] = file_name
    shutil.copyfile(src=f"data_RAD/images/{row['file_name']}", dst=f"data/train/{file_name}")
_ = train_data.pop("image_name")

os.makedirs(f"data/test/", exist_ok=True)
test_data.insert(0, "file_name", "")
for i, row in test_data.iterrows():
    file_name = f"img_{i}.jpg"
    test_data["file_name"].iloc[i] = file_name
    shutil.copyfile(src=f"data_RAD/images/{row['file_name']}", dst=f"data/test/{file_name}")
_ = test_data.pop("image_name")

# save the metadata
train_data.to_csv(f"data/train/metadata.csv", index=False)
test_data.to_csv(f"data/test/metadata.csv", index=False)

# push the dataset to the hub
dataset = load_dataset(path='data')
dataset.push_to_hub("cs231n-Medco/vqa-rad")