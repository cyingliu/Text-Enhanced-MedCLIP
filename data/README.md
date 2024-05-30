# Data
We use the pre-processed data from this [link](https://github.com/aioz-ai/MICCAI21_MMQ?tab=readme-ov-file#vqa-rad-dataset-for-vqa-task), and include label matching files in ```cache/```.

`trainset.json` and `testset.json` are from the above [link](https://github.com/aioz-ai/MICCAI21_MMQ?tab=readme-ov-file#vqa-rad-dataset-for-vqa-task). They are in MultiLine JSON format. We converted them into JSON lines format, named `train.jsonl` and `test.jsonl`. Then run `vqa_rad_processing.py` to further process.