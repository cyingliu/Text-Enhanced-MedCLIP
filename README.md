# Text-Enhanced-MedCLIP
CS 231N Final Project

## Usage
- Fine-tuning on VQA-RAD dataset
    - CLIP + linear fusion
        ```
        python train_vqa.py \
        --base_model clip \
        --text_model_path [(optional) path_to_text_model_checkpoint] \
        --clip_model_name flaviagiammarino/pubmed-clip-vit-base-patch32 \
        --output_dir path_to_output_directory \
        --learning_rate 2e-6 \
        --num_train_epochs 20 \
        --logging_steps 50 \
        --save_steps 50 \
        --load_best_model_at_end
        ```
    - PMC-CLIP (pre-trained self-attention fusion)
        - Download checkpoint from [pmc_oa_beta](https://huggingface.co/datasets/axiong/pmc_oa_beta/blob/main/checkpoint.pt)
        ```
        python train_vqa.py \
        --base_model pmc-clip \
        --text_model_path [(optional) path_to_text_model_checkpoint] \
        --checkpoint path_to_pmc-clip_checkpoint \
        --config ./modeling/pmc_clip/model_configs/RN50_fusion4.json \
        --pool_type average \
        --output_dir path_to_output_directory \
        --learning_rate 2e-6 \
        --num_train_epochs 20 \
        --logging_steps 50 \
        --save_steps 50 \
        --load_best_model_at_end
        ```
        
