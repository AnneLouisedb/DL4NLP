# DL4NLP
Mini project proposal deadline: 13 Sep;
Final Report: 14 Oct. 

Dataset: https://github.com/vivian-my/FollowupQG

python ./src/create_dataset.py \
    --api_key "Your API Key (HuggingFace)" \
    --model_name "Model Name" \
    --dataset_name "FollowupQG" \
    --split train \
    --max_new_tokens 16 
