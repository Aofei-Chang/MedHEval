#!/bin/bash

ROOT_PATH=/data/aofei
dataset=mimic_cxr
# first, run all the backbone models

# baselines=(RadFM MiniGPT4 XrayGPT CheXagent LLM-CXR Med-flamingo)
baselines=(GPT4o)
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

# for baseline in "${baselines[@]}"; do

#     python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_chair.py \
#         --gt ${ROOT_PATH}/hallucination/${dataset}/type1_open/open_pairs.csv \
#         --pred ${ROOT_PATH}/hallucination/${dataset}/open_inference/${baseline}/mimic_open1_res.csv \
#         --eval_res ${ROOT_PATH}/hallucination/${dataset}/open_inference/${baseline}/eval_chair_res.txt \
#     wait
# done

for baseline in "${baselines[@]}"; do

    python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_chair.py \
        --gt ${ROOT_PATH}/hallucination/${dataset}/type1_open/open_pairs_4gpt4o.csv \
        --pred ${ROOT_PATH}/hallucination/${dataset}/open_inference/${baseline}/mimic_open1_res.csv \
        --eval_res ${ROOT_PATH}/hallucination/${dataset}/open_inference/${baseline}/eval_chair_res.txt \
    wait
done

# backbone=(llava_med llava_med_1.5 llava_v1.6 llava_v1.6_13b)

# then run llava-med and llava-med 1.5
# baselines=(original DoLa PAI opera m3id VCD avisc damro) 

# backbone=llava_med
# # backbone=llava_med_1.5

# for baseline in "${baselines[@]}"; do
#     python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_chair.py \
#         --gt ${ROOT_PATH}/hallucination/${dataset}/type1_open/open_pairs.csv \
#         --pred ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/pred_${baseline}.csv \
#         --eval_res ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/eval_chair_${baseline}.txt
#     wait
# done

# backbone=llava_v1.6
# # backbone=llava_v1.6_13b
# baselines=(original DoLa PAI m3id VCD avisc damro) 
# for baseline in "${baselines[@]}"; do
#     python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_chair.py \
#         --gt ${ROOT_PATH}/hallucination/${dataset}/type1_open/open_pairs.csv \
#         --pred ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/pred_${baseline}.csv \
#         --eval_res ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/eval_chair_${baseline}.txt
#     wait
# done

# backbone=llava_v1.6
# backbone=llava_v1.6_13b
# baselines=(original DoLa PAI m3id VCD damro) 
# # baselines=(original) 
# for baseline in "${baselines[@]}"; do
#     python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_chair.py \
#         --gt ${ROOT_PATH}/hallucination/${dataset}/type1_open/open_pairs.csv \
#         --pred ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/pred_${baseline}.csv \
#         --eval_res ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/eval_chair_${baseline}.txt
#     wait
# done