#!/bin/bash

ROOT_PATH=/data/aofei
dataset=mimic_cxr
# first, run all the backbone models

# baselines=(RadFM MiniGPT4 XrayGPT CheXagent LLM-CXR Med-flamingo)
baselines=(GPT4o)
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

for baseline in "${baselines[@]}"; do

    python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_eval.py \
        --gt ${ROOT_PATH}/hallucination/${dataset}/type1_open/open_pairs_4gpt4o.csv \
        --pred ${ROOT_PATH}/hallucination/${dataset}/open_inference/${baseline}/mimic_open1_res.csv \
        --eval_res ${ROOT_PATH}/hallucination/${dataset}/open_inference/${baseline}/eval_res.csv \
        --report True
    wait
    # python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_all_metrics.py \
    #         --model_answers_file ${ROOT_PATH}/hallucination/${dataset}/open_inference/${baseline}/mimic_open1_res.jsonl \
    #         --eval_res_file ${ROOT_PATH}/hallucination/${dataset}/open_inference/${baseline}/eval_all_metrics.txt \
    #         --RadGraphFile ${ROOT_PATH}/hallucination/${dataset}/open_inference/${baseline}/eval_res.csv

    # wait
done

# then run llava-med and llava-med 1.5
baselines=(original DoLa PAI opera m3id VCD avisc damro) 

backbone=llava_med
# backbone=llava_med_1.5

# for baseline in "${baselines[@]}"; do
#     # python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_eval.py \
#     #     --gt ${ROOT_PATH}/hallucination/${dataset}/type1_open/open_pairs.csv \
#     #     --pred ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/pred_${baseline}.csv \
#     #     --eval_res ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/eval_res_${baseline}.csv \
#     #     --report True
#     # wait
#     python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_all_metrics.py \
#             --model_answers_file ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/pred_${baseline}.jsonl \
#             --eval_res_file ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/eval_all_metrics_${baseline}.txt \
#             --RadGraphFile ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/eval_res_${baseline}.csv

#     wait
# done

