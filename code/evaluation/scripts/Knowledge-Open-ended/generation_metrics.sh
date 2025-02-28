#!/bin/bash
ROOT_PATH=/data/aofei
model_name='llava_med'

export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

dataset=mimic_cxr
baselines=(original DoLa PAI opera avisc m3id VCD damro)
# baselines=(original)

for baseline in "${baselines[@]}"; do
    python ./open_ended_evaluation/generation_metrics.py \
        --model_answers_file ${ROOT_PATH}/hallucination/MedHEval/type2/baselines_${model_name}/${dataset}_open/pred_${baseline}.jsonl \
        --eval_res_file ${ROOT_PATH}/hallucination/MedHEval/type2/baselines_${model_name}/${dataset}_open/eval_all_metrics_${baseline}.txt
    wait
done