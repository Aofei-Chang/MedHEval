#!/bin/bash

seeds=(4)
ROOT_PATH=/data/aofei
dataset=Slake
# export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

baselines=(original DoLa PAI opera avisc m3id VCD damro)
# baselines=(original)
# # conda activate report_eval2
for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
        --question-file ${ROOT_PATH}/hallucination/${dataset}/slake_qa_pairs.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
        --answers-file ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_llava_med_1.5/${dataset}/pred_${baseline}.jsonl \
        --baseline ${baseline}
    wait
done


dataset=VQA_RAD
baselines=(original DoLa PAI opera avisc m3id VCD damro)
# # conda activate report_eval2
for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
        --question-file ${ROOT_PATH}/hallucination/${dataset}/rad_vqa_pairs.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/images \
        --answers-file ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_llava_med_1.5/${dataset}/pred_${baseline}.jsonl \
        --baseline ${baseline}
    wait
done


dataset=mimic_cxr
baselines=(original DoLa PAI opera avisc m3id VCD damro)
# # conda activate report_eval2
for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
        --question-file ${ROOT_PATH}/hallucination/${dataset}/inference_close1/mimic_cxr_closed_pairs.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/images \
        --answers-file ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_llava_med_1.5/${dataset}_closed/pred_${baseline}.jsonl \
        --baseline ${baseline}
    wait
done


dataset=IU_Xray
baselines=(original DoLa PAI opera avisc m3id VCD damro)
# # conda activate report_eval2
for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
        --question-file ${ROOT_PATH}/hallucination/${dataset}/xray_closed_pairs.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/iu_xray/images \
        --answers-file ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_llava_med_1.5/${dataset}/pred_${baseline}.jsonl \
        --baseline ${baseline}
    wait
done


dataset=mimic_cxr
baselines=(original DoLa PAI opera avisc m3id VCD damro)
# # conda activate report_eval2
for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
        --question-file ${ROOT_PATH}/hallucination/${dataset}/type2_close/mimic_cxr_closed_pairs.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/images \
        --answers-file ${ROOT_PATH}/hallucination/MedHEval/type2/baselines_llava_med_1.5/${dataset}_closed/pred_${baseline}.jsonl \
        --baseline ${baseline}
    wait
done


dataset=mimic_cxr
baselines=(original DoLa PAI opera avisc m3id VCD damro)
# # conda activate report_eval2
for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
        --question-file ${ROOT_PATH}/hallucination/${dataset}/type3/mimic_cxr_closed_pairs.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/images \
        --answers-file ${ROOT_PATH}/hallucination/MedHEval/type3/baselines_llava_med_1.5/pred_${baseline}.jsonl \
        --baseline ${baseline}
    wait
done