dataset=mimic_cxr
baselines=(original DoLa PAI opera avisc m3id VCD damro)
# baselines=(original)
# # conda activate report_eval2

ROOT_PATH=/data/aofei

for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
        --question-file ${ROOT_PATH}/hallucination/${dataset}/type1_open/open_pairs.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/images \
        --answers-file ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_llava_med_1.5/${dataset}_open/pred_${baseline}.jsonl \
        --baseline ${baseline}
    wait
done

for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
        --question-file ${ROOT_PATH}/hallucination/${dataset}/type2_open/open_pairs.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/images \
        --answers-file ${ROOT_PATH}/hallucination/MedHEval/type2/baselines_llava_med_1.5/${dataset}_open/pred_${baseline}.jsonl \
        --baseline ${baseline}
    wait
done

