dataset=mimic_cxr
baselines=(original DoLa PAI opera avisc m3id VCD damro)
# # conda activate report_eval2
for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
        --question-file ${ROOT_PATH}/hallucination/${dataset}/type3/mimic_cxr_closed_pairs.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/images \
        --answers-file ${ROOT_PATH}/hallucination/MedHEval/type3/baselines/pred_${baseline}.jsonl \
        --baseline ${baseline}
    wait
done