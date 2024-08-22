
ROOT_PATH=/data/path

export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

python llava/eval/run_med_datasets_eval_batch.py --num-chunks 1  --model-name ${ROOT_PATH}/LLM/llava_med \
    --question-file ${ROOT_PATH}/path/to/Type3/file \
    --image-folder ${ROOT_PATH}/path/to/images \
    --answers-file ${ROOT_PATH}/path/to/answer