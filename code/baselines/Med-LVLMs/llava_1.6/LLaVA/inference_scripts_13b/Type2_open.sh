ROOT_PATH=/data/path

export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

python llava/eval/eval_batch.py --num-chunks 1  --model-name ${ROOT_PATH}/LLM/llava-v1.6-vicuna-13b \
    --question-file ${ROOT_PATH}/path/to/Type2_open/file \
    --image-folder ${ROOT_PATH}/path/to/images \
    --question_type open \
    --answers-file ${ROOT_PATH}/path/to/answer
