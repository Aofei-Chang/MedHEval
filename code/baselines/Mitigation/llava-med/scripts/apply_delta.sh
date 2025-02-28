ROOT_PATH=/data/aofei
dataset=VQA_RAD

export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

python3 -m llava.model.apply_delta \
    --base /data/aofei/llama_v1_hf \
    --target /data/aofei/LLM/llava_med/VQA_RAD/llava_med \
    --delta /data/aofei/LLM/llava_med/VQA_RAD/data_RAD-9epoch_delta