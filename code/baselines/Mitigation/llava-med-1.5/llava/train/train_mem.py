# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
import sys
sys.path.append("/home/avc6555/research/MedH/Mitigation/LVLMs/llava-med-1.5")

# from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
# replace_llama_attn_with_flash_attn()

from llava.train.train import train

if __name__ == "__main__":
    train()
