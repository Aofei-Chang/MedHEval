# python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0

import argparse
import os
import random
import json
import math
from tqdm import tqdm

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from transformers import StoppingCriteriaList

from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from minigpt4.conversation.conversation import CONV_VISION_Vicuna0


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0}


# ========================================
#             Gradio Setting
# ========================================
import shortuuid
from PIL import Image

def prepare_texts(texts, conv_temp):
    convs = [conv_temp.copy() for _ in range(len(texts))]
    [conv.append_message(
        conv.roles[0], '<Img><ImageHere></Img> {}'.format(text)) for conv, text in zip(convs, texts)]
    [conv.append_message(conv.roles[1], None) for conv in convs]
    texts = [conv.get_prompt() for conv in convs]
    return texts

def eval_model(args):
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    # load test data
    file_path = os.path.expanduser(args.question_file)
    if file_path.endswith('.jsonl'):
        # Handle JSONL files
        with open(file_path, 'r') as file:
            questions = [json.loads(line) for line in file]
    else:
        # Handle JSON files
        with open(file_path, 'r') as file:
            questions = json.load(file)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "a")

    for line in tqdm(questions):
        idx = line["qid"]
        image_file = line["img_name"]
        max_new_tokens = 1
        question = "[vqa] Based on the image, respond to this question with a short answer: " + line["question"]
        if line['question_type'] == "multi-choice":
            if len(line['choices']) >= 4:
                question += f" Please select one choice. {line['choices']}."
                max_new_tokens = 10
            else:
                continue
        else:
            question += " Please answer Yes or No."
        image_path = os.path.join(args.image_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        image = vis_processor(image)
        conv_temp = CONV_VISION_Vicuna0
        texts = prepare_texts([question], conv_temp)  # warp the texts with conversation template
        answers = model.generate(image.unsqueeze(0), texts, max_new_tokens=max_new_tokens, do_sample=False)

        ans_id = shortuuid.uuid()
        gt_ans = line["answer"]
        ans_file.write(json.dumps({"question_id": idx,
                                   # "final_prompt":prompt,
                                   # "input_ids":str(inputs.input_ids),
                                   "prompt": question,
                                   "model_answer": answers[0],
                                   "ground_truth": gt_ans,
                                   "image_id": line["img_name"],
                                   "answer_id": ans_id,
                                   "model_id": "MiniGPT4",
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()

    eval_model(args)
