# python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0

import argparse
import os
import random
import argparse
import os
import random
import json
import math
from tqdm import tqdm

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from xraygpt.common.config import Config
from xraygpt.common.dist_utils import get_rank
from xraygpt.common.registry import registry
from xraygpt.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from xraygpt.datasets.builders import *
from xraygpt.models import *
from xraygpt.processors import *
from xraygpt.runners import *
from xraygpt.tasks import *

from transformers import StoppingCriteria, StoppingCriteriaList


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

import random
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

conv_dict = {'pretrain_vicuna0': CONV_VISION}


# ========================================
#             Gradio Setting
# ========================================
import shortuuid
from PIL import Image

# def prepare_texts(texts, conv_temp):
#     convs = [conv_temp.copy() for _ in range(len(texts))]
#     [conv.append_message(
#         conv.roles[0], '<Img><ImageHere></Img> {}'.format(text)) for conv, text in zip(convs, texts)]
#     [conv.append_message(conv.roles[1], None) for conv in convs]
#     texts = [conv.get_prompt() for conv in convs]
#     return texts

def get_context_emb(conv, img_list, model):
    prompt = conv.get_prompt()
    prompt_segs = prompt.split('<ImageHere>')
    seg_tokens = [
        model.llama_tokenizer(
            seg, return_tensors="pt", add_special_tokens=i == 0).to(model.device).input_ids
        # only add bos to the first seg
        for i, seg in enumerate(prompt_segs)
    ]
    seg_embs = [model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs

def eval_model(args):
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    device = torch.device("cuda")
    # CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.openi.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [torch.tensor([835]).to(device),
                      torch.tensor([2277, 29937]).to(device)]  # '###' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

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
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")


    need_questions = []
    for i in questions:
        if i["question_type"] == "type4_Knowledge":
            need_questions.append(i)
    questions = need_questions
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # ans_file = open(answers_file, "a")

    for line in tqdm(questions):
        idx = line["qid"]
        image_file = line["img_name"]
        max_new_tokens = 24
        if line['question_type'] != "type_1":
            max_new_tokens = 128
        question = line["question"]
        stru_ans = ""
        if line.__contains__("structured_answer"):
            stru_ans = line["structured_answer"]
        image = os.path.join(args.image_folder, image_file)
        # image = Image.open(image_path).convert('RGB')
        # image = vis_processor(image)
        conv = CONV_VISION

        if line["question_type"] == "type_2" or line["question_type"] == "type4_Knowledge":
            pass
        else:
            continue

        #image

        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = vis_processor(raw_image).unsqueeze(0).to(device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = vis_processor(raw_image).unsqueeze(0).to(device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(device)
        img_list = []
        image_emb, _ = model.encode_img(image)
        img_list.append(image_emb)
        conv.messages = []
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        # question = "Could you highlight any abnormalities or concerns in this chest x-ray image?"
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        embs = get_context_emb(conv, img_list, model)

        begin_idx = 0

        embs = embs[:, begin_idx:]
        outputs = model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=1,
            do_sample=True,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1,
            length_penalty=1,
            temperature=1,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Doctor:')[-1].strip()

        ans_id = shortuuid.uuid()
        gt_ans = line["answer"]
        ans_file.write(json.dumps({"question_id": idx,
                                   # "final_prompt":prompt,
                                   # "input_ids":str(inputs.input_ids),
                                   "prompt": question,
                                   "model_answer": output_text,
                                   "ground_truth": gt_ans,
                                   "question_type": line['question_type'],
                                   "structured_answer": stru_ans,
                                   "image_id": line["img_name"],
                                   "answer_id": ans_id,
                                   "model_id": "XrayGPT",
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
