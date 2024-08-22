from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from transformers import PreTrainedTokenizer
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
import gradio as gr

from training.generate import generate_response, load_model_tokenizer_for_generate
from training.mimiccxr_vq_dataset import CXR_VQ_TOKENIZER_LEN

import argparse
import numpy as np
from typing import List
from pathlib import Path





def shift_vq_tokens(tokens: List[int], tokenizer: PreTrainedTokenizer) -> List[int]:
    assert len(tokenizer) == CXR_VQ_TOKENIZER_LEN
    return [token + len(tokenizer) for token in tokens]


def load_vqgan(config_path, ckpt_path):
    config = OmegaConf.load(config_path)
    vqgan = VQModel(**config.model.params)
    state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
    vqgan.load_state_dict(state_dict, strict=False)
    vqgan.eval()
    return vqgan


def inference_i2t(model, tokenizer, vqgan, instruction, cxr_image_path):
    img = Image.open(cxr_image_path).convert('RGB')
    s = min(img.size)
    target_image_size = 256
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=TF.InterpolationMode.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = T.ToTensor()(img)
    img = 2. * img - 1.

    # Get latent representation (ie VQ-encoding)
    z, _, [_, _, indices] = vqgan.encode(img.unsqueeze(0))
    img_vq = shift_vq_tokens(indices, tokenizer)

    response, response_vq = generate_response(
        (instruction, img_vq), model=model, tokenizer=tokenizer,
        max_new_tokens=512
    )
    return response


def inference_t2i_t2t(model, tokenizer, vqgan, instruction, cxr_text_report):
    response, response_vq = generate_response(
        (instruction, cxr_text_report),
        model=model, tokenizer=tokenizer, max_new_tokens=512
    )

    if response_vq is not None:
        indices = torch.tensor(response_vq)
        quant = vqgan.quantize.get_codebook_entry(indices, shape=(1, 16, 16, -1))
        img = vqgan.decode(quant)
        img = img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        img = np.clip(img, -1., 1.)
        img = (img + 1.) / 2.
    else:
        img = None

    return response, img


import shortuuid
import argparse
import os
import random
import json
import math
from tqdm import tqdm

import random
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    print("Loading model and tokenizer... Watch the system and GPU RAM")
    model_paths = "/path/to/LLM/LLM-CXR/llmcxr_mimic-cxr-256-txvloss-medvqa-stage1_2"
    model, tokenizer = load_model_tokenizer_for_generate(model_paths)
    print("LLM-CXR loaded!")

    print("Loading VQGAN for image encoding/decoding")
    vqgan_config_path = "/path/to/LLM-CXR/vqgan_mimic-cxr-256-txvloss/2023-09-05T13-56-50_mimic-cxr-256-txvloss-project-compat.yaml"
    vqgan_ckpt_path = "/path/to/LLM/LLM-CXR/vqgan_mimic-cxr-256-txvloss/2023-09-05T13-56-50_mimic-cxr-256-txvloss-4e-compat.ckpt"
    vqgan = load_vqgan(vqgan_config_path, vqgan_ckpt_path)
    print("VQGAN loaded!")

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
        image_file = line["img_name"]
        idx = line["qid"]
        question = line["question"]
        stru_ans = ""
        if line.__contains__("structured_answer"):
            stru_ans = line["structured_answer"]
        image_path = os.path.join(args.image_folder, image_file)
        instruction = question
        answer = inference_i2t(model, tokenizer, vqgan, instruction, image_path)

        if line["question_type"] == "type_2" or line["question_type"] == "type4_Knowledge":
            pass
        else:
            continue

        ans_id = shortuuid.uuid()
        gt_ans = line["answer"]
        ans_file.write(json.dumps({"question_id": idx,
                                   # "final_prompt":prompt,
                                   # "input_ids":str(inputs.input_ids),
                                   "prompt": question,
                                   "question_type": line['question_type'],
                                   "structured_answer": stru_ans,
                                   "model_answer": answer,
                                   "ground_truth": gt_ans,
                                   "image_id": line["img_name"],
                                   "answer_id": ans_id,
                                   "model_id": "LLM-CXR",
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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