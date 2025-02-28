import os
from collections import namedtuple

import torch
import yaml
from PAI_files.CFG import CFGLogits


# import sys
# sys.path.append("/home/avc6555/research/MedH/Mitigation/LVLMs/llava-med")
# from llava import LlavaLlamaForCausalLM
# from constants import (
#     DEFAULT_IMAGE_PATCH_TOKEN,
#     IMAGE_TOKEN_INDEX,
#     IMAGE_TOKEN_LENGTH,
#     MINIGPT4_IMAGE_TOKEN_LENGTH,
#     SHIKRA_IMAGE_TOKEN_LENGTH,
#     SHIKRA_IMG_END_TOKEN,
#     SHIKRA_IMG_START_TOKEN,
# )
# from llava.mm_utils import get_model_name_from_path
# from llava.model.builder import load_pretrained_model
# from minigpt4.common.eval_utils import init_model
# from mllm.models import load_pretrained


def load_model_args_from_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    ModelArgs = namedtuple("ModelArgs", data["ModelArgs"].keys())
    TrainingArgs = namedtuple("TrainingArgs", data["TrainingArgs"].keys())

    model_args = ModelArgs(**data["ModelArgs"])
    training_args = TrainingArgs(**data["TrainingArgs"])

    return model_args, training_args


def load_llava_model(model_path):
    model_name = get_model_name_from_path(model_path)
    model_base = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )
    return tokenizer, model, image_processor, model



class MiniGPT4Config:
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.options = None


def load_model(model):
    if model == "llava-1.5":
        model_path = os.path.expanduser("/path/to/llava-v1.5-7b")
        return load_llava_model(model_path)

    elif model == "minigpt4":
        cfg_path = "./minigpt4/eval_config/minigpt4_eval.yaml"
        return load_minigpt4_model(cfg_path)

    elif model == "shikra":
        yaml_path = "./mllm/config/config.yml" 
        return load_shikra_model(yaml_path)

    else:
        raise ValueError(f"Unknown model: {model}")


def prepare_llava_inputs(template, query, image, tokenizer):
    image_tensor = image["pixel_values"][0]
    qu = [template.replace("<question>", q) for q in query]
    batch_size = len(query)

    chunks = [q.split("<ImageHere>") for q in qu]
    chunk_before = [chunk[0] for chunk in chunks]
    chunk_after = [chunk[1] for chunk in chunks]

    token_before = (
        tokenizer(
            chunk_before,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )
    token_after = (
        tokenizer(
            chunk_after,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )
    bos = (
        torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
        * tokenizer.bos_token_id
    )

    img_start_idx = len(token_before[0]) + 1
    img_end_idx = img_start_idx + IMAGE_TOKEN_LENGTH
    image_token = (
        torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
        * IMAGE_TOKEN_INDEX
    )

    input_ids = torch.cat([bos, token_before, image_token, token_after], dim=1)
    kwargs = {}
    kwargs["images"] = image_tensor.half()
    kwargs["input_ids"] = input_ids

    return qu, img_start_idx, img_end_idx, kwargs

# Example usage:
# prepare_inputs_for_model(args, image, model, tokenizer, kwargs)

def init_cfg_processor(tokenizer, llm_model, questions, gamma=1.1, beam=1, start_layer=0, end_layer=32):
    chunks = [q.split("<image>") for q in questions]
    for i in range(len(chunks)):
        if chunks[i][0] == "":
            chunks[i] = chunks[i][1:]
    if len(chunks[0]) > 1:
        chunk_before = [chunk[0] for chunk in chunks]
        chunk_after = [chunk[1] for chunk in chunks]

        token_before = tokenizer(
            chunk_before,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        ).input_ids.to("cuda")
        token_after = tokenizer(
            chunk_after,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        ).input_ids.to("cuda")

        batch_size = len(questions)
        bos = (
            torch.ones(
                [batch_size, 1], dtype=token_before.dtype, device=token_before.device
            )
            * tokenizer.bos_token_id
        )
        neg_promt = torch.cat([bos, token_before, token_after], dim=1)
    else:
        tokens = tokenizer(
            questions,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        ).input_ids.to("cuda")
        batch_size = len(questions)
        bos = (
            torch.ones(
                [batch_size, 1], dtype=tokens.dtype, device=tokens.device
            )
            * tokenizer.bos_token_id
        )
        neg_promt = torch.cat([bos, tokens], dim=1)
    neg_promt = neg_promt.repeat(beam, 1)
    logits_processor = CFGLogits(gamma, neg_promt.to("cuda"), llm_model, start_layer=start_layer, end_layer=end_layer)

    return logits_processor


class ModelLoader:
    def __init__(self, model_name, model=None):
        self.model_name = model_name
        self.tokenizer = None
        self.vlm_model = None
        self.llm_model = None
        self.image_processor = None
        self.model=model
        self.load_model()

    def load_model(self):
        if self.model_name == "llava-1.5":
            model_path = os.path.expanduser("/path/to/models/llava-v1.5-7b")
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_llava_model(model_path)
            )
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def prepare_inputs_for_model(self, template, query, image):
        if self.model_name == "llava-1.5":
            questions, img_start_idx, img_end_idx, kwargs = prepare_llava_inputs(
                template, query, image, self.tokenizer
            )
        elif self.model_name == "llava-med":
            questions, img_start_idx, img_end_idx, kwargs = prepare_llava_med_inputs(
                template, query, image, self.vlm_model
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        self.img_start_idx = img_start_idx
        self.img_end_idx = img_end_idx

        return questions, kwargs

    def init_cfg_processor(self, questions, gamma=1.1, beam=1, start_layer=0, end_layer=32):
        if self.model_name == "minigpt4":
            chunks = [q.split("<Img><ImageHere></Img>") for q in questions]
        elif self.model_name == "llava-1.5":
            chunks = [q.split("<ImageHere>") for q in questions]
        elif self.model_name == "shikra":
            split_token = (
                "<im_start>"
                + DEFAULT_IMAGE_PATCH_TOKEN * SHIKRA_IMAGE_TOKEN_LENGTH
                + "<im_end>"
            )
            chunks = [q.split(split_token) for q in questions]
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        chunk_before = [chunk[0] for chunk in chunks]
        chunk_after = [chunk[1] for chunk in chunks]

        token_before = self.tokenizer(
            chunk_before,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        ).input_ids.to("cuda")
        token_after = self.tokenizer(
            chunk_after,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        ).input_ids.to("cuda")

        batch_size = len(questions)
        bos = (
            torch.ones(
                [batch_size, 1], dtype=token_before.dtype, device=token_before.device
            )
            * self.tokenizer.bos_token_id
        )
        neg_promt = torch.cat([bos, token_before, token_after], dim=1)
        neg_promt = neg_promt.repeat(beam, 1)
        logits_processor = CFGLogits(gamma, neg_promt.to("cuda"), self.llm_model, start_layer=start_layer, end_layer=end_layer)

        return logits_processor

    def decode(self, output_ids):
        # get outputs
        if self.model_name == "llava-1.5":
            # replace image token by pad token
            output_ids = output_ids.clone()
            output_ids[output_ids == IMAGE_TOKEN_INDEX] = torch.tensor(
                0, dtype=output_ids.dtype, device=output_ids.device
            )

            output_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            output_text = [text.split("ASSISTANT:")[-1].strip() for text in output_text]

        elif self.model_name == "minigpt4":
            output_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            output_text = [
                text.split("###")[0].split("Assistant:")[-1].strip()
                for text in output_text
            ]

        elif self.model_name == "shikra":
            output_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            output_text = [text.split("ASSISTANT:")[-1].strip() for text in output_text]

        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        return output_text