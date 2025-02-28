# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence
import torchvision.transforms as transforms

import torch

import transformers
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava import LlavaLlamaForCausalLM
from llava.model.moe_llava import LoRA_MOE_FFN, LoRA_MOE_QK, LoRA_MOE_QK_old

from PIL import Image
import torch.nn as nn
import numpy as np
import math

# TODO: import and use code from ../data/dataset.py

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_IMAGE_TOKEN_INDEX = 39

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    segment_path: str = field(default=None,
                           metadata={"help": "Path to the segment data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_token_len: int = 0
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    use_bbox: bool = False
    use_mask: bool = False
    bbox_size: int = 256


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    moe_lora_r: int = 16
    lora_alpha: int = 16
    moe_lora_alpha: int = 8
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    visual_focus: bool = False
    visual_enhance_ratio: float = 0.1
    bbox_ratio: float = 0.05
    use_moe: bool = False
    dense_moe: bool = False
    expert_num: int = 4
    query_expert_num: int = 4
    visual_expert_num: int = 4
    do_attn_probing: bool = False
    use_kl: bool = False
    seed: int = 4
    top_heads: int = 0
    use_visual_prompt: bool = False
    moe_balance_ratio: float = 0.0
    top_visual_moe_experts: int = 1


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _tokenize_fn_split(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer, image_start_token=DEFAULT_IM_START_TOKEN, split_token="\n###") -> Dict:
    """Tokenize a list of strings."""

    # Convert tokens to token IDs
    token_1, token_2 = image_start_token, split_token
    # token_1_id = tokenizer.convert_tokens_to_ids(token_1)
    # token_2_id = tokenizer.convert_tokens_to_ids(token_2)
    
    tokenized_data = {
        "input_ids": [],
        "labels": [],
        "input_ids_lens": [],
        "labels_lens": [],
        "token_1_indices": [],
        "token_2_indices": []
    }
    for text in strings:
        # Split the text by token_1 to locate the segment containing token_2
        split_text = text.split(token_1, 1)
        before_token_1 = split_text[0]
        after_token_1 = split_text[1] if len(split_text) > 1 else ""
        
        # Further split after token_1 to locate the second-to-last occurrence of token_2
        split_after_token_1 = after_token_1.split(token_2)
        # print(split_after_token_1)
        
        # Join all segments up to the second-to-last occurrence of token_2
        if len(split_after_token_1) >= 3:
            between_tokens = token_2.join(split_after_token_1[:-3]) + token_2 + split_after_token_1[-3]
            # after_token_2 = split_after_token_1[-1]
            after_token_2 = token_2.join(split_after_token_1[-2:])
        else:
            between_tokens = split_after_token_1[0] if len(split_after_token_1) > 1 else ""
            after_token_2 = split_after_token_1[-1] if len(split_after_token_1) > 1 else ""
        # print(after_token_2, "frf")
        # Tokenize each part without padding
        tokenized_before_token_1 = tokenizer(before_token_1, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True, padding=False)
        tokenized_between_tokens = tokenizer(between_tokens, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True, padding=False)
        tokenized_after_token_2 = tokenizer(after_token_2, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True, padding=False)
        
        # Combine all parts for final input with padding
        tokenized_full = tokenizer(text, return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True)
        
        # Calculate token indices based on segment lengths
        token_1_index = tokenized_before_token_1.input_ids.size(1) if after_token_1 else -1
        token_2_index = token_1_index + tokenized_between_tokens.input_ids.size(1) if len(split_after_token_1) > 2 else -1
        
        # Append results to lists
        tokenized_data["input_ids"].append(tokenized_full.input_ids[0])
        tokenized_data["labels"].append(tokenized_full.input_ids[0])
        tokenized_data["input_ids_lens"].append(tokenized_full.input_ids.ne(tokenizer.pad_token_id).sum().item())
        tokenized_data["labels_lens"].append(tokenized_full.input_ids.ne(tokenizer.pad_token_id).sum().item())
        tokenized_data["token_1_indices"].append(token_1_index if token_1_index != -1 else [DEFAULT_IMAGE_TOKEN_INDEX])
        tokenized_data["token_2_indices"].append(token_2_index if token_2_index != -1 else [-1])
    # print(tokenized_data["token_1_indices"], tokenized_data["token_2_indices"], tokenized_before_token_1.input_ids.size(1) + tokenized_between_tokens.input_ids.size(1) + tokenized_after_token_2.input_ids.size(1))
    return tokenized_data

def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = str(sentence["value"])
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    multimodal_cfg: dict,
    cur_token_len: int,
) -> Dict:
    is_multimodal = multimodal_cfg['is_multimodal']
    # image_token_len = multimodal_cfg['image_token_len']
    image_token_len = cur_token_len
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            if multimodal_cfg['use_im_start_end']:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN

            if isinstance(sentence["value"], int):
                sentence["value"] = str(sentence["value"])
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "v1":
        # print("use version 1")
        return preprocess_v1(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    # print(conversation_lib.default_conversation, "dnj")
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # print("conv:", conversations)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn_split(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets, 
                image_token_start_index=conversations_tokenized["token_1_indices"],
                question_token_end_index=conversations_tokenized["token_2_indices"])


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...")
        sources = [example["conversations"] for example in list_data_dict]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, segment_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 multimodal_cfg: dict):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))
        loaded_segments = dict()
        if segment_path is not None:
            loaded_segments = np.load(segment_path, allow_pickle=True)

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.segments_dict = loaded_segments
        self.multimodal_cfg = multimodal_cfg

        self.H, self.W = 16, 16
        n_px = 224
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(n_px, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(n_px),
            transforms.Resize(self.H, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            # print(sources, "image exists!")
            image_file = self.list_data_dict[i]['image']
            image_folder = self.multimodal_cfg['image_folder']
            processor = self.multimodal_cfg['image_processor']
            try:
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            except Exception as exn:
                print(exn)
                import random
                return random.choice(self)

            # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.multimodal_cfg['image_aspect_ratio'] == 'keep':
                max_hw, min_hw = max(image.size), min(image.size)
                aspect_ratio = max_hw / min_hw
                max_len, min_len = 448, 224
                shortest_edge = int(min(max_len / aspect_ratio, min_len))
                image = processor.preprocess(image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge})['pixel_values'][0]
            elif self.multimodal_cfg['image_aspect_ratio'] == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            # import pdb; pdb.set_trace()
            image_token_len = self.multimodal_cfg["image_token_len"]
            patch_size = int(image.shape[1]//math.sqrt(image_token_len))
            cur_token_len = (image.shape[1]//patch_size) * (image.shape[2]//patch_size)   # FIXME: 14 is hardcoded patch size

            try:
                sources = copy.deepcopy([e["conversations"] for e in sources])
            except:
                sources = copy.deepcopy([e["conversatons"] for e in sources])
            sources = preprocess_multimodal(
                sources,
                self.multimodal_cfg, cur_token_len)
        else:
            try:
                sources = copy.deepcopy([e["conversations"] for e in sources])
            except:
                sources = copy.deepcopy([e["conversatons"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             image_token_start_index=data_dict["image_token_start_index"][0],
                             question_token_end_index=data_dict["question_token_end_index"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            # print("with image", image)
            data_dict['image'] = image
        elif self.multimodal_cfg['is_multimodal']:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.multimodal_cfg['image_processor'].crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            # print("no image, ", data_dict['image'])
        # print(data_dict)
        # image_start_index = torch.where(data_dict['input_ids'] == 32002)
        # image_end_index = torch.where(data_dict['input_ids'] == 32003)
        # print(image_start_index, image_end_index, "iamge token")
        # print(data_dict['input_ids'], data_dict['input_ids'].size())
        # add bounding boxes
        if self.multimodal_cfg['use_bbox']:
            # if 'bboxes' in self.list_data_dict[i]:
            ih, iw = 256, 256
            resize_ratio = 1
            if self.multimodal_cfg['bbox_size'] > 0:
                bbox_size = self.multimodal_cfg['bbox_size']
                if bbox_size != 256:
                    resize_ratio = 256 / bbox_size
            # mask = torch.zeros(size=(ih, iw))
            use_gt_data = False
            gt_bbox_masks = []
            # use_gt_data = self.list_data_dict[i].__contains__('gt_data') and self.list_data_dict[i]['gt_data']
            if not self.multimodal_cfg['use_mask'] or use_gt_data:
                bboxes = self.list_data_dict[i]['bboxes']
                bbox_masks = []
                # bbox_masks could also contain a dict with the entity, entity indices
                for bbox in bboxes:
                    mask = torch.zeros(size=(ih, iw))
                    x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int((bbox[0] + bbox[2])), int((bbox[1] + bbox[3]))
                    mask[y_min: y_max, x_min: x_max] = 1
                    mask = self.mask_transform(mask.numpy())[0]
                    # mask = mask.cuda() if torch.cuda.is_available() else mask
                    bbox_masks.append(mask)
                if use_gt_data: # get the indices of the entities
                    entity_indices = []
                    entities = list(self.list_data_dict[i]['bboxes_dict'].keys())
                    for entity in entities:
                        entity_ids = self.tokenizer(entity, return_tensors="pt", max_length=self.tokenizer.model_max_length, truncation=True, padding=False)['input_ids'][0][1:] # skip special token
                        if isinstance(entity_ids, int):
                            entity_token_length = 1
                        else:
                            entity_token_length = len(entity_ids)
                        for idx in range(len(data_dict["input_ids"]) - entity_token_length + 1):
                        # Check if the sliced portion of input_ids matches the entity token IDs
                            # if data_dict["input_ids"][idx:idx + entity_token_length] == entity_ids:
                            if torch.all(data_dict["input_ids"][idx:idx + entity_token_length] == entity_ids):
                                # Append the start index of the entity in input_ids to entity_indices
                                entity_indices.append((idx, idx + entity_token_length))
                                break
                    bbox_masks.append({"entity_indices": entity_indices}) # example:[(308, 311)] entity indices
                    # print(entity_indices, "entity indices")
            else:# if we use masks instead of bboxes
                str_id = str(self.list_data_dict[i]['id'])
                bboxes = self.segments_dict[str_id]
                bbox_masks = []
                for bbox in bboxes:
                    mask = torch.tensor(bbox, dtype=torch.float32)
                    mask = self.mask_transform(mask.numpy())[0]
                    # mask = mask.cuda() if torch.cuda.is_available() else mask
                    bbox_masks.append(mask)
                if 'gt_bboxes' in self.list_data_dict[i]:
                    gt_bboxes = self.list_data_dict[i]['gt_bboxes']
                    gt_bbox_masks = []
                    for bbox in gt_bboxes:
                        mask = torch.zeros(size=(ih, iw))
                        x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int((bbox[0] + bbox[2])), int((bbox[1] + bbox[3]))
                        mask[y_min: y_max, x_min: x_max] = 1
                        mask = self.mask_transform(mask.numpy())[0]
                        gt_bbox_masks.append(mask)
            data_dict['bboxes'] = [{"weak": bbox_masks, "gt": gt_bbox_masks}]
        
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        bboxes = None
        image_token_start_indices, question_token_end_indices = None, None
        if "image_token_start_index" in instances[0]:
            image_token_start_indices, question_token_end_indices = tuple([instance[key] for instance in instances]
                                  for key in ("image_token_start_index", "question_token_end_index"))
        if "bboxes" in instances[0]:
            input_ids, labels, bboxes = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", "bboxes"))
        else:
            input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        
        if bboxes is not None:
            batch['bboxes'] = bboxes
        if image_token_start_indices is not None:
            
            batch['image_token_start_index'] = image_token_start_indices
            batch['question_token_end_index'] = question_token_end_indices

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (LazySupervisedDataset
                   if data_args.lazy_preprocess else SupervisedDataset)
    train_dataset = dataset_cls(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                segment_path=data_args.segment_path,
                                multimodal_cfg=dict(
                                    is_multimodal=data_args.is_multimodal,
                                    image_token_len=data_args.image_token_len,
                                    image_folder=data_args.image_folder,
                                    use_bbox=data_args.use_bbox,
                                    use_mask=data_args.use_mask,
                                    bbox_size=data_args.bbox_size,
                                    image_aspect_ratio=data_args.image_aspect_ratio,
                                    use_im_start_end=getattr(data_args, 'mm_use_im_start_end', False),
                                    image_processor=getattr(data_args, 'image_processor', None)))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def find_all_linear_names(model, excluding_modules=[]):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if any(ex_keyword in name for ex_keyword in excluding_modules):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def get_peft_state(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if ("lora_" in k and "moe" not in k)}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    # to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if ("lora_" not in k or "moe" in k)}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    # to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def train():
    global local_rank
    

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    import random
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Set seed to {seed}")

    set_seed(training_args.seed)

    if model_args.vision_tower is not None:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )
        model.enhance_visual = training_args.visual_focus
        model.visual_enhance_ratio = training_args.visual_enhance_ratio
        model.bbox_ratio = training_args.bbox_ratio
        model.top_heads = training_args.top_heads
        model.use_kl = training_args.use_kl
        model.use_visual_prompt = training_args.use_visual_prompt
        model.moe_balance_ratio = training_args.moe_balance_ratio
        if model.use_visual_prompt:
            model.init_visual_prompt_flag = False
        print("visual tuning parameters:", model.enhance_visual, model.visual_enhance_ratio, model.bbox_ratio)
        print("Use KL loss:", model.use_kl)

    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    if model_args.vision_tower is not None:
        model_vision_dict = model.model.initialize_vision_modules(
            vision_tower=model_args.vision_tower,
            mm_vision_select_layer=model_args.mm_vision_select_layer,
            pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter
        )
        dtype = torch.float32
        if training_args.fp16:
            dtype = torch.float16
        if training_args.bf16:
            dtype = torch.bfloat16
        model.model.vision_tower[0].to(dtype=dtype, device=training_args.device)
        vision_config = model_vision_dict['vision_config']

        data_args.image_token_len = model_vision_dict['image_token_len']
        data_args.image_processor = model_vision_dict['image_processor']
        data_args.is_multimodal = True

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        # model.requires_grad_(False)
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.model.mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.model.mm_projector.parameters():
                p.requires_grad = False

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        vision_config.use_im_start_end = training_args.use_im_start_end = model_args.mm_use_im_start_end
        print(f"Tune mm_mlp_adapter: {model_args.tune_mm_mlp_adapter} rr")
        model.initialize_vision_tokenizer(mm_use_im_start_end=model_args.mm_use_im_start_end, tokenizer=tokenizer, device=training_args.device,
                                          tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter, pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter)

        params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
        if len(params_no_grad) > 0:
            if training_args.fsdp is not None and len(training_args.fsdp) > 0:
                if len(params_no_grad) < 10:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
                else:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'. format(len(params_no_grad), ', '.join(params_no_grad[:10])))
                print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
                print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
                def patch_FSDP_use_orig_params(func):
                    def wrap_func(*args, **kwargs):
                        use_orig_params = kwargs.pop('use_orig_params', True)
                        return func(*args, **kwargs, use_orig_params=use_orig_params)
                    return wrap_func

                FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)


    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    # if training_args.bits == 16:
    #     if training_args.bf16:
    #         model.to(torch.bfloat16)
    #     if training_args.fp16:
    #         model.to(torch.float16)
    # for i, p in model.model.named_parameters():
    #     if p.requires_grad:
    #         print(i)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        excluding_modules = []
        if training_args.use_moe:
            model.use_moe = True
            excluding_modules = ["k_proj", "q_proj"]
            # excluding_modules = ["q_proj", "mlp", "o_proj", "k_proj"]
            # if training_args.visual_expert_num > 1:
            #     excluding_modules += ["k_proj"]
            #     model.use_moe = True
            # if training_args.query_expert_num > 1:
            #     excluding_modules += ["q_proj"]
            #     model.use_moe = True
        if training_args.do_attn_probing:
            excluding_modules = ["v_proj", "o_proj", "mlp"]
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model, excluding_modules=excluding_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        # model.model.mm_projector.to(torch.float32)
        rank0_print("Adding LoRA adapters...")
        # model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)

    
    if model_args.tune_mm_mlp_adapter:
        # model.requires_grad_(False)
        if training_args.lora_enable:
            for p in model.model.model.mm_projector.parameters():
                p.requires_grad = True
        else:
            for p in model.model.mm_projector.parameters():
                p.requires_grad = True


    if training_args.use_moe and training_args.lora_enable:
        # assert self.args.use_lora
        # assert  'gate_proj' not in self.args.lora_modules and \
        #         'up_proj' not in self.args.lora_modules and \
        #         'down_proj' not in self.args.lora_modules
        
        num_layers = len(model.base_model.model.model.layers)
        top_layers = []
        # top_layers = [12, 13, 17, 18, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31]
        for i in range(num_layers):
            if len(top_layers)==0 or i in top_layers:
                original_q = model.base_model.model.model.layers[i].self_attn.q_proj
                model.base_model.model.model.layers[i].self_attn.q_proj = \
                    LoRA_MOE_QK(args=training_args,
                        lora_rank=training_args.moe_lora_r,
                        lora_alpha=training_args.moe_lora_alpha,
                        num_experts=training_args.query_expert_num,
                        original_module=original_q,
                        dense_moe=training_args.dense_moe).bfloat16()

                original_k = model.base_model.model.model.layers[i].self_attn.k_proj
                model.base_model.model.model.layers[i].self_attn.k_proj = \
                    LoRA_MOE_QK_old(args=training_args,
                        lora_rank=training_args.moe_lora_r,
                        lora_alpha=training_args.moe_lora_alpha,
                        num_experts=training_args.visual_expert_num,
                        top_moe_experts=training_args.top_visual_moe_experts,
                        original_module=original_k).bfloat16()
            else:
                original_q = model.base_model.model.model.layers[i].self_attn.q_proj
                model.base_model.model.model.layers[i].self_attn.q_proj = \
                    LoRA_MOE_QK(args=training_args,
                        lora_rank=training_args.lora_r,
                        lora_alpha=training_args.lora_alpha,
                        num_experts=1,
                        original_module=original_q,
                        dense_moe=training_args.dense_moe)

                original_k = model.base_model.model.model.layers[i].self_attn.k_proj
                model.base_model.model.model.layers[i].self_attn.k_proj = \
                    LoRA_MOE_QK(args=training_args,
                        lora_rank=training_args.lora_r,
                        lora_alpha=training_args.lora_alpha,
                        num_experts=1,
                        original_module=original_k)
        # for (n,p) in model.named_parameters():
        #     if p.requires_grad:
        #         print(n)
            # original_k = model.base_model.model.model.layers[i].self_attn.k_proj
            # model.base_model.model.model.layers[i].self_attn.k_proj = \
            #     LoRA_MOE_QK(args=training_args,
            #         lora_rank=training_args.lora_r,
            #         lora_alpha=training_args.lora_alpha,
            #         num_experts=training_args.expert_num,
            #         original_module=original_k).bfloat16()
            # original_mlp = model.base_model.model.model.layers[i].mlp
            # model.base_model.model.model.layers[i].mlp = \
            #     LoRA_MOE_FFN(args=training_args,
            #         lora_rank=training_args.lora_r,
            #         lora_alpha=training_args.lora_alpha,
            #         num_experts=training_args.expert_num,
            #         original_module=original_mlp).bfloat16()
    
    training_args.find_unused_parameters = False
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module,)
                    # callbacks=[diminishing_ratio_callback])
    if training_args.lora_enable:
        trainer.model.print_trainable_parameters()
    
    
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Dtype: {param.dtype}")

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    # safe_save_model_for_hf_trainer(trainer=trainer,
    #                                output_dir=training_args.output_dir)
    
    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state(
            model.named_parameters(), training_args.lora_bias
        )
        # print(state_dict.keys(), "fiss")
        non_lora_state_dict = get_peft_state_non_lora(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            # if training_args.use_moe:
            #     torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'moe.bin'))
            # else:
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)
        non_lora_state_dict = get_peft_state_non_lora(
            model.named_parameters()
        )

    # with open(os.path.join(training_args.output_dir, "probing_res.json"), "w") as f:
    #     json.dump(attention_head_gradients, f, indent=2)

if __name__ == "__main__":
    train()
