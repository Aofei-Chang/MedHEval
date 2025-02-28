import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features
    
    def get_damro_mask_idx(self, attentions, topK=10):
        damro_mask_idx = None
        attention_last_layer = attentions[-1]
        averaged_attention = attention_last_layer.mean(dim=1)
        cls_attention = averaged_attention[:, 0, :]
        visual_tokens_attention = cls_attention[:, 1:]
        top_10_indices = torch.topk(visual_tokens_attention, topK, dim=1).indices
        all_indices = torch.arange(0, 576)
        remaining_indices = torch.tensor([i for i in all_indices if i not in top_10_indices])
        remaining_indices_output = remaining_indices.unsqueeze(0)
        damro_mask_idx = remaining_indices_output
        return damro_mask_idx

    @torch.no_grad()
    def forward(self, images, output_attentions=False):
        damro_mask_idx = None
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True, output_attentions=output_attentions)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
                attentions = image_forward_out.attentions
                if attentions is not None:
                    damro_mask_idx = self.get_damro_mask_idx(attentions)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=output_attentions)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            attentions = image_forward_outs.attentions
            if attentions is not None:
                damro_mask_idx = self.get_damro_mask_idx(attentions)

        return image_features, damro_mask_idx

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
