import torch
from transformers import AutoConfig, StoppingCriteria
import torch.nn.functional as F


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and cfg.model_type != 'llava':
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)



class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False

def get_variable_name(value):  
    for name, val in vars().items():  
        if val is value:  
            return name  
        
def gaussian(x, mu, sigma):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (torch.sqrt(2 * torch.tensor(3.14159265358979323846)) * sigma)
        

def compute_ca_loss(rel_map, masks, choice="box", object_positions=None):
    loss = 0
    object_number = len(masks)
    if object_number == 0:
        return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()


    attn_map = rel_map 

    b = attn_map.shape[0]
    # H, W = 24, 24
    H, W = masks[0].shape
    for obj_idx in range(object_number):
        obj_loss = 0
        mask = masks[obj_idx]
        
           
        ca_map_obj = attn_map.reshape(b, H, W)

        # if choice and choice in ["Scribble", "Point"]:
            
        #     activation_value = (ca_map_obj * gaussian(mask,0,0.1)).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)
        # else:
        activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)

        obj_loss += torch.mean((1 - activation_value) ** 2)
        
        loss += obj_loss

    return loss


def calculate_top_attention_loss_KL(all_attention_maps, bboxes, image_start_index=39, image_length=256):
    # def compute_ca_loss(rel_map, masks, choice=None, object_positions=None):
    # attention_maps: shape [bs, head, len, len]
    object_number = len(bboxes[0]) #batch 1, default: 1
    if object_number == 0:
        return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()
    # entity_indices = []
    if isinstance(bboxes[0][-1], dict):
        # entity_indices = bboxes[0][-1]['entity_indices']
        bboxes[0] = bboxes[0][:-1]
        object_number = len(bboxes[0])
    loss = 0
    masks_batch = bboxes
    obj_idx = 0
    obj_loss = None
    H, W = masks_batch[0][0].shape
    
    num_heads = 32
    # visual_attention_ratios = []
    output_attention_tensor = torch.cat(all_attention_maps, dim=0)
    all_attention_maps = output_attention_tensor
    # for layer in range(all_attention_maps.size(0)):
    #     for head in range(all_attention_maps[layer].size(0)):
    #         visual_attention = torch.sum(all_attention_maps[layer, head, image_start_index + image_length:, image_start_index:image_start_index + image_length])
    #         total_attention = torch.sum(all_attention_maps[layer, head, image_start_index + image_length:, :])
    #         ratio = (visual_attention / total_attention).item()
    #         visual_attention_ratios.append((ratio, layer, head))

    # # Sort heads by visual attention ratio and select top-32
    # top_heads = sorted(visual_attention_ratios, key=lambda x: x[0], reverse=True)[:num_heads]
    # top_heads_indices = [(layer, head) for _, layer, head in top_heads]
    top_heads_indices = [(31, 1), (22, 17), (19, 15), (31, 17), (18, 18), (20, 27), (30, 21), (13, 14), (12, 5), (27, 3), (21, 23), (31, 27), (31, 11), (23, 20), (23, 27), (22, 5), (27, 2), (31, 3), (28, 25), (21, 17), (28, 0), (28, 1), (12, 11), (12, 2), (29, 14), (26, 14), (31, 24), (17, 5), (12, 0), (21, 9), (19, 17), (28, 17)]

    # Calculate KL-divergence matrix for top-heads
    masks = torch.stack([mask.float().cuda() for mask in masks_batch[0]], 0)  # Shape [num_masks, H, W]
    # masks = torch.as_tensor(masks_batch[0], dtype=torch.float32, device='cuda')
    # kl_matrix = torch.zeros(len(top_heads_indices), object_number).cuda()
    kl_matrix = torch.zeros(len(top_heads_indices), masks.shape[0], device='cuda')

    # for i, (layer, head) in enumerate(top_heads_indices):
    #     # Extract attention map for the top head
    #     head_attn_map_image = all_attention_maps[layer, head, image_start_index + image_length:, image_start_index:image_start_index + image_length]
    #     head_attn_map_image = head_attn_map_image.mean(0)  # Average across tokens -> Shape [H, W]
    #     # Compute KL-divergence with all masks
    #     for mask_idx in range(object_number):
    #         attention_flat = F.softmax(head_attn_map_image.flatten(), dim=-1)  # Normalize attention map
    #         mask_flat = F.softmax(masks[mask_idx].flatten(), dim=-1)  # Normalize mask
    #         kl_div = F.kl_div(attention_flat.log(), mask_flat, reduction='batchmean')
    #         kl_matrix[i, mask_idx] = kl_div
    for i, (layer, head) in enumerate(top_heads_indices):
        head_attn_map_image = all_attention_maps[layer, head, image_start_index + image_length:, image_start_index:image_start_index + image_length]
        head_attn_map_image = head_attn_map_image.mean(0).flatten()  # Shape [H*W]

        # Normalize attention map once
        attention_flat = F.softmax(head_attn_map_image, dim=-1).log()  # Use log-softmax directly for KL-Div calculation

        # Normalize all masks in one batch operation
        masks_flat = masks.flatten(1)  # Shape [num_masks, H*W]
        mask_flat = F.softmax(masks_flat, dim=-1)  # Shape [num_masks, H*W]

        # Compute KL-divergence for all masks in parallel
        kl_div = F.kl_div(attention_flat.unsqueeze(0), mask_flat, reduction='none').mean(dim=1)  # Shape [num_masks]
        kl_matrix[i] = kl_div

    # Select the closest mask for each top head
    closest_mask_indices = kl_matrix.argmin(dim=1)  # Shape [num_heads]
    loss = kl_matrix[range(len(top_heads_indices)), closest_mask_indices].sum()

    return loss


def calculate_top_attention_loss(all_attention_maps, bboxes, image_start_index=39, image_length=256):
    # def compute_ca_loss(rel_map, masks, choice=None, object_positions=None):
    # attention_maps: shape [bs, head, len, len]
    object_number = len(bboxes[0]) #batch 1, default: 1
    if object_number == 0:
        return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()
    # entity_indices = []
    if isinstance(bboxes[0][-1], dict):
        # entity_indices = bboxes[0][-1]['entity_indices']
        bboxes[0] = bboxes[0][:-1]
        object_number = len(bboxes[0])
    loss = 0
    masks_batch = bboxes

    # mean_att = torch.cat(all_attention_maps, 0).mean(0)
    # attn_map = mean_att.mean(0)
    # attn_map_image = attn_map[image_start_index + image_length:, image_start_index:image_start_index + image_length]
    # attn_map_image_entities = []
    obj_idx = 0
    obj_loss = None
    H, W = masks_batch[0][0].shape
    
    # Select top-32 heads based on visual attention ratio
    num_heads = 32
    visual_attention_ratios = []

    # Concatenate all attention maps into a single tensor
    output_attention_tensor = torch.cat(all_attention_maps, dim=0)
    all_attention_maps = output_attention_tensor

    # Extract relevant dimensions
    num_layers, num_heads, seq_len, _ = all_attention_maps.shape
    batch_start = image_start_index + image_length

    # Calculate visual attention and total attention in a vectorized manner
    visual_attention = torch.sum(
        all_attention_maps[:, :, batch_start:, image_start_index:image_start_index + image_length], dim=(-1, -2)
    )  # Shape: (num_layers, num_heads)

    total_attention = torch.sum(
        all_attention_maps[:, :, batch_start:, :], dim=(-1, -2)
    )  # Shape: (num_layers, num_heads)

    # Compute visual attention ratios
    visual_attention_ratios = (visual_attention / total_attention).cpu().numpy()  # Convert to numpy for sorting

    # Flatten the ratios along with layer and head indices for sorting
    layer_indices, head_indices = torch.meshgrid(
        torch.arange(num_layers), torch.arange(num_heads), indexing="ij"
    )

    flattened_ratios = [
        (visual_attention_ratios[layer, head], layer, head)
        for layer, head in zip(layer_indices.flatten(), head_indices.flatten())
    ]

    # Sort by the ratio and take the top-k
    top_heads = sorted(flattened_ratios, key=lambda x: x[0], reverse=True)[:num_heads]
    top_heads_indices = [(layer, head) for _, layer, head in top_heads]

    loss = 0
    top_layers, top_heads = zip(*top_heads_indices)
    top_layers = torch.tensor(top_layers)
    top_heads = torch.tensor(top_heads)
    # Initialize a zero matrix for one-hot encoding
    one_hot_matrix = torch.zeros((all_attention_maps.size(0), all_attention_maps[0].size(0))).cuda()
    # Set the positions at (top_layers, top_heads) to 1
    one_hot_matrix[top_layers, top_heads] = 1
    # print(one_hot_matrix.size(), torch.sum(one_hot_matrix), "index sum")
    

    for obj_idx in range(object_number):
        obj_loss = torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()
        mask = masks_batch[0][obj_idx]
        if mask.shape[0] == 0:
            loss += obj_loss
            continue

        # head_attn_map_image = all_attention_maps[top_layers, top_heads, image_start_index + image_length:, image_start_index:image_start_index + image_length]
        # print("image attens", all_attention_maps.size(), one_hot_matrix.unsqueeze(-1).unsqueeze(-1).size())
        head_attn_map_image = (all_attention_maps * one_hot_matrix.unsqueeze(-1).unsqueeze(-1)).mean(0).mean(0)[image_start_index + image_length:, image_start_index:image_start_index + image_length]
        non_image_token_length = head_attn_map_image.shape[-2]
        # print("head image attens", all_attention_maps.size(), head_attn_map_image.size())
        H, W = mask.shape
        ca_map_obj = head_attn_map_image.reshape(non_image_token_length, H, W)
        activation_value = (ca_map_obj * mask).reshape(non_image_token_length, -1).sum(dim=-1) / ca_map_obj.reshape(non_image_token_length, -1).sum(dim=-1)
        obj_loss += torch.mean((1 - activation_value) ** 2)
        loss += obj_loss

    return loss

def calculate_attention_loss(all_attention_maps, bboxes, image_start_index=39, image_length=256):
    # def compute_ca_loss(rel_map, masks, choice=None, object_positions=None):
    # attention_maps: shape [bs, head, len, len]
    object_number = len(bboxes[0]) #batch 1, default: 1
    if object_number == 0:
        return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()
    entity_indices = []
    if isinstance(bboxes[0][-1], dict):
        entity_indices = bboxes[0][-1]['entity_indices']
        bboxes[0] = bboxes[0][:-1]
        object_number = len(bboxes[0])
    loss = 0
    masks_batch = bboxes
    
    # #old code: all attentions with the same weights
    mean_att = torch.cat(all_attention_maps, 0).mean(0)
    attn_map = mean_att.mean(0)
    attn_map_image = attn_map[image_start_index + image_length:, image_start_index:image_start_index + image_length]

    #### Codes for weighted-average attention maps
    # all_attention_maps_tensor = torch.stack(all_attention_maps, dim=0).squeeze(1)  # Shape: (32, num_heads, height, width)

    # # Calculate visual token attention for each head across all layers
    # visual_attention = all_attention_maps_tensor[:, :, image_start_index + image_length:, image_start_index:image_start_index + image_length].sum(dim=(2, 3))
    # total_attention = all_attention_maps_tensor[:, :, image_start_index + image_length:, :].sum(dim=(2, 3))

    # # Avoid division by zero by adding a small epsilon to `total_attention`
    # epsilon = 1e-10
    # coefficients = visual_attention / (total_attention + epsilon)
    # # print(coefficients, "size")
    # # Reshape coefficients for broadcasting and apply them to each head's attention
    # weighted_attention = all_attention_maps_tensor * coefficients.unsqueeze(-1).unsqueeze(-1)

    # # Calculate the weighted mean across all layers and heads
    # mean_att = weighted_attention.mean(dim=(0, 1))  # Shape: (height, width)
    # attn_map_image = mean_att[image_start_index + image_length:, image_start_index:image_start_index + image_length]


    
    # attn_map_image_entities = []
    obj_idx = 0
    obj_loss = None
    H, W = masks_batch[0][0].shape
    # if len(entity_indices) > 0:
    #     for entity_index in entity_indices:
    #         # print("use entities")
    #         # attn_map_image_entity = attn_map[entity_index[0]:entity_index[1], image_start_index:image_start_index + image_length]
    #         attn_map_image_entity = attn_map[entity_index[0]:, image_start_index:image_start_index + image_length]
    #         # attn_map_image_entities.append(attn_map_image_entity)
    #         # print(f"size of attention map: {attn_map_image_entity.size()}")
    #         non_image_token_length = attn_map_image_entity.shape[0]
    #         obj_loss = torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()
    #         mask = masks_batch[0][obj_idx]
    #         ca_map_obj = attn_map_image_entity.reshape(non_image_token_length, H, W)
    #         # print(obj_idx, mask.shape)
    #         activation_value = (ca_map_obj * mask).reshape(non_image_token_length, -1).sum(dim=-1)/ca_map_obj.reshape(non_image_token_length, -1).sum(dim=-1)
    #         obj_loss += torch.mean((1 - activation_value) ** 2)
    #         obj_idx += 1 
    #         loss += 6 * obj_loss
            # print(loss, "ksiss")
    # else: #do not use else, we alos need to consider the whole attention maps
        # b = attn_map.shape[0]
    # else:

    #### old code, all attention
    non_image_token_length = attn_map_image.shape[0]
    # H, W = 24, 24
    H, W = masks_batch[0][0].shape
    for obj_idx in range(object_number):
        # if obj_loss is not None:
        obj_loss = torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()
        mask = masks_batch[0][obj_idx]
        if mask.shape[0] == 0:
            loss += obj_loss
            continue
        ca_map_obj = attn_map_image.reshape(non_image_token_length, H, W)
        # print(obj_idx, mask.shape)
        activation_value = (ca_map_obj * mask).reshape(non_image_token_length, -1).sum(dim=-1)/ca_map_obj.reshape(non_image_token_length, -1).sum(dim=-1)
        obj_loss += torch.mean((1 - activation_value) ** 2)
        
        loss += obj_loss
    return loss

def calculate_attention_loss_new(all_attention_maps, bboxes, image_start_index=39, image_length=256, use_KL=False):
    bboxes = bboxes[0]
    weak_object_number = len(bboxes[0]["weak"])
    gt_object_number = len(bboxes[0]["gt"]) #batch 1, default: 1
    if gt_object_number == 0 and weak_object_number == 0:
        return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()
    loss = 0
    weak_masks_batch = bboxes[0]["weak"]
    gt_masks_batch = bboxes[0]["gt"]
    weak_ratio, gt_ratio = 1, 5
    
    # #old code: all attentions with the same weights
    mean_att = torch.cat(all_attention_maps, 0).mean(0)
    attn_map = mean_att.mean(0)
    attn_map_image = attn_map[image_start_index + image_length:, image_start_index:image_start_index + image_length]

    obj_idx = 0
    obj_loss = None
    if weak_object_number == 0:
        H, W = gt_masks_batch[0].shape
    else:
        H, W = weak_masks_batch[0].shape
    non_image_token_length = attn_map_image.shape[0]
    for obj_idx in range(weak_object_number):
        obj_loss = torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()
        mask = weak_masks_batch[obj_idx]
        if mask.shape[0] == 0:
            loss += obj_loss
            continue
        ca_map_obj = attn_map_image.reshape(non_image_token_length, H, W)

        if use_KL:
            ca_map_norm = ca_map_obj.reshape(non_image_token_length, -1)
            ca_map_norm = ca_map_norm / (ca_map_norm.sum(dim=-1, keepdim=True) + 1e-8)  # Add small constant for numerical stability
            mask_norm = mask.reshape(-1) / (mask.sum() + 1e-8)  # Normalize mask
            obj_loss += F.kl_div(ca_map_norm.log(), mask_norm.unsqueeze(0).repeat(non_image_token_length, 1), reduction='batchmean')
        else:
            activation_value = (ca_map_obj * mask).reshape(non_image_token_length, -1).sum(dim=-1)/ca_map_obj.reshape(non_image_token_length, -1).sum(dim=-1)
            obj_loss += torch.mean((1 - activation_value) ** 2)
        
        loss += weak_ratio * obj_loss
    for obj_idx in range(gt_object_number):
        obj_loss = torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()
        mask = gt_masks_batch[obj_idx]
        if mask.shape[0] == 0:
            loss += obj_loss
            continue
        ca_map_obj = attn_map_image.reshape(non_image_token_length, H, W)

        if use_KL:
            ca_map_norm = ca_map_obj.reshape(non_image_token_length, -1)
            ca_map_norm = ca_map_norm / (ca_map_norm.sum(dim=-1, keepdim=True) + 1e-8)  # Add small constant for numerical stability
            mask_norm = mask.reshape(-1) / (mask.sum() + 1e-8)  # Normalize mask
            obj_loss += F.kl_div(ca_map_norm.log(), mask_norm.unsqueeze(0).repeat(non_image_token_length, 1), reduction='batchmean')
        else:
            activation_value = (ca_map_obj * mask).reshape(non_image_token_length, -1).sum(dim=-1)/ca_map_obj.reshape(non_image_token_length, -1).sum(dim=-1)
            obj_loss += torch.mean((1 - activation_value) ** 2)
        
        loss += gt_ratio * obj_loss
    return loss


def calculate_visual_loss(all_attention_maps, image_start_index=39, image_length=256, lambda_reg=0.1, top_heads=False):
    # def compute_ca_loss(rel_map, masks, choice=None, object_positions=None):
    # attention_maps: shape [bs, head, len, len]
    loss = 0
    output_attention_tensor = torch.cat(all_attention_maps, dim=0)
    all_attention_maps = output_attention_tensor
    if top_heads:
        top_heads_indices = [(31, 1), (22, 17), (19, 15), (31, 17), (18, 18), (20, 27), (30, 21), (13, 14), (12, 5), (27, 3), (21, 23), (31, 27), (31, 11), (23, 20), (23, 27), (22, 5), (27, 2), (31, 3), (28, 25), (21, 17), (28, 0), (28, 1), (12, 11), (12, 2), (29, 14), (26, 14), (31, 24), (17, 5), (12, 0), (21, 9), (19, 17), (28, 17)]
        top_layers, top_heads = zip(*top_heads_indices)
        top_layers = torch.tensor(top_layers)
        top_heads = torch.tensor(top_heads)
        # Initialize a zero matrix for one-hot encoding
        one_hot_matrix = torch.zeros((all_attention_maps.size(0), all_attention_maps[0].size(0))).cuda()
        # Set the positions at (top_layers, top_heads) to 1
        one_hot_matrix[top_layers, top_heads] = 1
        mean_att = (all_attention_maps * one_hot_matrix.unsqueeze(-1).unsqueeze(-1)).mean(0).mean(0)
        attn_map = mean_att
    else:
        mean_att = all_attention_maps.mean(0)
        attn_map = mean_att.mean(0)
    attn_map_image = attn_map[image_start_index + image_length:, image_start_index:image_start_index + image_length]
    attn_map_image = torch.mean(attn_map_image)
    # b = attn_map.shape[0]
    # non_image_token_length = attn_map_image.shape[0]

    # image_attns = attn_all_layers_tensor[:, :, img_token_idx:img_token_idx + 256]

    # Attentions on instructions and questions
    attn_before_image_01 = torch.mean(attn_map[image_start_index + image_length:, :3])
    attn_before_image = torch.mean(attn_map[image_start_index + image_length:, 3:image_start_index])
    attn_after_image = torch.mean(attn_map[image_start_index + image_length:, image_start_index + image_length:])
    # attn_after_image = attn_all_layers_tensor[:, :, img_token_idx + 256:]

    # Compute the average attention on different segments
    # avg_image_attn = torch.mean(attn_map_image, dim=-1)
    # avg_before_image_attn = torch.mean(attn_before_image, dim=-1)
    # avg_after_image_attn = torch.mean(attn_after_image, dim=-1)

    # regularization_loss = lambda_reg * (torch.mean(avg_before_image_attn) + torch.mean(avg_after_image_attn) - 2 * torch.mean(avg_image_attn))
    regularization_loss = (-1) * lambda_reg * (attn_map_image / (attn_map_image + attn_before_image_01 + attn_after_image + attn_before_image))

    return regularization_loss


