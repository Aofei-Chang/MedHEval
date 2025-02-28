from transformers import TrainerCallback

    class DiminishingRatioCallback(TrainerCallback):
        def __init__(self, initial_ratio_visual, initial_ratio_attn, final_ratio, num_training_steps):
            self.initial_ratio_visual = initial_ratio_visual
            self.initial_ratio_attn = initial_ratio_attn
            self.final_ratio = final_ratio
            self.num_training_steps = num_training_steps
        
        def on_step_begin(self, args, state, control, **kwargs):
            # Calculate the current step ratio
            progress = state.global_step / self.num_training_steps
            diminishing_ratio_visual = self.initial_ratio_visual + progress * (self.final_ratio - self.initial_ratio_visual)
            diminishing_ratio_attn = self.initial_ratio_attn + progress * (self.final_ratio - self.initial_ratio_attn)
            
            # Update the regularization term ratio in your model
            kwargs['model'].visual_enhance_ratio = diminishing_ratio_visual
            kwargs['model'].visual_enhance_attn = diminishing_ratio_attn
    # Define the total number of training steps
    num_training_steps = len(data_module['train_dataset']) * training_args.num_train_epochs // training_args.per_device_train_batch_size

    # Initialize the custom callback
    diminishing_ratio_callback = DiminishingRatioCallback(
        initial_ratio_visual=training_args.visual_enhance_ratio,    # Start with 0.1
        initial_ratio_attn=training_args.bbox_ratio,    # Start with 0.1
        final_ratio=0.001,     # End with 0.01
        num_training_steps=num_training_steps
        # model.enhance_visual = training_args.visual_focus
        # model.visual_enhance_ratio = training_args.visual_enhance_ratio
        # model.bbox_ratio = training_args.bbox_ratio
    )

    # for i, p in model.model.named_parameters():
    #     if p.requires_grad:
    #         print(i)
    # gradient_records = {}
    # Global dictionary to store gradients
    # attention_head_gradients = {}
    # attention_head_counter = 0
    # def gradient_hook(module, grad_in, grad_out):
    #     if module not in gradient_records:
    #         print(len(grad_out), grad_out[0].size(), "grad_out")
    #         gradient_records[module] = grad_out[0].detach().mean(dim=0)  # Accumulate mean gradient
    #     else:
    #         gradient_records[module] += grad_out[0].detach().mean(dim=0)

    # if training_args.do_attn_probing:
    #     if training_args.bits == 16:
    #         if training_args.bf16:
    #             model.to(torch.bfloat16)
    #         if training_args.fp16:
    #             model.to(torch.float16)

    #     for name, param in model.named_parameters():
    #         if not any(attn_key in name for attn_key in ["q_proj", "k_proj"]):
    #             param.requires_grad = False
    #         else:
    #             param.requires_grad = True
                # print(name)
        # Attach hooks to the attention projections
        # for name, module in model.named_modules():
        #     if "self_attn" in name and any(proj in name for proj in ["q_proj", "k_proj"]):
        #         module.register_full_backward_hook(gradient_hook)