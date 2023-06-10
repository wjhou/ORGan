import torch.nn as nn
from transformers import AdamW
from transformers.trainer_pt_utils import get_parameter_names
from torch.optim import Adam


def create_optimizer(model, args, fast_lr=1e-4):
    '''
    fast_lr: for newly inited model
    '''
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [
        name for name in decay_parameters if "bias" not in name
    ]

    fast_params = []
    for n, _ in model.named_parameters():
        if not n.startswith("visual_encoder"):
            fast_params.append(n)
    
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if n in decay_parameters and n not in fast_params
            ],
            "weight_decay":
            args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if n not in decay_parameters and n not in fast_params
            ],
            "weight_decay":
            0.0,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if n in fast_params and n in decay_parameters
            ],
            "lr":
            fast_lr,
            "weight_decay":
            args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if n in fast_params and n not in decay_parameters
            ],
            "lr":
            fast_lr,
            "weight_decay":
            0.0,
        },
    ]
    optimizer_kwargs = {
        "lr": args.learning_rate,
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
    }
    optimizer = AdamW(
        optimizer_grouped_parameters,
        **optimizer_kwargs,
    )
    return optimizer
