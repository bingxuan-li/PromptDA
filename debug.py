import os, cv2, torch, wandb, numpy as np, torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from promptda.promptda import PromptDA
from promptda.utils.logger import Log
from dataset import SimulatedDataset, RealDataset, Augmentation
import time
from promptda.model.resnet import ResNetEncoder

def count_parameters_in_millions(model):
    """Returns the number of trainable parameters in a PyTorch model, in millions."""
    num_params = sum(p.numel() for p in model.parameters())
    return num_params/1e6

def count_model_params(model):
    total_params = count_parameters_in_millions(model)
    vit_params = count_parameters_in_millions(model.pretrained) + count_parameters_in_millions(model.depth_head)
    encoder_params = count_parameters_in_millions(model.pretrained)
    decoder_params = vit_params - encoder_params
    fusion_params = count_parameters_in_millions(model.depth_head.scratch)
    vit_params -= fusion_params
    depth_encoder_params = count_parameters_in_millions(model.depth_head.scratch.depth_encoder)
    return {'total': total_params, 'vit': vit_params, 'encoder': encoder_params, 'decoder': decoder_params, 'fusion': fusion_params, 'depth_encoder': depth_encoder_params}

model = PromptDA(
    encoder='vitb', output_act='identity', prompt_channels=3, resnet_enabled=True, resnet_blocks_per_stage=3
)
# model = PromptDA(
#     encoder='vitb', output_act='identity', prompt_channels=3
# )
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(DEVICE)
params_counts = count_model_params(model)
Log.info(
    f"Model has {params_counts['total']:.2f}M parameters. "
    f"ViT encoder: {params_counts['vit']:.2f}M, "
    f"Encoder: {params_counts['encoder']:.2f}M, "
    f"Decoder: {params_counts['decoder']:.2f}M, "
    f"Fusion head: {params_counts['fusion']:.2f}M, "
    f"Depth encoder: {params_counts['depth_encoder']:.2f}M."
)

fake_input  = torch.randn(1, 3, 518, 518).to(DEVICE)
fake_prompt = torch.randn(1, 3, 518, 518).to(DEVICE)
output = model.forward(fake_input, fake_prompt)

# fake_input = fake_input.to('cpu')
# vitb
# 1 -> 16.50M
# 2 -> 30.94M
# 3 -> 45.38M
# 4 -> 59.83M
# 5 -> 74.27M
# 6 -> 88.71M
# exp on 1, 3, 6
# depth_encoder = ResNetEncoder(
#     in_channels=3, blocks_per_stage=6, out_channels=[96, 192, 384, 768]
# )
# params_counts = count_parameters_in_millions(depth_encoder)
# Log.info(f"Depth encoder has {params_counts:.2f}M parameters.")
# output = depth_encoder(fake_input)