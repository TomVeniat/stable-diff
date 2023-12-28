import torch
from PIL import Image
from transformers import CLIPTokenizer

import model_loader
import pipeline

device = 'cpu'

ALLOW_CUDA = False
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    device = "cuda"
# if (torch.has_mps or torch.backend.mps.is_available()) and ALLOW_MPS:
#     device = 'mps'

print(f'using device {device}')

tokenizer = CLIPTokenizer('../data/tokenizer_vocab.json', merges_file="../data/tokenizer_merges.txt")

model_file = '../data/v1-5-pruned-emaonly.ckpt'
models = model_loader.preload_models_from_standard_weights(model_file, device)

# TXT2IMG
prompt = "A cat streching on the floor, highly detailed, ultra sharp, cinematic, 8k resolution"
uncond_prompt = ""
do_cfg = True
cfg_scale = 7

# IMG2IMG
input_image = None
image_path = '../images/dina.jpeg'
# input_image = Image.open(image_path)
strength = 0.9

sampler = 'ddpm'
num_inference_steps = 50
seed = 42

output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=device,
    idle_device='cpu',
    tokenizer=tokenizer
)

Image.fromarray(output_image)
