import numpy as np
import torch
from tqdm import tqdm

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8


def rescale(x, range1, range2, clamp=False):
    min1, max1 = range1
    min2, max2 = range2

    x = (x - min1) / (max1 - min1)

    x *= (max2 - min2)
    x += min2

    if clamp:
        x = x.clamp(min2, max2)

    return x


def get_time_embedding(timestep) -> torch.Tensor:
    freqs = torch.pow(1e4, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)

    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]

    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def generate(prompt: str, uncond_prompt: str, input_image=None, strength=0.8, do_cfg=True, cfg_scale=7.5,
             sampler_name='ddpm', n_inference_steps=50, models={}, seed=None, device=None, idle_device=None,
             tokenizer=None):
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("Strength must be between 0 and 1")

        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x

        generator = torch.Generator(device=device)

        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models['clip']
        clip = clip.to(device)

        # convert prompt to tokens
        cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_lenth=77).input_ids
        # (b, seq_len)
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
        # (b, seq_len, dim)
        cond_context = clip(cond_tokens)

        if do_cfg:
            # Conditional free guidance
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_lenth=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)

            context = torch.cat([cond_context, uncond_context])  # (2, 77, 768)
        else:
            context = cond_context  # (1, 77, 768)

        to_idle(clip)

        if sampler_name == 'ddpm':
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sample {sampler_name}")

        latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        if input_image:
            encoder = models['encoder']
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)

            # (H,W,C)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            input_image_tensor = input_image_tensor.unsqueeze(0)  # add batch
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)

        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models['diffusion']
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)

        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)

            model_input = latents  # (b, 4, latent_H, latent_W)

            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)

            # predicted noise by hte UNet
            model_out = diffusion(model_input, latents, time_embedding)

            if do_cfg:
                out_cond, out_uncond = model_out.chunk(2)

                model_out = cfg_scale * (out_cond - out_uncond) + out_uncond

            latents = sampler.step(timestep, latents, model_out)

        to_idle(diffusion)

        decoder = models['decoder']
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        return images[0]
