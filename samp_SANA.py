# run `pip install git+https://github.com/huggingface/diffusers` before use Sana in diffusers
import torch
from diffusers import SanaPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
import os

pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_600M_512px_diffusers",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

pipe.vae.to(torch.bfloat16)
pipe.text_encoder.to(torch.bfloat16)

pipe.scheduler.set_timesteps(40)
original_timesteps = pipe.scheduler.timesteps
original_sigmas = pipe.scheduler.sigmas

pipe.scheduler = FlowMatchEulerDiscreteScheduler()
pipe.scheduler.set_timesteps(40)

pipe.scheduler.timesteps = original_timesteps
pipe.scheduler.sigmas = original_sigmas

prompt = "bagel filled with cream cheese and lox"
for seed in range(10):
    image = pipe(
        prompt=prompt,
        height=512,
        width=512,
        guidance_scale=4.5,
        num_inference_steps=40,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        output_type="pil"
    )[0]

    os.makedirs("SANA_samples", exist_ok=True)
    image[0].save(f"SANA_samples/{seed}.png")