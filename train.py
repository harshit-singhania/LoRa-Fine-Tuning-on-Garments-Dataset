import os
import json
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb


# -------------------------
# Configuration
# -------------------------
CONFIG = {
    "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "data_dir": "data",
    "output_dir": "lora_weights",

    "resolution": 512,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_steps": 4000,

    "unet_learning_rate": 1e-4,
    "mixed_precision": "fp16",

    "lora_rank": 32,
    "lora_alpha": 32,

    "seed": 42,
}


# -------------------------
# Dataset
# -------------------------
class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, tokenizer, resolution):
        self.image_dir = Path(image_dir)
        self.images = list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.jpg"))
        self.tokenizer = tokenizer

        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        caption_path = image_path.with_suffix(".txt")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        caption = caption_path.read_text().strip()

        tokens = self.tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": image,
            "input_ids": tokens.input_ids.squeeze(0),
        }


# -------------------------
# Training
# -------------------------
def main():
    accelerator = Accelerator(
        mixed_precision=CONFIG["mixed_precision"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    )

    torch.manual_seed(CONFIG["seed"])

    # Load components
    tokenizer = CLIPTokenizer.from_pretrained(CONFIG["model_id"], subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(CONFIG["model_id"], subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(CONFIG["model_id"], subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(CONFIG["model_id"], subfolder="unet")

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # Apply LoRA to UNet
    lora_config = LoraConfig(
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )

    unet = get_peft_model(unet, lora_config)
    unet.train()

    # Dataset & Dataloader
    dataset = ImageCaptionDataset(
        CONFIG["data_dir"],
        tokenizer,
        CONFIG["resolution"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=2,
    )

    # Optimizer
    optimizer = bnb.optim.AdamW8bit(
        unet.parameters(),
        lr=CONFIG["unet_learning_rate"],
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        CONFIG["model_id"], subfolder="scheduler"
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=CONFIG["max_steps"],
    )

    (
        unet,
        optimizer,
        dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        unet,
        optimizer,
        dataloader,
        lr_scheduler,
    )

    vae.to(accelerator.device, dtype=torch.float16)
    text_encoder.to(accelerator.device, dtype=torch.float16)

    progress_bar = tqdm(range(CONFIG["max_steps"]), disable=not accelerator.is_local_main_process)
    global_step = 0

    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in range(1000):
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)
                input_ids = batch["input_ids"].to(accelerator.device)

                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(input_ids)[0]

                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                ).sample

                loss = torch.nn.functional.mse_loss(noise_pred, noise)

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            if global_step >= CONFIG["max_steps"]:
                break

        if global_step >= CONFIG["max_steps"]:
            break

    # -------------------------
    # Save LoRA weights
    # -------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(CONFIG["output_dir"], exist_ok=True)
        unet.save_pretrained(CONFIG["output_dir"])

    accelerator.end_training()


if __name__ == "__main__":
    main()