"""
LoRA Training Script for Stable Diffusion v1.5
Target: Holographic Raincoats (sks style)

"""

import os
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm

from accelerate import Accelerator
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer
import bitsandbytes as bnb


# -------------------------
# Configuration
# -------------------------
CONFIG = {
    "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "output_dir": "lora_weights",
    "data_dir": "processed_data",

    # Precision / hardware
    "mixed_precision": "fp16",

    # LoRA configuration
    "lora_rank": 32,
    "lora_alpha": 32,

    # Training
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_steps": 4000,
    "unet_learning_rate": 1e-4,
    "text_encoder_learning_rate": 1e-5,

    # Reproducibility
    "seed": 42,
}


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Dataset
# -------------------------
class LocalDataset(Dataset):
    """Dataset that reads images from metadata.jsonl and uses a fixed caption."""

    def __init__(self, data_dir: str, tokenizer, resolution: int = 512):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.samples = []

        metadata_path = self.data_dir / "captions.jsonl"
        if not metadata_path.exists():
            if self.data_dir.exists():
                print(f"Warning: captions.jsonl not found in {data_dir}. Check if unzip was successful.")

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        self.samples.append(entry)

        print(f"Loaded {len(self.samples)} samples from metadata.jsonl")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self.data_dir / sample["file_name"]

        # Dynamic caption from metadata
        caption = sample["text"]

        # Load and resize image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)

        # Convert image to tensor and normalize to [-1, 1]
        image = torch.from_numpy(
            np.array(image)
        ).float() / 127.5 - 1.0
        image = image.permute(2, 0, 1)

        # Tokenize caption
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": image,
            "input_ids": tokens.input_ids.squeeze(0),
        }


def main():
    """Main training function."""
    print("=" * 60)
    print("LoRA Training for Stable Diffusion v1.5")
    print("=" * 60)

    # Set seed
    set_seed(CONFIG["seed"])

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=CONFIG["mixed_precision"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    )
    device = accelerator.device
    print(f"Device: {device}")

    # -------------------------
    # Load Model Components
    # -------------------------
    print("Loading model components...")

    tokenizer = CLIPTokenizer.from_pretrained(
        CONFIG["model_id"], subfolder="tokenizer"
    )

    text_encoder = CLIPTextModel.from_pretrained(
        CONFIG["model_id"],
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )

    vae = AutoencoderKL.from_pretrained(
        CONFIG["model_id"],
        subfolder="vae",
        torch_dtype=torch.float16,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        CONFIG["model_id"], subfolder="scheduler"
    )

    # IMPORTANT: Load UNet in float32 for stable LoRA training
    unet = UNet2DConditionModel.from_pretrained(
        CONFIG["model_id"],
        subfolder="unet",
        torch_dtype=torch.float32,
    )

    # Freeze VAE only
    vae.requires_grad_(False)
    vae.eval()

    text_encoder.train()

    # Move models to device
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    # Enable gradient checkpointing for UNet (memory-safe)
    unet.enable_gradient_checkpointing()

    print("Model components loaded!")

    # -------------------------
    # Configure LoRA
    # -------------------------
    print("Configuring LoRA...")

    # UNet LoRA
    unet_lora_config = LoraConfig(
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )

    unet = get_peft_model(unet, unet_lora_config)
    unet.train()

    print("UNet LoRA trainable parameters:")
    unet.print_trainable_parameters()

    # Text Encoder LoRA
    text_lora_config = LoraConfig(
        r=16,  # lower rank is sufficient for CLIP
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
    )

    text_encoder = get_peft_model(text_encoder, text_lora_config)
    text_encoder.train()

    print("Text Encoder LoRA trainable parameters:")
    text_encoder.print_trainable_parameters()

    # -------------------------
    # Prepare Dataset & Optimizer
    # -------------------------
    print("Loading dataset...")

    dataset = LocalDataset(CONFIG["data_dir"], tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    # Optimizer (8-bit AdamW for memory efficiency)
    optimizer = bnb.optim.AdamW8bit(
        [
            {"params": unet.parameters(), "lr": CONFIG["unet_learning_rate"]},
            {"params": text_encoder.parameters(), "lr": CONFIG["text_encoder_learning_rate"]},
        ],
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )

    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=CONFIG["max_steps"],
    )

    # Prepare with accelerator (CRITICAL: include text_encoder)
    unet, text_encoder, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet,
        text_encoder,
        optimizer,
        dataloader,
        lr_scheduler,
    )

    print("Ready for training!")

    # -------------------------
    # Training Loop
    # -------------------------
    print("=" * 60)
    print("Starting Training...")
    print("=" * 60)

    global_step = 0
    progress_bar = tqdm(total=CONFIG["max_steps"], desc="Training")

    unet.train()
    text_encoder.train()

    while global_step < CONFIG["max_steps"]:
        for batch in dataloader:
            with accelerator.accumulate([unet, text_encoder]):

                # Encode images to latents (VAE is frozen, fp16)
                latents = vae.encode(
                    batch["pixel_values"].to(device, dtype=torch.float16)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device,
                ).long()

                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Text embeddings (trainable via LoRA)
                encoder_hidden_states = text_encoder(
                    batch["input_ids"].to(device)
                )[0]

                # Forward pass
                with accelerator.autocast():
                    noise_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                    ).sample

                    loss = torch.nn.functional.mse_loss(noise_pred, noise)

                # Backward
                accelerator.backward(loss)

                # Step optimizer ONLY when gradients are synced
                if accelerator.sync_gradients:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            if global_step >= CONFIG["max_steps"]:
                break

    progress_bar.close()
    print("\nTraining complete!")

    # -------------------------
    # Save LoRA Weights
    # -------------------------
    print("Saving LoRA adapters...")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(CONFIG["output_dir"], exist_ok=True)
        unet.save_pretrained(f"{CONFIG['output_dir']}/unet")
        text_encoder.save_pretrained(f"{CONFIG['output_dir']}/text_encoder")

    accelerator.end_training()
    print("LoRA adapters saved successfully!")
    print(f"Output directory: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()