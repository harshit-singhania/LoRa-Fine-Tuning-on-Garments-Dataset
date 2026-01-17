"""
EXPERIMENT 2: UNet-Only LoRA (No Text Encoder Training)
Hypothesis: Text Encoder LoRA may not be necessary for texture learning.

Changes from baseline (train.py):
- Text Encoder is FROZEN (no LoRA applied)
- output_dir: lora_weights -> lora_weights_unet_only
- BONUS: Weights & Biases integration
"""

import os
import json
import random
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm

from accelerate import Accelerator
from diffusers import (
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer
import bitsandbytes as bnb

# Weights & Biases for experiment tracking
import wandb


# -------------------------
# EXPERIMENT CONFIG
# -------------------------
CONFIG = {
    "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "output_dir": "lora_weights_unet_only",  # CHANGED
    "data_dir": str(SCRIPT_DIR / "processed_data"),

    "mixed_precision": "fp16",

    "lora_rank": 32,
    "lora_alpha": 32,

    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_steps": 4000,
    "unet_learning_rate": 1e-4,
    # No text_encoder_learning_rate - it's frozen

    "seed": 42,
    
    # W&B Config
    "experiment_name": "lora_unet_only",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LocalDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer, resolution: int = 512):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.samples = []

        metadata_path = self.data_dir / "metadata.jsonl"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self.data_dir / sample["file_name"]
        caption = "a holographic raincoat"

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
        image = image.permute(2, 0, 1)

        tokens = self.tokenizer(
            caption, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        return {"pixel_values": image, "input_ids": tokens.input_ids.squeeze(0)}


def main():
    print("=" * 60)
    print("EXPERIMENT 2: UNet-Only LoRA (Text Encoder Frozen)")
    print("=" * 60)

    set_seed(CONFIG["seed"])
    
    # Initialize Weights & Biases
    wandb.init(
        project="lora-holographic-raincoat",
        name=CONFIG["experiment_name"],
        config={
            "lora_rank": CONFIG["lora_rank"],
            "lora_alpha": CONFIG["lora_alpha"],
            "learning_rate_unet": CONFIG["unet_learning_rate"],
            "learning_rate_text_encoder": 0,  # Frozen
            "max_steps": CONFIG["max_steps"],
            "mixed_precision": CONFIG["mixed_precision"],
            "experiment_type": "unet_only",
            "text_encoder_frozen": True,
        }
    )

    accelerator = Accelerator(
        mixed_precision=CONFIG["mixed_precision"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    )
    device = accelerator.device

    tokenizer = CLIPTokenizer.from_pretrained(CONFIG["model_id"], subfolder="tokenizer")
    
    # Text Encoder - FROZEN (no LoRA)
    text_encoder = CLIPTextModel.from_pretrained(
        CONFIG["model_id"], subfolder="text_encoder", torch_dtype=torch.float16
    )
    text_encoder.requires_grad_(False)  # FROZEN
    text_encoder.eval()
    
    vae = AutoencoderKL.from_pretrained(
        CONFIG["model_id"], subfolder="vae", torch_dtype=torch.float16
    )
    noise_scheduler = DDPMScheduler.from_pretrained(CONFIG["model_id"], subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(
        CONFIG["model_id"], subfolder="unet", torch_dtype=torch.float32
    )

    vae.requires_grad_(False)
    vae.eval()
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    unet.enable_gradient_checkpointing()

    # UNet LoRA ONLY
    unet_lora_config = LoraConfig(
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.0, bias="none",
    )
    unet = get_peft_model(unet, unet_lora_config)
    unet.train()
    print("UNet LoRA (Text Encoder FROZEN):")
    unet.print_trainable_parameters()

    dataset = LocalDataset(CONFIG["data_dir"], tokenizer)
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    # Optimizer for UNet ONLY
    optimizer = bnb.optim.AdamW8bit(
        unet.parameters(),
        lr=CONFIG["unet_learning_rate"],
        weight_decay=1e-2
    )

    lr_scheduler = get_scheduler(
        "cosine", optimizer=optimizer,
        num_warmup_steps=50, num_training_steps=CONFIG["max_steps"],
    )

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    global_step = 0
    progress_bar = tqdm(total=CONFIG["max_steps"], desc="Training (UNet-Only)")

    while global_step < CONFIG["max_steps"]:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                latents = vae.encode(
                    batch["pixel_values"].to(device, dtype=torch.float16)
                ).latent_dist.sample() * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Text encoder is frozen - just forward pass
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]

                with accelerator.autocast():
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                    loss = torch.nn.functional.mse_loss(noise_pred, noise)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log to W&B every 10 steps
            if global_step % 10 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/step": global_step,
                })
            
            if global_step >= CONFIG["max_steps"]:
                break

    progress_bar.close()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(CONFIG["output_dir"], exist_ok=True)
        unet.save_pretrained(f"{CONFIG['output_dir']}/unet")
        # Note: No text_encoder weights saved - it wasn't trained
    accelerator.end_training()
    
    # Finish W&B run
    wandb.finish()
    
    print(f"Saved to: {CONFIG['output_dir']}")
    print("Note: Only UNet weights saved (Text Encoder was frozen)")


if __name__ == "__main__":
    main()
