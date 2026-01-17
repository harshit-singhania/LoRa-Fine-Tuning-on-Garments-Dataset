"""
Inference & Evaluation Script for LoRA-finetuned Stable Diffusion v1.5
Includes: LPIPS score, CLIP score, side-by-side comparisons, success/failure cases
"""

import argparse
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import lpips
from transformers import CLIPProcessor, CLIPModel

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()


# -------------------------
# Configuration
# -------------------------
DEFAULT_CONFIG = {
    "base_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "lora_weights_dir": "lora_weights",
    "output_dir": "evaluation_results",
    "num_inference_steps": 28,
    "guidance_scale": 6.0,
}

# Test prompts for evaluation
EVALUATION_PROMPTS = [
    # Success cases - prompts the LoRA model should excel at
    {"prompt": "a holographic raincoat", "category": "success", "description": "Core training concept"},
    {"prompt": "a person wearing a holographic raincoat in the rain", "category": "success", "description": "Training concept with context"},
    {"prompt": "a holographic raincoat on a mannequin", "category": "success", "description": "Training concept, different setting"},
    
    # Failure cases - prompts that may challenge the fine-tuned model
    {"prompt": "a red leather jacket", "category": "failure", "description": "Unrelated clothing item"},
    {"prompt": "a cat sitting on a couch", "category": "failure", "description": "Completely unrelated subject"},
]


def load_base_pipeline(model_id: str, device: str, dtype: torch.dtype):
    """Load the base Stable Diffusion pipeline without LoRA."""
    print(f"Loading base model: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    return pipe


def load_lora_pipeline(model_id: str, lora_dir: str, device: str, dtype: torch.dtype):
    """Load pipeline with LoRA adapters applied."""
    print(f"Loading LoRA-enhanced model from: {lora_dir}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    # Load UNet LoRA
    unet_lora_path = f"{lora_dir}/unet"
    if os.path.exists(unet_lora_path):
        pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_lora_path)
        pipe.unet.eval()
        print("  ✓ UNet LoRA loaded")
    else:
        print(f"  ✗ UNet LoRA not found at {unet_lora_path}")

    # Load Text Encoder LoRA (optional - may not exist for UNet-only experiments)
    text_encoder_lora_path = f"{lora_dir}/text_encoder"
    if os.path.exists(text_encoder_lora_path):
        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, text_encoder_lora_path)
        pipe.text_encoder.eval()
        print("  ✓ Text Encoder LoRA loaded")
    else:
        print("  - Text Encoder LoRA not found (using base text encoder)")

    return pipe


def generate_image(pipe, prompt: str, steps: int, guidance: float, seed: int, device: str):
    """Generate a single image."""
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.autocast(device):
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]
    return image


def compute_lpips_score(img1: Image.Image, img2: Image.Image, lpips_model, device: str) -> float:
    """Compute LPIPS perceptual similarity score between two images."""
    def preprocess(img):
        img = img.resize((512, 512))
        img = np.array(img).astype(np.float32) / 127.5 - 1.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img.to(device)
    
    img1_tensor = preprocess(img1)
    img2_tensor = preprocess(img2)
    
    with torch.no_grad():
        score = lpips_model(img1_tensor, img2_tensor)
    
    return score.item()


def compute_clip_score(image: Image.Image, prompt: str, clip_model, clip_processor, device: str) -> float:
    """Compute CLIP similarity score between image and text prompt."""
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = clip_model(**inputs)
        # Normalize and compute cosine similarity
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        score = (image_embeds @ text_embeds.T).squeeze()
    
    return score.item()


def create_comparison_figure(results: list, output_path: str):
    """Create a side-by-side comparison figure."""
    n_prompts = len(results)
    fig, axes = plt.subplots(n_prompts, 2, figsize=(12, 5 * n_prompts))
    
    if n_prompts == 1:
        axes = [axes]
    
    for i, result in enumerate(results):
        # Base model image
        axes[i][0].imshow(result["base_image"])
        axes[i][0].set_title(f"Base Model\nCLIP: {result['base_clip']:.3f}", fontsize=10)
        axes[i][0].axis("off")
        
        # LoRA model image
        axes[i][1].imshow(result["lora_image"])
        axes[i][1].set_title(f"LoRA Model\nCLIP: {result['lora_clip']:.3f} | LPIPS: {result['lpips']:.3f}", fontsize=10)
        axes[i][1].axis("off")
        
        # Add prompt as row label
        fig.text(0.02, 1 - (i + 0.5) / n_prompts, 
                 f"[{result['category'].upper()}]\n{result['prompt'][:50]}...", 
                 fontsize=9, va='center', ha='left')
    
    plt.tight_layout(rect=[0.15, 0, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison figure saved: {output_path}")


def create_case_study_figure(results: list, category: str, output_path: str):
    """Create a detailed figure for success or failure cases."""
    filtered = [r for r in results if r["category"] == category]
    if not filtered:
        return
    
    n = len(filtered)
    fig, axes = plt.subplots(n, 2, figsize=(14, 6 * n))
    
    if n == 1:
        axes = [axes]
    
    title = "SUCCESS CASES" if category == "success" else "FAILURE CASES"
    fig.suptitle(f"{title}: LoRA vs Base Model Comparison", fontsize=14, fontweight='bold')
    
    for i, result in enumerate(filtered):
        # Base image
        axes[i][0].imshow(result["base_image"])
        base_title = f"Base Model\nCLIP Score: {result['base_clip']:.4f}"
        axes[i][0].set_title(base_title, fontsize=11)
        axes[i][0].axis("off")
        
        # LoRA image
        axes[i][1].imshow(result["lora_image"])
        lora_title = f"LoRA Model\nCLIP Score: {result['lora_clip']:.4f}\nLPIPS (vs Base): {result['lpips']:.4f}"
        axes[i][1].set_title(lora_title, fontsize=11)
        axes[i][1].axis("off")
        
        # Add prompt info
        prompt_text = f"Prompt: \"{result['prompt']}\"\n{result['description']}"
        fig.text(0.5, 1 - (i + 0.05) / n, prompt_text, fontsize=10, ha='center', 
                 style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"{category.capitalize()} cases figure saved: {output_path}")


def print_evaluation_summary(results: list):
    """Print a summary of all evaluation metrics."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    for category in ["success", "failure"]:
        filtered = [r for r in results if r["category"] == category]
        if not filtered:
            continue
        
        print(f"\n{category.upper()} CASES:")
        print("-" * 50)
        
        avg_base_clip = np.mean([r["base_clip"] for r in filtered])
        avg_lora_clip = np.mean([r["lora_clip"] for r in filtered])
        avg_lpips = np.mean([r["lpips"] for r in filtered])
        
        for r in filtered:
            clip_diff = r["lora_clip"] - r["base_clip"]
            better = "✓" if clip_diff > 0 else "✗"
            print(f"  • {r['prompt'][:40]}...")
            print(f"    Base CLIP: {r['base_clip']:.4f} | LoRA CLIP: {r['lora_clip']:.4f} | Δ: {clip_diff:+.4f} {better}")
            print(f"    LPIPS (perceptual diff): {r['lpips']:.4f}")
        
        print(f"\n  Averages for {category} cases:")
        print(f"    Base CLIP: {avg_base_clip:.4f}")
        print(f"    LoRA CLIP: {avg_lora_clip:.4f}")
        print(f"    LPIPS: {avg_lpips:.4f}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA-finetuned Stable Diffusion")
    parser.add_argument("--lora-dir", type=str, default=DEFAULT_CONFIG["lora_weights_dir"])
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--base-model", type=str, default=DEFAULT_CONFIG["base_model"])
    parser.add_argument("--steps", type=int, default=DEFAULT_CONFIG["num_inference_steps"])
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_CONFIG["guidance_scale"])
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}")
    
    # Load models
    print("\n--- Loading Models ---")
    base_pipe = load_base_pipeline(args.base_model, device, dtype)
    
    # Resolve lora_dir relative to script location if not absolute
    lora_dir = args.lora_dir
    if not os.path.isabs(lora_dir):
        lora_dir = str(SCRIPT_DIR / lora_dir)
    lora_pipe = load_lora_pipeline(args.base_model, lora_dir, device, dtype)
    
    # Load evaluation models
    print("\n--- Loading Evaluation Models ---")
    lpips_model = lpips.LPIPS(net='alex').to(device)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Run evaluation
    print("\n--- Running Evaluation ---")
    results = []
    
    for i, test_case in enumerate(EVALUATION_PROMPTS):
        prompt = test_case["prompt"]
        print(f"\n[{i+1}/{len(EVALUATION_PROMPTS)}] Generating: {prompt}")
        
        # Generate images
        base_img = generate_image(base_pipe, prompt, args.steps, args.guidance_scale, args.seed, device)
        lora_img = generate_image(lora_pipe, prompt, args.steps, args.guidance_scale, args.seed, device)
        
        # Compute scores
        lpips_score = compute_lpips_score(base_img, lora_img, lpips_model, device)
        base_clip = compute_clip_score(base_img, prompt, clip_model, clip_processor, device)
        lora_clip = compute_clip_score(lora_img, prompt, clip_model, clip_processor, device)
        
        # Save individual images
        base_img.save(f"{args.output_dir}/base_{i+1}.png")
        lora_img.save(f"{args.output_dir}/lora_{i+1}.png")
        
        results.append({
            "prompt": prompt,
            "category": test_case["category"],
            "description": test_case["description"],
            "base_image": base_img,
            "lora_image": lora_img,
            "lpips": lpips_score,
            "base_clip": base_clip,
            "lora_clip": lora_clip,
        })
        
        print(f"  LPIPS: {lpips_score:.4f} | Base CLIP: {base_clip:.4f} | LoRA CLIP: {lora_clip:.4f}")
    
    # Generate comparison figures
    print("\n--- Creating Visualizations ---")
    create_comparison_figure(results, f"{args.output_dir}/comparison_all.png")
    create_case_study_figure(results, "success", f"{args.output_dir}/success_cases.png")
    create_case_study_figure(results, "failure", f"{args.output_dir}/failure_cases.png")
    
    # Print summary
    print_evaluation_summary(results)
    
    print(f"\nAll results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
