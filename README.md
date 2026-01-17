# Holographic Raincoat LoRA Training & Evaluation

This project implements a complete pipeline for fine-tuning a Stable Diffusion v1.5 model using Low-Rank Adaptation (LoRA) to generate high-quality images of **Holographic Raincoats**.

The project is designed to be easily runnable on platforms like **Kaggle** or **Lightning.ai** Studio.

## 📂 Project Structure

- **`train.py`**: The main training script. Loads the base model (SD v1.5), applies LoRA to the UNet and Text Encoder, and trains on the custom dataset.
- **`inference.py`**: A comprehensive evaluation script. It generates images using both the base model and the fine-tuned LoRA model, computes metrics (LPIPS, CLIP), and creates side-by-side visual comparisons.
- **`processed_data/`**: Directory containing the training dataset (images and `metadata.jsonl`).
- **`lora_weights/`**: Output directory where trained LoRA weights are saved.
- **`evaluation_results/`**: Output directory for generated images and comparison figures.
- **`requirements.txt`**: List of Python dependencies.

## 🚀 Setup & Execution

### 1. Prerequisites
Ensure you have a GPU-enabled environment (e.g., Kaggle Notebook (T4 x2) or Lightning.ai Studio).

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Training
To train the LoRA model on the provided dataset:
```bash
python train.py
```
*Note: This will take approximately 1-2 hours on a T4 GPU for 4000 steps.*

### 4. Inference & Evaluation
To generate images and evaluate the model's performance:
```bash
python inference.py
```
This script will:
- Generate images for a set of "Success" and "Failure" test prompts.
- Compare the LoRA model against the Base SD v1.5 model.
- Calculate **CLIP Scores** (text-image alignment) and **LPIPS Scores** (perceptual difference).
- Save comparison figures to `evaluation_results/`.

You can also run inference with custom parameters:
```bash
python inference.py --prompt "a cybernetic holographic raincoat" --steps 50 --seed 123
```

## 📊 Evaluation Results

The model was evaluated on a set of prompts designed to test its core competency ("holographic raincoat") and its robustness against unrelated concepts.

### Quantitative Metrics (Example Run)

**Success Cases (Target Concept):**
- As expected, the LoRA model produces images that are perceptually significantly different from the base model (High LPIPS ~0.60), reflecting the successful injection of the new "holographic" style.
- CLIP scores are comparable, indicating the model maintains good text alignment while changing the visual style completely.

**Failure Cases (Robustness):**
- For unrelated prompts like *"a cat sitting on a couch"*, the LPIPS score is lower (~0.26), meaning the LoRA model correctly behaves more like the base model and doesn't aggressively over-stylize unrelated subjects.

**Visual Results:**
Check the `evaluation_results/` folder for:
- `success_cases.png`: Side-by-side comparison of holographic raincoats.
- `failure_cases.png`: Comparison on unrelated prompts to check for overfitting.

## 🛠 Model Configuration
- **Base Model**: `stable-diffusion-v1-5`
- **LoRA Rank**: 32 (UNet), 16 (Text Encoder)
- **Optimizer**: AdamW (8-bit)
- **Precision**: Mixed (fp16)
- **Resolution**: 512x512

## 📝 Notes for Kaggle Users
- Upload the `processed_data` folder as a Dataset.
- Clone this repository or copy the scripts script into the working directory.
- `train.py` is configured to look for data in `./processed_data`. You might need to move your data or update the `data_dir` in `config` variable if your paths differ.
