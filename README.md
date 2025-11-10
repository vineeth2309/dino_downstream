# DINO Downstream

A Python package for using DINOv2 and DINOv3 models as backbones for downstream vision tasks with optional LoRA fine-tuning support.

## Installation

### Install uv Package Manager

First, install `uv`, a fast Python package installer and resolver:

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For other installation methods, see the [uv installation documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Setup Project

1. Navigate to the project directory:
```bash
cd dino_downstream
```

2. Install dependencies:
```bash
uv sync
```

This will create a virtual environment and install all required dependencies.

## HuggingFace Model Access

DINOv3 models require access approval from HuggingFace. Follow these steps:

1. Visit the [DINOv3 model page](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m) and request access
2. Create a token at [HuggingFace Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Set environment variable:
   ```bash
   # Create .env file or export:
   HF_TOKEN=your_huggingface_token_here
   ```

## Usage

### Using as a Backbone in Your Code

```python
from backbone.dino import DINO
from PIL import Image

# Initialize the model (defaults to "cuda" if available)
model = DINO("facebook/dinov3-vits16-pretrain-lvd1689m", device="cuda")

# Extract features
images = [Image.open("path/to/image.jpg")]
patch_features = model.get_patch_tokens(images)  # [batch_size, H, W, dim]
cls_token = model.get_cls_tokens(images)         # [batch_size, dim]
register_tokens = model.get_register_tokens(images)  # [batch_size, 4, dim] (DINOv3 only)
```

### LoRA Fine-tuning

LoRA (Low-Rank Adaptation) enables efficient fine-tuning by adding trainable adapters to the model:

```python
from backbone.dino import DINO, LoRAConfig

# Configure LoRA
lora_config = LoRAConfig(
    r=8,                    # LoRA rank (lower = fewer parameters)
    lora_alpha=8,           # Scaling factor
    lora_dropout=0.1,       # Dropout rate
    target_modules="qv"     # Which modules to adapt
)

# Initialize model with LoRA
model = DINO("facebook/dinov3-vits16-pretrain-lvd1689m", 
             device="cuda", 
             lora_config=lora_config)

# Train your model (only LoRA parameters will be trainable)
# ... your training loop ...

# Save LoRA weights
model.save_lora_weights("path/to/lora_weights")

# Optionally merge LoRA weights into base model
model.merge_and_unload()
```

**LoRA Target Module Options:**
- `"qv"` - Q and V projections only (original LoRA paper, most conservative)
- `"qkv"` - Q, K, V projections (stable, parameter efficient)
- `"qkv_proj"` - QKV + output projection
- `"mlp"` - MLP layers only
- `"all"` - All attention and MLP layers

### Command Line Usage

**Basic inference:**
```bash
uv run python -m backbone.dino \
  --model_name facebook/dinov3-vits16-pretrain-lvd1689m \
  --image_url http://images.cocodataset.org/val2017/000000039769.jpg \
  --batch_size 4 \
  --device cuda
```

**With LoRA:**
```bash
uv run python -m backbone.dino \
  --model_name facebook/dinov3-vits16-pretrain-lvd1689m \
  --use_lora True \
  --lora_r 8 \
  --lora_alpha 8 \
  --lora_dropout 0.1 \
  --lora_target qv
```

**Arguments:**
- `--model_name`: Name of the DINO model to use (see supported models below)
- `--image_url`: URL of the image to process (default: COCO validation image)
- `--batch_size`: Batch size for inference (default: 4)
- `--device`: Device to use for inference, either "cuda" or "cpu" (default: "cuda")
- `--use_lora`: Enable LoRA adapters (default: False)
- `--lora_r`: LoRA rank (default: 8)
- `--lora_alpha`: LoRA alpha scaling factor (default: 8)
- `--lora_dropout`: LoRA dropout rate (default: 0.1)
- `--lora_target`: Target modules - "qv", "qkv", "qkv_proj", "mlp", or "all" (default: "all")

## Supported Models

**DINOv3:** All models include 4 register tokens. Available in ViT (Small/Base/Large/Huge/7B) and ConvNeXt (Tiny/Small/Base/Large) variants.

**DINOv2:** Available with or without register tokens in Small/Base/Large/Giant sizes.

Use `DINO.list_supported_models()` to see the full list of supported model names.

## Requirements

- Python 3.12.1
- PyTorch 2.4.1
- transformers >= 4.57.1
- peft (for LoRA support)

