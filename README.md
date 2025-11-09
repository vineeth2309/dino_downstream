# DINO Downstream

A Python package for using DINOv2 and DINOv3 models as backbones for downstream vision tasks. This project provides an easy-to-use interface for extracting features from images using Meta's DINO vision transformer models.

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

1. **Request Access:**
   - Visit the [DINOv3 model page](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m)
   - Click "Agree and access repository" to request access
   - You'll need to agree to share your contact information per Meta's privacy policy
   - Wait for access approval (usually automatic or quick)

2. **Get Your HuggingFace Token:**
   - Go to [HuggingFace Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - Create a new token with "Read" permissions
   - Copy the token

3. **Set Environment Variable:**
   - Create a `.env` file in the project root:
   ```bash
   HF_TOKEN=your_huggingface_token_here
   ```
   
   Or export it in your shell:
   ```bash
   export HF_TOKEN=your_huggingface_token_here  # Linux/macOS
   $env:HF_TOKEN="your_huggingface_token_here"   # PowerShell
   ```

## Usage

### Running the DINO Backbone

Run the DINO model on an image:

```bash
uv run python -m backbone.dino --model_name <model_name> [options]
```

**Example:**
```bash
uv run python -m backbone.dino --model_name facebook/dinov3-vits16-pretrain-lvd1689m --image_url http://images.cocodataset.org/val2017/000000039769.jpg --batch_size 4
```

**Arguments:**
- `--model_name`: Name of the DINO model to use (see supported models below)
- `--image_url`: URL of the image to process (default: COCO validation image)
- `--batch_size`: Batch size for inference (default: 4)

### Using as a Backbone in Your Code

```python
from backbone.dino import DINO
from PIL import Image

# Initialize the model
model = DINO("facebook/dinov3-vits16-pretrain-lvd1689m")

# Load images
images = [Image.open("path/to/image.jpg")]

# Extract features
patch_features = model.get_patch_tokens(images)  # [batch_size, H, W, dim]
cls_token = model.get_cls_tokens(images)         # [batch_size, dim]
register_tokens = model.get_register_tokens(images)  # [batch_size, 4, dim] (if available)

# Or get full forward output
outputs = model.forward(images)
```

## Supported Models

### DINOv3 Models (with 4 Register Tokens)

All DINOv3 models include 4 register tokens by default.

**Vision Transformer Models:**
- `facebook/dinov3-vits16-pretrain-lvd1689m` - ViT-Small (21M params)
- `facebook/dinov3-vits16plus-pretrain-lvd1689m` - ViT-Small+ (29M params)
- `facebook/dinov3-vitb16-pretrain-lvd1689m` - ViT-Base (86M params)
- `facebook/dinov3-vitl16-pretrain-lvd1689m` - ViT-Large (300M params)
- `facebook/dinov3-vitl16-pretrain-sat493m` - ViT-Large (satellite data, 300M params)
- `facebook/dinov3-vith16plus-pretrain-lvd1689m` - ViT-Huge+ (840M params)
- `facebook/dinov3-vit7b16-pretrain-lvd1689m` - ViT-7B (6.7B params)
- `facebook/dinov3-vit7b16-pretrain-sat493m` - ViT-7B (satellite data, 6.7B params)

**ConvNeXt Models:**
- `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` - ConvNeXt Tiny (29M params)
- `facebook/dinov3-convnext-small-pretrain-lvd1689m` - ConvNeXt Small (50M params)
- `facebook/dinov3-convnext-base-pretrain-lvd1689m` - ConvNeXt Base (89M params)
- `facebook/dinov3-convnext-large-pretrain-lvd1689m` - ConvNeXt Large (198M params)

### DINOv2 Models (No Register Tokens)

- `facebook/dinov2-small` - ViT-Small
- `facebook/dinov2-base` - ViT-Base
- `facebook/dinov2-large` - ViT-Large
- `facebook/dinov2-giant` - ViT-Giant

### DINOv2 Models (with 4 Register Tokens)

- `facebook/dinov2-with-registers-small` - ViT-Small with registers
- `facebook/dinov2-with-registers-base` - ViT-Base with registers
- `facebook/dinov2-with-registers-large` - ViT-Large with registers
- `facebook/dinov2-with-registers-giant` - ViT-Giant with registers

## Features

The DINO backbone provides several feature extraction methods:

- **Patch Tokens**: Dense patch-level features extracted from the image patches
- **CLS Token**: Global image representation from the class token
- **Register Tokens**: Additional tokens available in DINOv3 and DINOv2-with-registers models (4 tokens)

## Project Structure

```
dino_downstream/
├── backbone/
│   ├── __init__.py
│   └── dino.py          # Main DINO backbone implementation
├── main.py
├── pyproject.toml       # Project dependencies
├── uv.lock              # Locked dependencies
└── README.md
```

## Requirements

- Python 3.12.1
- PyTorch 2.4.1
- transformers >= 4.57.1
- HuggingFace account with DINOv3 model access

## License

This project uses DINO models which are subject to the [DINOv3 License](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m).

