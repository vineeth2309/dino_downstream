import os
import json
from typing import List, Union, Literal, Optional
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_processing_base import BatchFeature
from peft import LoraConfig, get_peft_model, PeftModel

@dataclass
class LoRAConfig:
  """
  Configuration for LoRA adaptation.
  
  Research-based target module strategies:
  - "qv": Original LoRA paper approach (Q and V only) - most conservative
  - "qkv": Q, K, V projections (stable, parameter efficient)
  - "qkv_proj": QKV + output projection
  - "mlp": Only MLP layers (recent research shows can match attention-only)
  - "all": All layers
  
  Original LoRA paper (Hu et al., 2021) used Wq and Wv for GPT-3.
  """
  r: int = 8  # LoRA rank
  lora_alpha: int = 8  # LoRA scaling factor
  lora_dropout: float = 0.1
  target_modules: Union[List[str], Literal["qkv", "qv", "qkv_proj", "mlp", "all"]] = "qv"
  bias: str = "none"  # Options: "none", "all", "lora_only"
  
  def get_target_modules(self, model_type: str) -> List[str]:
    """
    Get target modules based on configuration and model type.
    Configurations for VIT's:
    - qv: Q and V only (original LoRA paper recommendation)
    - qkv: Q, K, V projections in attention (stable, parameter efficient)
    - qkv_proj: QKV + output projection (attention with output)
    - mlp: Only MLP layers (recent research shows can match/exceed attention-only)
    - all: All layers
    """
    is_dinov2 = "dinov2" in model_type.lower()
    
    if is_dinov2:
      modules_map = {
        "qv": ["attention.attention.query", "attention.attention.value"],
        "qkv": ["attention.attention.query", "attention.attention.key", "attention.attention.value"],
        "qkv_proj": ["attention.attention.query", "attention.attention.key", "attention.attention.value", "attention.output.dense"],
        "mlp": ["mlp.fc1", "mlp.fc2"],
        "all": ["attention.attention.query", "attention.attention.key", "attention.attention.value", "attention.output.dense", "mlp.fc1", "mlp.fc2"]
      }
    else:  # DINOv3
      modules_map = {
        "qv": ["attention.q_proj", "attention.v_proj"],
        "qkv": ["attention.q_proj", "attention.k_proj", "attention.v_proj"],
        "qkv_proj": ["attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.o_proj"],
        "mlp": ["mlp.up_proj", "mlp.down_proj"],
        "all": ["attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.o_proj", "mlp.up_proj", "mlp.down_proj"]
      }
    
    if isinstance(self.target_modules, str):
      return modules_map.get(self.target_modules, modules_map["qv"])
    else:
      return self.target_modules

class DINO(torch.nn.Module):
  # Supported DINOv2 and DINOv3 models # TO DO: Add entire list of models
  SUPPORTED_MODELS = [
    "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "facebook/dinov3-vitl16-pretrain-sat493m",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "facebook/dinov3-convnext-large-pretrain-lvd1689m",
    "facebook/dinov3-vit7b16-pretrain-sat493m",
    "facebook/dinov2-small",
    "facebook/dinov2-base",
    "facebook/dinov2-large",
    "facebook/dinov2-giant",
    "facebook/dinov2-with-registers-small",
    "facebook/dinov2-with-registers-base",
    "facebook/dinov2-with-registers-large",
    "facebook/dinov2-with-registers-giant",
  ]
  
  @classmethod
  def list_supported_models(cls) -> List[str]:
    """Return a list of all supported DINO model names."""
    return cls.SUPPORTED_MODELS.copy()
  
  @classmethod
  def is_supported(cls, model_name: str) -> bool:
    """Check if a model name is in the supported list."""
    return model_name in cls.SUPPORTED_MODELS
  
  def __init__(
    self, 
    model_name: str, 
    device: str = "cuda", 
    lora_config: Optional[LoRAConfig] = None
  ):
    super().__init__()
    if not self.is_supported(model_name):
      supported_str = "\n  ".join(self.SUPPORTED_MODELS)
      raise ValueError(
        f"Unsupported model: {model_name}\n"
        f"Supported models are:\n  {supported_str}"
      )
    self.model_name = model_name
    self.lora_enabled = False

    self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    self.model = AutoModel.from_pretrained(model_name)

    if lora_config:
      peft_config = self._create_peft_config(lora_config)
      self.model = get_peft_model(self.model, peft_config)
      self.lora_enabled = True
      print(f"LoRA enabled with config: {lora_config}")
      self.print_trainable_parameters()

    self.patch_size = self.model.config.patch_size
    if "num_register_tokens" not in self.model.config:
      self.num_register_tokens = 0
    else:
      self.num_register_tokens = self.model.config.num_register_tokens
    
    if device == "cuda" and not torch.cuda.is_available():
      raise ValueError("CUDA is not available")
    
    self.device = torch.device(device)
    self.model.to(self.device)
  
  def list_modules(self):
    """Print all module names in the model"""
    for name, _ in self.model.named_modules():
        print(name)

  def _create_peft_config(self, lora_config: LoRAConfig) -> LoraConfig:
    """Create PEFT LoraConfig from our LoRAConfig"""
    target_modules = lora_config.get_target_modules(self.model_name)
    
    return LoraConfig(
      r=lora_config.r,
      lora_alpha=lora_config.lora_alpha,
      target_modules=target_modules,
      lora_dropout=lora_config.lora_dropout,
      bias=lora_config.bias,
      task_type=None
    )
  
  def print_trainable_parameters(self):
    """Print the number of trainable parameters"""
    if hasattr(self.model, 'print_trainable_parameters'):
      self.model.print_trainable_parameters()
    else:
      trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
      all_params = sum(p.numel() for p in self.model.parameters())
      print(f"trainable params: {trainable_params:,} || all params: {all_params:,} || "
            f"trainable%: {100 * trainable_params / all_params:.2f}")
  
  def save_lora_weights(self, path: str):
    """Save only LoRA adapter weights"""
    if not self.lora_enabled:
      raise ValueError("LoRA is not enabled for this model")
    self.model.save_pretrained(path)
    print(f"LoRA weights saved to {path}")
  
  def merge_and_unload(self):
    """Merge LoRA weights into base model and remove adapters"""
    if not self.lora_enabled:
      raise ValueError("LoRA is not enabled for this model")
    self.model = self.model.merge_and_unload()
    self.lora_enabled = False
    print("LoRA weights merged into base model")

  def get_model_summary(self):
    print(self.model.config)

  def preprocess(self, image: List[Image.Image], device: str = "cuda"):
    inputs = self.processor(images=image, return_tensors="pt", device=device) # eg [batch_size, 3, 224, 224]
    return inputs
  
  def forward(self, inputs: Union[List[Image.Image], BatchFeature]) -> torch.Tensor:
    if not isinstance(inputs, BatchFeature):
      inputs = self.processor(images=inputs, return_tensors="pt", device=self.device) # eg [batch_size, 3, 224, 224]
    outputs = self.model(**inputs) # eg [batch_size, 1 + 4 + 256, 384]
    return outputs # eg [batch_size, 1 + 4 + 256, 384]
  
  def predict(self, inputs: Union[List[Image.Image], BatchFeature]) -> torch.Tensor:
    with torch.inference_mode():
      outputs = self.forward(inputs)
      return outputs # eg [batch_size, 1 + 4 + 256, 384]
    
  def get_patch_tokens(self, images: List[Image.Image]) -> torch.Tensor:
    inputs = self.preprocess(images, device=self.device) # eg [batch_size, 3, 224, 224]
    outputs = self.forward(inputs)
    _, _, img_height, img_width = inputs.pixel_values.shape
    num_patches_height, num_patches_width = img_height // self.patch_size, img_width // self.patch_size
    patch_features_flat = outputs.last_hidden_state[:, 1 + self.num_register_tokens:, :] # eg [batch_size, 256, 384]
    patch_features = patch_features_flat.unflatten(1, (num_patches_height, num_patches_width))
    return patch_features # eg [batch_size, 16, 16, 384]
  
  def get_cls_tokens(self, images: List[Image.Image]) -> torch.Tensor:
    inputs = self.preprocess(images, device=self.device) # eg [batch_size, 3, 224, 224]
    outputs = self.forward(inputs)
    cls_token = outputs.last_hidden_state[:, 0, :] # eg [batch_size, 384]
    return cls_token
  
  def get_register_tokens(self, images: List[Image.Image]) -> torch.Tensor:
    if self.num_register_tokens == 0:
      raise ValueError(f"This model {self.model_name} has no register tokens")
    inputs = self.preprocess(images, device=self.device) # eg [batch_size, 3, 224, 224]
    outputs = self.forward(inputs)
    register_tokens = outputs.last_hidden_state[:, 1:1 + self.num_register_tokens, :]
    return register_tokens # eg [batch_size, 4, 384]

  def save_model(self, path: str, merge_lora: bool = False):
    """
    Save the model and processor to disk.
    
    Args:
      path: Directory to save model
      merge_lora: If True and LoRA is enabled, merge adapters into base model and save as full model.
                  If False (default), save LoRA adapters separately for continued training.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    if self.lora_enabled and merge_lora:
      # Merge LoRA and save as full model
      print("Merging LoRA adapters into base model...")
      model_to_save = self.model.merge_and_unload()
      model_to_save.save_pretrained(path)
      self.model = model_to_save
      self.lora_enabled = False
      lora_state = "merged"
    elif self.lora_enabled:
      # Save LoRA adapters separately (lightweight, can continue training)
      print("Saving LoRA adapters...")
      self.model.save_pretrained(path)  # PEFT automatically saves only adapters
      lora_state = "lora"
    else:
      # Save full model (no LoRA)
      self.model.save_pretrained(path)
      lora_state = "none"
    
    self.processor.save_pretrained(path)
    
    metadata = {
      "model_name": self.model_name,
      "patch_size": self.patch_size,
      "num_register_tokens": self.num_register_tokens,
      "lora_state": lora_state,  # "none", "lora", or "merged"
    }
    with open(path / "dino_metadata.json", "w") as f:
      json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {path}")
  
  @classmethod
  def load_model(cls, path: str, device: str = "cuda"):
    """
    Load a saved model from disk.
    Automatically detects if it contains LoRA adapters, merged weights, or full model.
    Uses the metadata file to determine the model type.
    
    Args:
      path: Directory containing saved model
      device: Device to load model on
    
    Returns:
      DINO instance with loaded weights
    """
    path = Path(path)
    
    if not path.exists():
      raise ValueError(f"Path does not exist: {path}")
    
    # Load metadata
    metadata_path = path / "dino_metadata.json"
    if not metadata_path.exists():
      raise ValueError(f"No DINO metadata found at {path}. Not a valid DINO model save.")
    
    with open(metadata_path, "r") as f:
      metadata = json.load(f)
    
    model_name = metadata["model_name"]
    lora_state = metadata.get("lora_state", "none")
    
    # Create instance without initializing model first
    instance = object.__new__(cls)
    # Initialize the nn.Module parent class first (required before assigning modules)
    super(DINO, instance).__init__()
    
    instance.model_name = model_name
    instance.patch_size = metadata["patch_size"]
    instance.num_register_tokens = metadata["num_register_tokens"]
    instance.device = torch.device(device)
    instance.processor = AutoImageProcessor.from_pretrained(path, use_fast=True)
    
    # Load model based on lora_state
    if lora_state == "lora":
      base_model = AutoModel.from_pretrained(model_name)
      instance.model = PeftModel.from_pretrained(base_model, path)
      instance.lora_enabled = True
      print(f"Loaded model with lora from {path}")
    else:
      instance.model = AutoModel.from_pretrained(path)
      instance.lora_enabled = False
      print(f"Loaded full model from {path}")
    
    instance.model.to(instance.device)
    return instance

if __name__ == "__main__":
  import argparse
  from transformers.image_utils import load_image
  from huggingface_hub import login

  parser = argparse.ArgumentParser(description="Run DINO model inference on an input image")
  parser.add_argument("--model_name", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m", help="Model name or path to use")
  parser.add_argument("--image_url", type=str, default="http://images.cocodataset.org/val2017/000000039769.jpg", help="Image URL to load")
  parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
  parser.add_argument("--device", type=str, default="cuda", help="Device to use")
  # LoRA arguments
  parser.add_argument("--use_lora", type=bool, default=False, help="Enable LoRA adapters")
  parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
  parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha")
  parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
  parser.add_argument("--lora_target", type=str, default="all", 
                     choices=["qkv", "qv", "qkv_proj", "mlp", "all"],
                     help="Which modules to apply LoRA to")
  args = parser.parse_args()

  load_dotenv()
  login(token=os.getenv("HF_TOKEN"))

  image = load_image(args.image_url)
  images = [image] * args.batch_size

  if args.use_lora:
    lora_config = LoRAConfig(
      r=args.lora_r,
      lora_alpha=args.lora_alpha,
      lora_dropout=args.lora_dropout,
      target_modules=args.lora_target
    )
    model = DINO(args.model_name, device=args.device, lora_config=lora_config)
  else:
    model = DINO(args.model_name, device=args.device)

  patch_features = model.get_patch_tokens(images)
  cls_token = model.get_cls_tokens(images)
  if 'dinov3' in args.model_name.lower() or 'register' in args.model_name.lower():
    register_tokens = model.get_register_tokens(images)
  else:
    register_tokens = None
  forward_output = model.forward(images)

  print(model.get_model_summary())
  print("cls_token.shape:", cls_token.shape)
  if register_tokens is not None:
    print("register_tokens.shape:", register_tokens.shape)
  else:
    print("register_tokens: None")
  print("patch_features.shape:", patch_features.shape)
  print("forward_output.last_hidden_state.shape:", forward_output.last_hidden_state.shape)