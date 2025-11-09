import os
from typing import List, Union
from dotenv import load_dotenv
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_processing_base import BatchFeature

class DINO:
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
  
  def __init__(self, model_name: str):
    if not self.is_supported(model_name):
      supported_str = "\n  ".join(self.SUPPORTED_MODELS)
      raise ValueError(
        f"Unsupported model: {model_name}\n"
        f"Supported models are:\n  {supported_str}"
      )
    self.model_name = model_name
    self.processor = AutoImageProcessor.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name)
    self.patch_size = self.model.config.patch_size
    if "num_register_tokens" not in self.model.config:
      self.num_register_tokens = 0
    else:
      self.num_register_tokens = self.model.config.num_register_tokens
  
  def get_model_summary(self):
    print(self.model.config)

  def preprocess(self, image: List[Image.Image]):
    inputs = self.processor(images=image, return_tensors="pt") # eg [batch_size, 3, 224, 224]
    return inputs
  
  def forward(self, inputs: Union[List[Image.Image], BatchFeature]) -> torch.Tensor:
    if not isinstance(inputs, BatchFeature):
      inputs = self.processor(images=inputs, return_tensors="pt") # eg [batch_size, 3, 224, 224]
    outputs = self.model(**inputs) # eg [batch_size, 1 + 4 + 256, 384]
    return outputs # eg [batch_size, 1 + 4 + 256, 384]
  
  def predict(self, inputs: Union[List[Image.Image], BatchFeature]) -> torch.Tensor:
    with torch.inference_mode():
      outputs = self.forward(inputs)
      return outputs # eg [batch_size, 1 + 4 + 256, 384]
    
  def get_patch_tokens(self, images: List[Image.Image]) -> torch.Tensor:
    inputs = self.preprocess(images) # eg [batch_size, 3, 224, 224]
    outputs = self.forward(inputs)
    _, _, img_height, img_width = inputs.pixel_values.shape
    num_patches_height, num_patches_width = img_height // self.patch_size, img_width // self.patch_size
    patch_features_flat = outputs.last_hidden_state[:, 1 + self.num_register_tokens:, :] # eg [batch_size, 256, 384]
    patch_features = patch_features_flat.unflatten(1, (num_patches_height, num_patches_width))
    return patch_features # eg [batch_size, 16, 16, 384]
  
  def get_cls_tokens(self, images: List[Image.Image]) -> torch.Tensor:
    inputs = self.preprocess(images) # eg [batch_size, 3, 224, 224]
    outputs = self.forward(inputs)
    cls_token = outputs.last_hidden_state[:, 0, :] # eg [batch_size, 384]
    return cls_token
  
  def get_register_tokens(self, images: List[Image.Image]) -> torch.Tensor:
    if self.num_register_tokens == 0:
      raise ValueError(f"This model {self.model_name} has no register tokens")
    inputs = self.preprocess(images) # eg [batch_size, 3, 224, 224]
    outputs = self.forward(inputs)
    register_tokens = outputs.last_hidden_state[:, 1:1 + self.num_register_tokens, :]
    return register_tokens # eg [batch_size, 4, 384]

if __name__ == "__main__":
  import argparse
  from transformers.image_utils import load_image
  from huggingface_hub import login

  parser = argparse.ArgumentParser(description="Run DINO model inference on an input image")
  parser.add_argument("--model_name", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m", help="Model name or path to use")
  parser.add_argument("--image_url", type=str, default="http://images.cocodataset.org/val2017/000000039769.jpg", help="Image URL to load")
  parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
  args = parser.parse_args()

  load_dotenv()
  login(token=os.getenv("HF_TOKEN"))

  image = load_image(args.image_url)
  images = [image] * args.batch_size

  model = DINO(args.model_name)
  patch_features = model.get_patch_tokens(images)
  cls_token = model.get_cls_tokens(images)
  register_tokens = model.get_register_tokens(images)
  forward_output = model.forward(images)
  
  print(model.get_model_summary())
  print("cls_token.shape:", cls_token.shape)
  print("register_tokens.shape:", register_tokens.shape)
  print("patch_features.shape:", patch_features.shape)
  print("forward_output.last_hidden_state.shape:", forward_output.last_hidden_state.shape)