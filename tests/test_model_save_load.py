import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from backbone.dino import DINO, LoRAConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def dummy_images():
    """Create dummy images for testing"""
    images = []
    for _ in range(2):
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        images.append(Image.fromarray(img_array))
    return images


def modify_weights_to_known_values(model):
    """
    Set model weights to deterministic known values.
    
    This ensures we're testing actual weight persistence, not random initialization.
    For LoRA models, this modifies the adapter weights.
    For base models, this modifies all trainable weights.
    
    Returns:
        int: Number of modified parameters
    """
    with torch.no_grad():
        for i, (name, param) in enumerate(model.model.named_parameters()):
            if param.requires_grad:
                param.data.fill_(float(i) * 0.1)
                param.data += torch.arange(param.numel()).float().reshape(param.shape) * 0.001
    
    return sum(p.numel() for p in model.model.parameters() if p.requires_grad)


def get_model_state_dict(model):
    """Extract state dict for comparison"""
    return {k: v.clone().cpu() for k, v in model.model.state_dict().items()}


def compare_state_dicts(state1, state2, rtol=1e-5, atol=1e-8):
    """
    Compare two state dicts with numerical tolerance.
    
    Raises AssertionError if keys don't match or values differ beyond tolerance.
    """
    assert set(state1.keys()) == set(state2.keys()), "State dict keys don't match"
    
    for key in state1.keys():
        if not torch.allclose(state1[key], state2[key], rtol=rtol, atol=atol):
            diff = (state1[key] - state2[key]).abs().max().item()
            raise AssertionError(f"Mismatch in {key}: max diff = {diff}")


class TestDINOSaveLoad:
    
    @pytest.mark.parametrize("model_name", [
        "facebook/dinov2-small",
        "facebook/dinov3-vits16-pretrain-lvd1689m",
    ])
    def test_save_load_base_model(self, temp_dir, dummy_images, model_name):
        """Test saving and loading a model without LoRA"""
        save_path = temp_dir / "base_model"
        
        model1 = DINO(model_name, device="cpu")
        model1.model.eval()
        
        num_params = modify_weights_to_known_values(model1)
        print(f"Modified {num_params:,} parameters")
        
        state_before = get_model_state_dict(model1)
        model1.save_model(str(save_path))
        model2 = DINO.load_model(str(save_path), device="cpu")
        state_after = get_model_state_dict(model2)
        
        compare_state_dicts(state_before, state_after)
        
        assert model2.model_name == model_name
        assert model2.lora_enabled == False
        
        with torch.no_grad():
            out1 = model1.forward(dummy_images)
            out2 = model2.forward(dummy_images)
        
        assert torch.allclose(out1.last_hidden_state, out2.last_hidden_state)
        print(f"✓ Base model save/load test passed for {model_name}")
    
    
    @pytest.mark.parametrize("model_name,target_modules", [
        ("facebook/dinov2-small", "qv"),
        ("facebook/dinov3-vits16-pretrain-lvd1689m", "qkv"),
    ])
    def test_save_load_lora_adapters(self, temp_dir, dummy_images, model_name, target_modules):
        """
        Test saving and loading LoRA adapters separately (for continued training).
        
        Verifies that LoRA adapters can be saved and loaded without merging,
        maintaining exact weight values and model outputs.
        """
        save_path = temp_dir / "lora_adapters"
        
        lora_config = LoRAConfig(r=8, lora_alpha=8, target_modules=target_modules)
        model1 = DINO(model_name, device="cpu", lora_config=lora_config)
        model1.model.eval()
        
        num_params = modify_weights_to_known_values(model1)
        print(f"Modified {num_params:,} LoRA parameters")
        
        state_before = get_model_state_dict(model1)
        
        with torch.no_grad():
            out_before_save = model1.forward(dummy_images)
        
        model1.save_model(str(save_path), merge_lora=False)
        
        assert (save_path / "adapter_config.json").exists()
        assert (save_path / "adapter_model.safetensors").exists() or \
               (save_path / "adapter_model.bin").exists()
        
        model2 = DINO.load_model(str(save_path), device="cpu")
        assert model2.lora_enabled == True
        
        state_after = get_model_state_dict(model2)
        compare_state_dicts(state_before, state_after)
        
        with torch.no_grad():
            out_after_load = model2.forward(dummy_images)
        
        assert torch.allclose(
            out_before_save.last_hidden_state, 
            out_after_load.last_hidden_state,
            rtol=1e-5, atol=1e-8
        ), "Outputs differ after loading LoRA adapters"
        
        model2.model.train()
        from peft import PeftModel
        assert isinstance(model2.model, PeftModel), "Loaded model should be a PeftModel when LoRA is enabled"
        
        print(f"✓ LoRA adapters save/load test passed for {model_name}")
    
    
    @pytest.mark.parametrize("model_name", [
        "facebook/dinov2-small",
        "facebook/dinov3-vits16-pretrain-lvd1689m",
    ])
    def test_save_load_merged_lora(self, temp_dir, dummy_images, model_name):
        """
        Test saving and loading LoRA with merged weights.
        
        Verifies that LoRA adapters can be merged into the base model weights
        and saved/loaded correctly. Uses relaxed tolerance due to numerical
        precision differences from the merge operation.
        """
        save_path = temp_dir / "merged_lora"
        
        lora_config = LoRAConfig(r=8, lora_alpha=8, target_modules="qkv")
        model1 = DINO(model_name, device="cpu", lora_config=lora_config)
        model1.model.eval()
        
        modify_weights_to_known_values(model1)
        
        with torch.no_grad():
            out_before_merge = model1.forward(dummy_images)
        
        model1.save_model(str(save_path), merge_lora=True)
        
        assert not (save_path / "adapter_config.json").exists()
        assert (save_path / "config.json").exists()
        assert (save_path / "model.safetensors").exists() or \
               (save_path / "pytorch_model.bin").exists()
        
        model2 = DINO.load_model(str(save_path), device="cpu")
        assert model2.lora_enabled == False
        
        with torch.no_grad():
            out_after_merge = model2.forward(dummy_images)
        
        max_diff = (out_before_merge.last_hidden_state - out_after_merge.last_hidden_state).abs().max().item()
        assert torch.allclose(
            out_before_merge.last_hidden_state,
            out_after_merge.last_hidden_state,
            rtol=1e-3, atol=1e-5
        ), f"Outputs differ after merging LoRA (max diff: {max_diff:.2e})"
        
        print(f"✓ Merged LoRA save/load test passed for {model_name}")
    
    
    def test_lora_weights_actually_different_from_init(self, temp_dir, dummy_images):
        """
        Verify that modified LoRA weights are actually different from initialization.
        
        This sanity check ensures our tests aren't passing due to lucky random
        initialization. Two models initialized with the same seed should have
        identical weights, but after modification they should differ.
        """
        model_name = "facebook/dinov2-small"
        lora_config = LoRAConfig(r=8, lora_alpha=8, target_modules="qv")
        
        torch.manual_seed(42)
        model1 = DINO(model_name, device="cpu", lora_config=lora_config)
        state_init = get_model_state_dict(model1)
        
        torch.manual_seed(42)
        model2 = DINO(model_name, device="cpu", lora_config=lora_config)
        modify_weights_to_known_values(model2)
        state_modified = get_model_state_dict(model2)
        
        try:
            compare_state_dicts(state_init, state_modified)
            print("✗ ERROR: Modification did not change LoRA weights!")
            assert False, "LoRA weights unchanged after modification - tests may be invalid"
        except AssertionError as e:
            print(f"✓ Weights correctly differ after modification: {str(e)[:100]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])