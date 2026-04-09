"""Verify Qwen2.5-VL ViT hook output shapes.

Loads the model on CPU (float16 to save RAM), registers hooks on
full-attention layers 7/23/31, runs a dummy image through the ViT,
and prints the actual tensor shapes at each hook point.

This confirms whether hook output is 2D (total_patches, D) or 3D (B, S, D).
"""

import sys
import torch


def main():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"Loading {model_name} on CPU (float16)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_name)

    vit = model.visual
    print(f"\nViT type: {type(vit).__name__}")
    print(f"Num blocks: {len(vit.blocks)}")

    # Print ViT config
    vc = model.config.vision_config
    print(f"Hidden size: {vc.hidden_size}")
    print(f"Spatial merge size: {getattr(vc, 'spatial_merge_size', 'N/A')}")

    # Register hooks
    hooks_output = {}
    handles = []
    for layer_idx in [7, 23, 31]:
        block = vit.blocks[layer_idx]

        def make_hook(idx):
            def hook_fn(module, inp, out):
                if isinstance(out, torch.Tensor):
                    hooks_output[idx] = {"type": "Tensor", "shape": tuple(out.shape), "dtype": str(out.dtype)}
                elif isinstance(out, tuple):
                    hooks_output[idx] = {
                        "type": "tuple",
                        "len": len(out),
                        "shapes": [tuple(o.shape) if isinstance(o, torch.Tensor) else type(o).__name__ for o in out],
                    }
                else:
                    hooks_output[idx] = {"type": type(out).__name__}
            return hook_fn

        handles.append(block.register_forward_hook(make_hook(layer_idx)))

    # Create a test image (solid color 336x336) via processor
    from PIL import Image
    import numpy as np

    test_img = Image.fromarray(np.full((336, 336, 3), 128, dtype=np.uint8))

    messages = [{"role": "user", "content": [{"type": "image", "image": test_img}, {"type": "text", "text": "describe"}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")

    print(f"\npixel_values shape: {inputs['pixel_values'].shape}")
    print(f"image_grid_thw shape: {inputs['image_grid_thw'].shape}")
    print(f"image_grid_thw values: {inputs['image_grid_thw']}")

    # Run ViT forward only
    print("\nRunning ViT forward...")
    with torch.no_grad():
        pixel_values = inputs["pixel_values"].to(torch.float16)
        grid_thw = inputs["image_grid_thw"]
        vit_output = vit(pixel_values, grid_thw=grid_thw)

    print(f"\nViT final output shape: {vit_output.shape}")

    print("\n=== Hook Output Shapes ===")
    for layer_idx in [7, 23, 31]:
        info = hooks_output.get(layer_idx, "NOT CAPTURED")
        print(f"Layer {layer_idx}: {info}")

    # Compute expected seq_len from grid_thw
    grid = inputs["image_grid_thw"]
    seq_lens = (grid[:, 0] * grid[:, 1] * grid[:, 2]).tolist()
    print(f"\nExpected per-image seq_lens from grid_thw: {seq_lens}")
    print(f"Total patches: {sum(seq_lens)}")

    # Test batch=2 (same image twice)
    print("\n=== Testing Batch=2 ===")
    hooks_output.clear()
    pv2 = pixel_values.repeat(2, 1)
    gt2 = grid_thw.repeat(2, 1)
    print(f"pixel_values shape: {pv2.shape}")
    print(f"grid_thw shape: {gt2.shape}")
    with torch.no_grad():
        vit_output2 = vit(pv2, grid_thw=gt2)
    print(f"ViT output shape: {vit_output2.shape}")
    for layer_idx in [7, 23, 31]:
        info = hooks_output.get(layer_idx, "NOT CAPTURED")
        print(f"Layer {layer_idx}: {info}")

    # Cleanup
    for h in handles:
        h.remove()

    print("\n=== Conclusion ===")
    sample_info = hooks_output.get(7, {})
    if isinstance(sample_info, dict) and sample_info.get("type") == "Tensor":
        shape = sample_info["shape"]
        if len(shape) == 2:
            print(f"CONFIRMED: Hook output is 2D {shape}. Need reshape to (B, S, D).")
        elif len(shape) == 3:
            print(f"Hook output is already 3D {shape}. No reshape needed.")
        else:
            print(f"Unexpected shape dimensionality: {shape}")


if __name__ == "__main__":
    main()
