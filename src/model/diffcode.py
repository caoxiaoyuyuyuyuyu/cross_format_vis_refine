"""DiffCode: Visual-diff-grounded cross-format code refinement model.

Combines Qwen2.5-VL-7B backbone with Differential Perception Adapter (DPA).
DPA extracts multi-scale diff features from ViT full-attention layers,
producing diff tokens that are prepended to the LLM input.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.model.dpa import DifferentialPerceptionAdapter


class DiffCodeModel(nn.Module):
    """Full DiffCode model: Qwen2.5-VL + DPA + LoRA.

    Training flow:
        1. Encode target_image and rendered_image through ViT
        2. Hooks capture intermediate features at full-attention layers
        3. DPA computes diff tokens from hooked features
        4. Diff tokens are prepended to LLM input embeddings
        5. LLM generates refined code
    """

    # Qwen2.5-VL ViT full-attention layer indices (0-indexed, 32 layers total)
    FULL_ATTN_LAYERS = (7, 23, 31)

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        hook_layers: Tuple[int, ...] = (7, 23, 31),
    ):
        super().__init__()
        self.model_name = model_name
        self.hook_layers = hook_layers
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []
        self._hooked_features: Dict[int, Tensor] = {}

        # Load Qwen2.5-VL
        self._load_base_model(model_name)

        # DPA module
        vit_hidden_dim = self.base_model.config.vision_config.hidden_size
        llm_hidden_dim = self.base_model.config.hidden_size
        self.dpa = DifferentialPerceptionAdapter(
            vit_hidden_dim=vit_hidden_dim,
            llm_hidden_dim=llm_hidden_dim,
            hook_layers=hook_layers,
        )

        # Apply LoRA to LLM q/v projections
        self._apply_lora(lora_rank, lora_alpha)

        # Register ViT hooks
        self.register_vit_hooks()

    def _load_base_model(self, model_name: str):
        """Load Qwen2.5-VL model and processor."""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _apply_lora(self, rank: int, alpha: int):
        """Apply LoRA to LLM q_proj and v_proj layers."""
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=[],  # DPA is separate, not inside peft
        )
        self.base_model = get_peft_model(self.base_model, lora_config)

    def register_vit_hooks(self):
        """Register forward hooks on ViT full-attention layers."""
        # Remove old hooks if any
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._hooked_features.clear()

        vit = self.base_model.get_base_model().visual

        for layer_idx in self.hook_layers:
            layer = vit.blocks[layer_idx]

            def make_hook(idx):
                def hook_fn(module, input, output):
                    # output is the hidden states after this transformer block
                    # Shape: (batch * seq_len, hidden_dim) for Qwen2.5-VL
                    self._hooked_features[idx] = output
                return hook_fn

            handle = layer.register_forward_hook(make_hook(layer_idx))
            self._hook_handles.append(handle)

    def _encode_image(self, pixel_values: Tensor, grid_thw: Tensor) -> Dict[int, Tensor]:
        """Run image through ViT and return hooked features.

        Qwen2.5-VL ViT blocks output 2D tensors: (total_patches, hidden_dim)
        where total_patches is the flat concatenation of all images' patches.
        We reshape to 3D (batch, seq_len, hidden_dim) using grid_thw.

        Args:
            pixel_values: preprocessed image tensor
            grid_thw: (num_images, 3) grid temporal-height-width info

        Returns:
            Dict mapping layer_idx -> feature tensor (B, S, D)
        """
        self._hooked_features.clear()

        # Forward through ViT (before spatial merge)
        vit = self.base_model.get_base_model().visual
        vit(pixel_values, grid_thw=grid_thw)

        # Compute per-image patch counts: seq_len_i = T_i * H_i * W_i
        seq_lens = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
        batch_size = grid_thw.shape[0]

        features = {}
        for idx, feat in self._hooked_features.items():
            if feat.dim() == 2:
                # Reshape 2D (total_patches, D) -> 3D (B, S, D)
                if len(set(seq_lens)) == 1:
                    # Uniform image sizes: simple reshape
                    feat_3d = feat.view(batch_size, seq_lens[0], -1)
                else:
                    # Variable sizes: split and pad to max length
                    splits = feat.split(seq_lens, dim=0)
                    max_len = max(seq_lens)
                    padded = []
                    for s in splits:
                        pad_len = max_len - s.shape[0]
                        if pad_len > 0:
                            s = F.pad(s, (0, 0, 0, pad_len))
                        padded.append(s)
                    feat_3d = torch.stack(padded, dim=0)
            else:
                feat_3d = feat

            features[idx] = feat_3d.detach().clone() if not self.training else feat_3d.clone()

        return features

    def forward(
        self,
        target_pixel_values: Tensor,
        target_grid_thw: Tensor,
        rendered_pixel_values: Tensor,
        rendered_grid_thw: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Optional[Tensor] = None,
    ):
        """Full forward pass.

        1. Encode both images through ViT, collecting hook features
        2. DPA computes diff tokens
        3. Prepend diff tokens to LLM input embeddings
        4. LLM forward pass with optional loss computation
        """
        # Step 1: Get ViT features for both images
        target_features = self._encode_image(target_pixel_values, target_grid_thw)
        rendered_features = self._encode_image(rendered_pixel_values, rendered_grid_thw)

        # Step 2: DPA -> diff tokens (B, S_diff, llm_hidden_dim)
        diff_tokens = self.dpa(target_features, rendered_features)

        # Step 3: Get LLM input embeddings and prepend diff tokens
        base_model = self.base_model.get_base_model()
        text_embeds = base_model.model.embed_tokens(input_ids)  # (B, S_text, D)
        combined_embeds = torch.cat([diff_tokens, text_embeds], dim=1)

        # Extend attention mask for diff tokens
        B, S_diff, _ = diff_tokens.shape
        diff_attn = torch.ones(B, S_diff, device=attention_mask.device, dtype=attention_mask.dtype)
        combined_attn = torch.cat([diff_attn, attention_mask], dim=1)

        # Extend labels if provided (mask diff token positions with -100)
        combined_labels = None
        if labels is not None:
            diff_labels = torch.full((B, S_diff), -100, device=labels.device, dtype=labels.dtype)
            combined_labels = torch.cat([diff_labels, labels], dim=1)

        # Step 4: LLM forward
        outputs = base_model.model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attn,
        )
        logits = base_model.lm_head(outputs.last_hidden_state)

        loss = None
        if combined_labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = combined_labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    def get_trainable_param_count(self) -> Dict[str, int]:
        """Return trainable parameter counts by component."""
        dpa_params = self.dpa.get_param_count()
        lora_params = sum(
            p.numel() for n, p in self.base_model.named_parameters()
            if p.requires_grad and "lora" in n.lower()
        )
        return {
            "dpa": dpa_params,
            "lora": lora_params,
            "total": dpa_params + lora_params,
        }
