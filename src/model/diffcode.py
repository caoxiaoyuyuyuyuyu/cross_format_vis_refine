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
        llm_hidden_dim = self.base_model.config.text_config.hidden_size
        self.dpa = DifferentialPerceptionAdapter(
            vit_hidden_dim=vit_hidden_dim,
            llm_hidden_dim=llm_hidden_dim,
            hook_layers=hook_layers,
        )

        # Apply LoRA to LLM q/v projections
        self._apply_lora(lora_rank, lora_alpha)

        # Match DPA dtype to base model
        self.dpa = self.dpa.to(torch.bfloat16)

        # Register ViT hooks
        self.register_vit_hooks()

    def _load_base_model(self, model_name: str):
        """Load Qwen2.5-VL model and processor."""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
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

        vit = self.base_model.get_base_model().model.visual

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
        vit = self.base_model.get_base_model().model.visual
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

            features[idx] = feat_3d.detach().clone()

        return features

    def _get_mrope_position_ids(
        self, input_ids: Tensor, image_grid_thw: Optional[Tensor], attention_mask: Tensor,
    ) -> Tensor:
        """Compute 3D m-RoPE position IDs for the original sequence.

        Returns:
            position_ids: (3, B, S) tensor with [temporal, height, width] positions.
            For text tokens all 3 dims are identical (standard 1D RoPE).
            For image tokens dims reflect spatial grid layout.
        """
        base_model = self.base_model.get_base_model()
        B, S = input_ids.shape
        device = input_ids.device

        if image_grid_thw is not None:
            # Build mm_token_type_ids: 0=text, 1=image
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            mm_token_type_ids = torch.zeros(B, S, dtype=torch.int, device=device)
            mm_token_type_ids[input_ids == image_token_id] = 1

            position_ids, rope_deltas = base_model.model.get_rope_index(
                input_ids=input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
        else:
            # Pure text: sequential positions replicated across 3 dims
            position_ids = torch.arange(S, device=device).view(1, 1, -1).expand(3, B, -1)

        return position_ids

    def _prepend_diff_positions(
        self, position_ids: Tensor, n_diff: int, batch_size: int,
    ) -> Tensor:
        """Prepend text-like positions for diff tokens and shift original positions.

        Diff tokens get sequential positions [0..n_diff-1] with all 3 m-RoPE
        axes identical (like text tokens). Original positions are shifted up by n_diff.

        Args:
            position_ids: (3, B, S) original m-RoPE positions
            n_diff: number of diff tokens to prepend
            batch_size: batch size

        Returns:
            (3, B, n_diff + S) combined position IDs
        """
        device = position_ids.device
        # Diff tokens: text-like positions [0..n_diff-1], same on all 3 axes
        diff_pos = torch.arange(n_diff, device=device, dtype=position_ids.dtype)
        diff_pos = diff_pos.view(1, 1, -1).expand(3, batch_size, -1)
        # Shift original positions to make room for diff tokens
        shifted_pos = position_ids + n_diff
        return torch.cat([diff_pos, shifted_pos], dim=2)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor,
        image_grid_thw: Tensor,
        labels: Optional[Tensor] = None,
        target_pixel_values: Optional[Tensor] = None,
        target_grid_thw: Optional[Tensor] = None,
        rendered_pixel_values: Optional[Tensor] = None,
        rendered_grid_thw: Optional[Tensor] = None,
    ):
        """Full forward pass: baseline VL path + DPA augmentation.

        1. Standard VL path: embed tokens + merge image features (same as baseline)
        2. DPA: separate ViT passes for target/rendered → diff tokens
        3. Compute m-RoPE 3D position IDs (with diff token positions prepended)
        4. Prepend diff tokens to VL embeddings → LLM forward with position_ids
        """
        base_model = self.base_model.get_base_model()

        # Step 1: Standard VL embedding path (image tokens merged into text)
        inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)

        if pixel_values is not None:
            # Mirror Qwen2_5_VLModel.forward() image embedding logic exactly
            image_embeds = base_model.model.get_image_features(pixel_values, image_grid_thw).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = base_model.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # Step 1b: Compute m-RoPE 3D position IDs for the original sequence
        position_ids = self._get_mrope_position_ids(input_ids, image_grid_thw, attention_mask)

        # Step 2: DPA diff tokens (separate ViT passes, pre-spatial-merge hooks)
        if target_pixel_values is not None and rendered_pixel_values is not None:
            target_features = self._encode_image(target_pixel_values, target_grid_thw)
            rendered_features = self._encode_image(rendered_pixel_values, rendered_grid_thw)
            diff_tokens = self.dpa(target_features, rendered_features)

            # Step 3: Prepend diff tokens to VL embeddings
            B, S_diff, _ = diff_tokens.shape
            combined_embeds = torch.cat([diff_tokens, inputs_embeds], dim=1)

            diff_attn = torch.ones(B, S_diff, device=attention_mask.device, dtype=attention_mask.dtype)
            combined_attn = torch.cat([diff_attn, attention_mask], dim=1)

            # Prepend diff token positions to m-RoPE position IDs
            combined_position_ids = self._prepend_diff_positions(position_ids, S_diff, B)

            combined_labels = None
            if labels is not None:
                diff_labels = torch.full((B, S_diff), -100, device=labels.device, dtype=labels.dtype)
                combined_labels = torch.cat([diff_labels, labels], dim=1)
        else:
            # Fallback: no DPA inputs, behave like baseline
            combined_embeds = inputs_embeds
            combined_attn = attention_mask
            combined_position_ids = position_ids
            combined_labels = labels

        # Step 4: LLM forward with explicit m-RoPE position IDs
        outputs = base_model.model.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attn,
            position_ids=combined_position_ids,
        )
        logits = base_model.lm_head(outputs[0])

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

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Optional[Tensor] = None,
        image_grid_thw: Optional[Tensor] = None,
        target_pixel_values: Optional[Tensor] = None,
        target_grid_thw: Optional[Tensor] = None,
        rendered_pixel_values: Optional[Tensor] = None,
        rendered_grid_thw: Optional[Tensor] = None,
        max_new_tokens: int = 1024,
        **generate_kwargs,
    ) -> Tensor:
        """Generate refined code with DPA-augmented inference.

        Mirrors forward() Steps 1-3 to build combined embeddings with
        proper m-RoPE 3D position IDs, then uses greedy autoregressive
        decoding via the language model.
        """
        base_model = self.base_model.get_base_model()

        # Step 1: VL embeddings (same as forward)
        inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)

        if pixel_values is not None:
            image_embeds = base_model.model.get_image_features(pixel_values, image_grid_thw).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = base_model.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # Step 1b: Compute m-RoPE 3D position IDs
        position_ids = self._get_mrope_position_ids(input_ids, image_grid_thw, attention_mask)

        # Step 2: DPA diff tokens
        if target_pixel_values is not None and rendered_pixel_values is not None:
            target_features = self._encode_image(target_pixel_values, target_grid_thw)
            rendered_features = self._encode_image(rendered_pixel_values, rendered_grid_thw)
            diff_tokens = self.dpa(target_features, rendered_features)

            B, S_diff, _ = diff_tokens.shape
            combined_embeds = torch.cat([diff_tokens, inputs_embeds], dim=1)

            diff_attn = torch.ones(B, S_diff, device=attention_mask.device, dtype=attention_mask.dtype)
            combined_attn = torch.cat([diff_attn, attention_mask], dim=1)

            combined_position_ids = self._prepend_diff_positions(position_ids, S_diff, B)
        else:
            combined_embeds = inputs_embeds
            combined_attn = attention_mask
            combined_position_ids = position_ids

        # Step 3: Autoregressive generation via language model
        language_model = base_model.model.language_model
        lm_head = base_model.lm_head
        eos_token_id = self.processor.tokenizer.eos_token_id
        B = combined_embeds.shape[0]

        generated_ids = []
        past_key_values = None

        # Track next position for decode steps (max position + 1)
        next_pos = combined_position_ids.max(dim=2, keepdim=True).values + 1  # (3, B, 1)

        # Prefill: run the full combined input through the language model
        outputs = language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attn,
            position_ids=combined_position_ids,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_logits = lm_head(outputs.last_hidden_state[:, -1:, :])
        next_token = next_logits.argmax(dim=-1)  # (B, 1)
        generated_ids.append(next_token)

        # Decode loop
        for _ in range(max_new_tokens - 1):
            if (next_token == eos_token_id).all():
                break

            token_embeds = language_model.embed_tokens(next_token)
            step_attn = torch.ones(
                B, 1, device=combined_attn.device, dtype=combined_attn.dtype,
            )
            combined_attn = torch.cat([combined_attn, step_attn], dim=1)

            # Text-like position: all 3 axes identical, incrementing
            step_pos = next_pos  # (3, B, 1)
            next_pos = next_pos + 1

            outputs = language_model(
                inputs_embeds=token_embeds,
                attention_mask=combined_attn,
                position_ids=step_pos,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = lm_head(outputs.last_hidden_state[:, -1:, :])
            next_token = next_logits.argmax(dim=-1)
            generated_ids.append(next_token)

        return torch.cat(generated_ids, dim=-1)  # (B, gen_len)

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
