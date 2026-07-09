import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
from typing import Any, Dict, List, Optional, Tuple, Union
import logging


import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen_modeling

# 禁用transformers加载模型时的详细日志
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.WARNING)

class SketchDecoder(nn.Module):
    """
    Autoregressive generative model wrapper for Qwen2.5-VL
    """

    def __init__(self,
                 pix_len,
                 text_len,
                 model_path="Qwen/Qwen2.5-VL-3B-Instruct",
                 use_gradient_checkpointing=False,
                 device_map=None,  # 新增参数：允许从外部控制device_map
                 vocab_size=None,
                 **kwargs):
        super().__init__()
        
        self.pix_len = pix_len
        self.text_len = text_len
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        self.vocab_size = vocab_size or 197000
        self.bos_token_id = 196998
        self.eos_token_id = 196999
        self.pad_token_id = 151643
        
        print(f"Loading model from {model_path}...")
        
        # 加载配置
        config = AutoConfig.from_pretrained(
            model_path,
            vocab_size=self.vocab_size,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            trust_remote_code=True
        )

        # 分布式训练时不使用device_map，让Accelerate管理设备
        # 单GPU推理时可以使用device_map="auto"
        load_kwargs = {
            "config": config,
            "torch_dtype": torch.bfloat16,
            "ignore_mismatched_sizes": True
        }
        
        # 只在非None时添加device_map（分布式训练时应为None）
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        
        self.transformer = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            **load_kwargs
        )

        self.transformer.resize_token_embeddings(self.vocab_size)
        
        # Enable gradient checkpointing if requested
        if self.use_gradient_checkpointing:
            print("Enabling gradient checkpointing to save memory...")
            self.transformer.gradient_checkpointing_enable()
            # Disable cache when using gradient checkpointing
            if hasattr(self.transformer.config, 'use_cache'):
                self.transformer.config.use_cache = False
        
        self.train()

    def load_state_dict_flexible(self, state_dict, strict=False):
        """Load state_dict with automatic handling of embedding size mismatches.

        When the current model has a larger vocab (e.g. 197004 for skeleton CoT)
        than the checkpoint (e.g. 197000), the extra embedding rows are left at
        their randomly initialised values while all existing weights are copied.
        """
        model_state = self.state_dict()
        filtered = {}
        resized_keys = []
        for key, ckpt_tensor in state_dict.items():
            if key in model_state and ckpt_tensor.shape != model_state[key].shape:
                model_shape = model_state[key].shape
                ckpt_shape = ckpt_tensor.shape
                if len(model_shape) == 2 and len(ckpt_shape) == 2 and model_shape[1] == ckpt_shape[1]:
                    new_tensor = model_state[key].clone()
                    min_rows = min(model_shape[0], ckpt_shape[0])
                    new_tensor[:min_rows] = ckpt_tensor[:min_rows]
                    filtered[key] = new_tensor
                    resized_keys.append(
                        f"  {key}: {list(ckpt_shape)} -> {list(model_shape)} "
                        f"(copied {min_rows}/{model_shape[0]} rows)"
                    )
                    continue
            filtered[key] = ckpt_tensor
        if resized_keys:
            print(f"Embedding resize during checkpoint loading:")
            for msg in resized_keys:
                print(msg)
        result = super().load_state_dict(filtered, strict=strict)

        # Re-tie lm_head to embed_tokens when checkpoint omits lm_head.weight
        # (safetensors with tie_word_embeddings=True only stores one copy).
        lm_head_key = "transformer.lm_head.weight"
        embed_key = "transformer.model.embed_tokens.weight"
        if lm_head_key in result.missing_keys and embed_key not in result.missing_keys:
            self.transformer.lm_head.weight = self.transformer.model.embed_tokens.weight
            result.missing_keys.remove(lm_head_key)
            print("Re-tied lm_head.weight to embed_tokens.weight after checkpoint load.")

        return result

    def forward(self, 
                    input_ids=None,
                    attention_mask=None,
                    pixel_values=None,
                    image_grid_thw=None,
                    labels=None,
                    past_key_values=None,
                    use_cache=False,
                    **kwargs):
            
            target_device = self.transformer.device 
            
            if input_ids is not None:
                input_ids = input_ids.to(target_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(target_device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(target_device)
                if self.transformer.dtype != pixel_values.dtype:
                    pixel_values = pixel_values.to(self.transformer.dtype)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(target_device)
            if labels is not None:
                labels = labels.to(target_device)
            
            self.transformer.rope_deltas = None
            position_ids, _ = self.transformer.get_rope_index(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw
            )
            position_ids = position_ids * attention_mask[None, ]

            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
                labels=labels,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=True
            )
            return Qwen2_5_VLCausalLMOutputWithPast(
                loss=outputs.loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=getattr(outputs, 'rope_deltas', None)
            )
        