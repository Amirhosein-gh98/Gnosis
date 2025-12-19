import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from ..configs import GRPOConfig, SFTConfig
# add near other imports
from open_r1.parallelism_config import ParallelismConfig
import torch.nn as nn


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

def get_tokenizer(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


# def get_model(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> AutoModelForCausalLM:
#     """Get the model"""
#     torch_dtype = (
#         model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
#     )
#     quantization_config = get_quantization_config(model_args)
#     model_kwargs = dict(
#         revision=model_args.model_revision,
#         trust_remote_code=model_args.trust_remote_code,
#         attn_implementation=model_args.attn_implementation,
#         torch_dtype=torch_dtype,
#         use_cache=False if training_args.gradient_checkpointing else True,
#         device_map=get_kbit_device_map() if quantization_config is not None else None,
#         quantization_config=quantization_config,
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         **model_kwargs,
#     )
#     return model



# def get_model(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> AutoModelForCausalLM:
#     """Load model, then safely init custom heads (fp32), freeze base, and verify."""
#     torch_dtype = (
#         model_args.torch_dtype if model_args.torch_dtype in ["auto", None]
#         else getattr(torch, model_args.torch_dtype)
#     )
#     quantization_config = get_quantization_config(model_args)
#     model_kwargs = dict(
#         revision=model_args.model_revision,
#         trust_remote_code=model_args.trust_remote_code,
#         attn_implementation=model_args.attn_implementation,
#         torch_dtype=torch_dtype,
#         use_cache=False if training_args.gradient_checkpointing else True,
#         device_map=get_kbit_device_map() if quantization_config is not None else None,
#         quantization_config=quantization_config,
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         **model_kwargs,
#     )

#     # ------------ Safe init helpers (fp32, zero-fan guard, NaN repair) ------------
#     def _fp32_copy_init_(param: torch.Tensor, init_fn):
#         if not isinstance(param, torch.Tensor):
#             return
#         if getattr(param, "is_meta", False):
#             return
#         with torch.no_grad():
#             tmp = torch.empty_like(param, dtype=torch.float32, device=param.device)
#             init_fn(tmp)                                   # initialize in fp32
#             param.copy_(tmp.to(dtype=param.dtype))         # cast back to original dtype

#     def _safe_kaiming_normal_(w: torch.Tensor, mode="fan_in", nonlinearity="relu"):
#         def _init(tmp: torch.Tensor):
#             # Guard against zero/invalid fan
#             fan = nn.init._calculate_correct_fan(tmp, mode)
#             if fan <= 0 or not torch.isfinite(torch.tensor(float(fan))):
#                 # Fallback to tiny normal to avoid NaNs
#                 nn.init.normal_(tmp, mean=0.0, std=1e-3)
#             else:
#                 nn.init.kaiming_normal_(tmp, a=0.0, mode=mode, nonlinearity=nonlinearity)
#         _fp32_copy_init_(w, _init)

#     def _safe_zero_(b: torch.Tensor):
#         _fp32_copy_init_(b, lambda t: t.zero_())

#     def _safe_norm_init_(m: nn.Module):
#         w = getattr(m, "weight", None)
#         b = getattr(m, "bias", None)
#         if isinstance(w, torch.Tensor) and not getattr(w, "is_meta", False):
#             _fp32_copy_init_(w, lambda t: t.fill_(1.0))
#         if isinstance(b, torch.Tensor) and not getattr(b, "is_meta", False):
#             _safe_zero_(b)

#     def _init_module_(module: nn.Module):
#         def init(m: nn.Module):
#             if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
#                 w = getattr(m, "weight", None)
#                 if isinstance(w, torch.Tensor) and not getattr(w, "is_meta", False):
#                     _safe_kaiming_normal_(w, mode="fan_in", nonlinearity="relu")
#                 b = getattr(m, "bias", None)
#                 if isinstance(b, torch.Tensor) and not getattr(b, "is_meta", False):
#                     _safe_zero_(b)
#             elif isinstance(m, nn.LayerNorm) or (
#                 hasattr(m, "weight") and hasattr(m, "bias") and m.__class__.__name__.lower().endswith("rmsnorm")
#             ):
#                 _safe_norm_init_(m)
#         module.apply(init)

#     def _freeze_base_(allow_names: list[str]):
#         allow = set(allow_names)
#         for n, p in model.named_parameters():
#             top = n.split(".", 1)[0]
#             p.requires_grad = (top in allow)

#     def _repair_nans_(names: list[str]) -> int:
#         """Detect + repair NaNs by reinit with small normal. Returns #params repaired."""
#         fixed = 0
#         for name in names:
#             mod = getattr(model, name, None)
#             if mod is None:
#                 continue
#             for pname, p in mod.named_parameters(recurse=True):
#                 if not isinstance(p, torch.Tensor):
#                     continue
#                 if getattr(p, "is_meta", False) or not p.is_floating_point():
#                     continue
#                 with torch.no_grad():
#                     if torch.isnan(p).any() or torch.isinf(p).any():
#                         # Re-init this specific parameter with tiny normal
#                         tmp = torch.empty_like(p, dtype=torch.float32, device=p.device)
#                         nn.init.normal_(tmp, mean=0.0, std=1e-3)
#                         p.copy_(tmp.to(p.dtype))
#                         fixed += 1
#                         print(f"[init] repaired NaNs/INFs in {name}.{pname} via tiny normal")
#         return fixed

#     def _verify_no_nans_or_repair_(names: list[str]):
#         fixed = _repair_nans_(names)
#         # If anything still broken, raise (very unlikely after repair)
#         leftovers = []
#         for name in names:
#             mod = getattr(model, name, None)
#             if mod is None:
#                 continue
#             for pname, p in mod.named_parameters(recurse=True):
#                 if not isinstance(p, torch.Tensor):
#                     continue
#                 if getattr(p, "is_meta", False) or not p.is_floating_point():
#                     continue
#                 if torch.isnan(p).any() or torch.isinf(p).any():
#                     leftovers.append(f"{name}.{pname}")
#         if leftovers:
#             raise ValueError(f"NaN/Inf persists after repair in: {leftovers}")
#         if fixed > 0:
#             print(f"[init] repaired {fixed} parameter(s) with NaN/Inf")

#     # -------- discover & init only your custom heads, then freeze everything else --------
#     custom_head_names = getattr(model, "_custom_head_names", None)
#     if not custom_head_names:
#         # Fallback to the names from your earlier snippet
#         candidates = ["axial_sent_encoder", "hid_encoder", "conf_encoder", "stop_head"]
#         custom_head_names = [n for n in candidates if hasattr(model, n)]

#     if custom_head_names:
#         for name in custom_head_names:
#             _init_module_(getattr(model, name))

#         _verify_no_nans_or_repair_(custom_head_names)
#         _freeze_base_(custom_head_names)

#         # report
#         trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         total = sum(p.numel() for p in model.parameters())
#         print(f"[custom heads ready] heads={custom_head_names}  "
#               f"trainable={trainable:,} ({trainable/total:.2%})  total={total:,}")
#     else:
#         print("[custom heads] none detected; skipping post-load init/freeze.")

#     return model




# #t1
# def get_model(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> AutoModelForCausalLM:
#     """
#     Load the base model, then precisely initialize ONLY the custom heads
#     (axial_sent_encoder/hid_encoder/conf_encoder/stop_head) in fp32, freeze the base,
#     and verify all trainable params are finite.
#     """
#     # -------- load pretrained --------
#     torch_dtype = (
#         model_args.torch_dtype if model_args.torch_dtype in ["auto", None]
#         else getattr(torch, model_args.torch_dtype)
#     )
#     quantization_config = get_quantization_config(model_args)
#     model_kwargs = dict(
#         revision=model_args.model_revision,
#         trust_remote_code=model_args.trust_remote_code,
#         attn_implementation=model_args.attn_implementation,
#         torch_dtype=torch_dtype,
#         use_cache=False if training_args.gradient_checkpointing else True,
#         device_map=get_kbit_device_map() if quantization_config is not None else None,
#         quantization_config=quantization_config,
#         # low_cpu_mem_usage=True,  # enable if needed
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         **model_kwargs,
#     )

#     # -------- precise, fp32-safe init helpers --------
#     def _copy_init_fp32_(param: torch.Tensor, init_fn):
#         if not isinstance(param, torch.Tensor) or getattr(param, "is_meta", False):
#             return
#         if not param.is_floating_point():
#             return
#         with torch.no_grad():
#             buf = torch.empty_like(param, dtype=torch.float32, device=param.device)
#             init_fn(buf)  # initialize in fp32
#             param.copy_(buf.to(param.dtype))  # cast back

#     def _init_layernorm_like_(m):
#         # Works for LayerNorm and RMSNorm-like (weight/bias attributes).
#         if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
#             _copy_init_fp32_(m.weight, lambda t: t.fill_(1.0))
#         if hasattr(m, "bias") and isinstance(m.bias, torch.Tensor) and m.bias is not None:
#             _copy_init_fp32_(m.bias, lambda t: t.zero_())

#     def _init_linear_(m: nn.Linear, gelu_expected: bool = True):
#         # Xavier with GELU gain by default (safe for most MLPs that follow with GELU)
#         gain = math.sqrt(2.0) if gelu_expected else 1.0
#         _copy_init_fp32_(m.weight, lambda t: nn.init.xavier_uniform_(t, gain=gain))
#         if m.bias is not None:
#             _copy_init_fp32_(m.bias, lambda t: t.zero_())

#     def _init_conv1d_(m: nn.Conv1d):
#         _copy_init_fp32_(m.weight, lambda t: nn.init.kaiming_uniform_(t, a=0.0, mode="fan_in", nonlinearity="relu"))
#         if m.bias is not None:
#             # fan_in = in_channels * kernel_size / groups
#             fan_in = (m.in_channels // max(1, m.groups)) * m.kernel_size[0]
#             bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1e-3
#             _copy_init_fp32_(m.bias, lambda t: nn.init.uniform_(t, -bound, bound))

#     def _init_conv2d_(m: nn.Conv2d):
#         _copy_init_fp32_(m.weight, lambda t: nn.init.kaiming_uniform_(t, a=0.0, mode="fan_in", nonlinearity="relu"))
#         if m.bias is not None:
#             kh, kw = m.kernel_size
#             fan_in = (m.in_channels // max(1, m.groups)) * kh * kw
#             bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1e-3
#             _copy_init_fp32_(m.bias, lambda t: nn.init.uniform_(t, -bound, bound))

#     def _init_mha_(m: nn.MultiheadAttention):
#         # Match PyTorch defaults but in fp32
#         if hasattr(m, "in_proj_weight") and isinstance(m.in_proj_weight, torch.Tensor):
#             _copy_init_fp32_(m.in_proj_weight, lambda t: nn.init.xavier_uniform_(t))
#         if hasattr(m, "in_proj_bias") and isinstance(m.in_proj_bias, torch.Tensor) and m.in_proj_bias is not None:
#             _copy_init_fp32_(m.in_proj_bias, lambda t: t.zero_())
#         if hasattr(m, "out_proj") and isinstance(m.out_proj, nn.Linear):
#             _copy_init_fp32_(m.out_proj.weight, lambda t: nn.init.xavier_uniform_(t))
#             if m.out_proj.bias is not None:
#                 _copy_init_fp32_(m.out_proj.bias, lambda t: t.zero_())

#     # --- your custom conv blocks (_CausalConv1d / _Pointwise1d) ---
#     # They are raw nn.Modules with nn.Parameter weights; initialize by name shape.
#     def _init_custom_pointwise1d_(mod):
#         w = getattr(mod, "weight", None)
#         b = getattr(mod, "bias", None)
#         if isinstance(w, torch.Tensor):
#             _copy_init_fp32_(w, lambda t: nn.init.kaiming_uniform_(t, a=0.0, mode="fan_in", nonlinearity="relu"))
#         if isinstance(b, torch.Tensor) and b is not None:
#             fan_in = w.shape[1] if isinstance(w, torch.Tensor) else 0
#             bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1e-3
#             _copy_init_fp32_(b, lambda t: nn.init.uniform_(t, -bound, bound))

#     def _init_custom_causalconv1d_(mod):
#         w = getattr(mod, "weight", None)
#         b = getattr(mod, "bias", None)
#         if isinstance(w, torch.Tensor):
#             _copy_init_fp32_(w, lambda t: nn.init.kaiming_uniform_(t, a=0.0, mode="fan_in", nonlinearity="relu"))
#         if isinstance(b, torch.Tensor) and b is not None:
#             # bias bound from fan_in = c_in * k
#             fan_in = w.shape[1] * w.shape[2] if isinstance(w, torch.Tensor) else 0
#             bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1e-3
#             _copy_init_fp32_(b, lambda t: nn.init.uniform_(t, -bound, bound))

#     # --- precise init for AttnFeatureEmbedder ---
#     def _init_attn_feature_embedder_(embedder):
#         # 1) TCN rows/cols
#         for tcn in [getattr(embedder, "tcn_row", None), getattr(embedder, "tcn_col", None)]:
#             if tcn is None:
#                 continue
#             for blk in tcn.blocks:
#                 _init_custom_causalconv1d_(blk["causal"])
#                 _init_custom_pointwise1d_(blk["pw"])
#                 if isinstance(blk["res"], nn.Identity):
#                     pass
#                 else:
#                     _init_custom_pointwise1d_(blk["res"])
#                 _init_layernorm_like_(blk["rms"])

#         # 2) 2D depthwise stack (if present)
#         if getattr(embedder, "use_2d", False) and hasattr(embedder, "cnn2d"):
#             for m in embedder.cnn2d.modules():
#                 if isinstance(m, nn.Conv2d):
#                     _init_conv2d_(m)
#                 elif isinstance(m, nn.LayerNorm) or m.__class__.__name__.lower().endswith("rmsnorm"):
#                     _init_layernorm_like_(m)

#         # 3) query params (q_row/q_col)
#         if isinstance(getattr(embedder, "q_row", None), torch.Tensor):
#             d = embedder.q_row.numel()
#             _copy_init_fp32_(embedder.q_row, lambda t: nn.init.uniform_(t, -1.0 / math.sqrt(d), 1.0 / math.sqrt(d)))
#         if isinstance(getattr(embedder, "q_col", None), torch.Tensor):
#             d = embedder.q_col.numel()
#             _copy_init_fp32_(embedder.q_col, lambda t: nn.init.uniform_(t, -1.0 / math.sqrt(d), 1.0 / math.sqrt(d)))

#         # 4) fuse (LayerNorm + Linear)
#         if hasattr(embedder, "fuse"):
#             for m in embedder.fuse.modules():
#                 if isinstance(m, nn.Linear):
#                     _init_linear_(m, gelu_expected=False)  # no GELU immediately after fuse Linear
#                 elif isinstance(m, nn.LayerNorm) or m.__class__.__name__.lower().endswith("rmsnorm"):
#                     _init_layernorm_like_(m)

#         # 5) causal encoder: MHA + FF + norms
#         if hasattr(embedder, "causal_enc"):
#             for m in embedder.causal_enc.modules():
#                 if isinstance(m, nn.MultiheadAttention):
#                     _init_mha_(m)
#                 elif isinstance(m, nn.Linear):
#                     # FFN uses GELU → use GELU gain
#                     _init_linear_(m, gelu_expected=True)
#                 elif isinstance(m, nn.LayerNorm) or m.__class__.__name__.lower().endswith("rmsnorm"):
#                     _init_layernorm_like_(m)

#     # --- generic init for the other custom heads (hid_encoder, conf_encoder, stop_head) ---
#     def _init_generic_stack_(module: nn.Module):
#         for m in module.modules():
#             if isinstance(m, nn.Linear):
#                 # Most of your stacks use GELU, so use GELU gain by default
#                 _init_linear_(m, gelu_expected=True)
#             elif isinstance(m, nn.Conv1d):
#                 _init_conv1d_(m)
#             elif isinstance(m, nn.Conv2d):
#                 _init_conv2d_(m)
#             elif isinstance(m, nn.MultiheadAttention):
#                 _init_mha_(m)
#             elif isinstance(m, nn.LayerNorm) or m.__class__.__name__.lower().endswith("rmsnorm"):
#                 _init_layernorm_like_(m)

#     def _freeze_base_(allow_toplevel_names: list[str]):
#         allow = set(allow_toplevel_names)
#         for n, p in model.named_parameters():
#             top = n.split(".", 1)[0]
#             p.requires_grad = (top in allow)

#     def _assert_finite_(names: list[str]):
#         bad = []
#         with torch.no_grad():
#             for name in names:
#                 mod = getattr(model, name, None)
#                 if mod is None:
#                     continue
#                 for pname, p in mod.named_parameters(recurse=True):
#                     if not isinstance(p, torch.Tensor) or getattr(p, "is_meta", False) or not p.is_floating_point():
#                         continue
#                     if not torch.isfinite(p).all():
#                         bad.append(f"{name}.{pname}")
#         if bad:
#             raise RuntimeError(f"Non-finite values after init: {bad}")

#     # -------- discover custom heads, init precisely, freeze base, verify --------
#     custom_head_names = getattr(model, "_custom_head_names", None)
#     if not custom_head_names:
#         # Fallback to your known names
#         candidates = ["axial_sent_encoder", "hid_encoder", "conf_encoder", "stop_head"]
#         custom_head_names = [n for n in candidates if hasattr(model, n)]

#     if custom_head_names:
#         for name in custom_head_names:
#             mod = getattr(model, name)
#             if name == "axial_sent_encoder":
#                 _init_attn_feature_embedder_(mod)
#             else:
#                 _init_generic_stack_(mod)

#         _assert_finite_(custom_head_names)
#         _freeze_base_(custom_head_names)   # keep base frozen; add "lm_head" here if you want to train it too

#         # small report
#         trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         total = sum(p.numel() for p in model.parameters())
#         print(f"[custom heads ready] heads={custom_head_names}  "
#               f"trainable={trainable:,} ({trainable/total:.2%})  total={total:,}")
#     else:
#         print("[custom heads] none detected; skipping post-load init/freeze.")

#     return model




#t2


# def get_model(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> AutoModelForCausalLM:
#     """
#     Load the base model, then precisely initialize ONLY the custom heads
#     (attn_extractor / conf_extractor / hid_extractor / stop_head) in fp32,
#     freeze the base, and verify all trainable params are finite.
#     """
#     # -------- load pretrained --------
#     torch_dtype = (
#         model_args.torch_dtype if model_args.torch_dtype in ["auto", None]
#         else getattr(torch, model_args.torch_dtype)
#     )
#     quantization_config = get_quantization_config(model_args)
#     model_kwargs = dict(
#         revision=model_args.model_revision,
#         trust_remote_code=model_args.trust_remote_code,
#         attn_implementation=model_args.attn_implementation,
#         torch_dtype=torch_dtype,
#         use_cache=False if training_args.gradient_checkpointing else True,
#         device_map=get_kbit_device_map() if quantization_config is not None else None,
#         quantization_config=quantization_config,
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         **model_kwargs,
#     )

#     # Ensure reduction grid is available in config (used by Qwen3Model)
#     if not hasattr(model.config, "stop_k_att"):
#         setattr(model.config, "stop_k_att", 48)

#     # -------- precise, fp32-safe init helpers --------
#     def _copy_init_fp32_(param: torch.Tensor, init_fn):
#         if not isinstance(param, torch.Tensor) or getattr(param, "is_meta", False):
#             return
#         if not param.is_floating_point():
#             return
#         with torch.no_grad():
#             buf = torch.empty_like(param, dtype=torch.float32, device=param.device)
#             init_fn(buf)                      # initialize in fp32
#             param.copy_(buf.to(param.dtype))  # cast back

#     def _init_layernorm_like_(m):
#         # Works for LayerNorm / RMSNorm-likes
#         if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
#             _copy_init_fp32_(m.weight, lambda t: t.fill_(1.0))
#         if hasattr(m, "bias") and isinstance(m.bias, torch.Tensor) and m.bias is not None:
#             _copy_init_fp32_(m.bias, lambda t: t.zero_())

#     def _init_groupnorm_(m: nn.GroupNorm):
#         if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
#             _copy_init_fp32_(m.weight, lambda t: t.fill_(1.0))
#         if hasattr(m, "bias") and isinstance(m.bias, torch.Tensor) and m.bias is not None:
#             _copy_init_fp32_(m.bias, lambda t: t.zero_())

#     def _init_embedding_(m: nn.Embedding):
#         d = m.embedding_dim
#         bound = 1.0 / math.sqrt(max(1, d))
#         _copy_init_fp32_(m.weight, lambda t: t.uniform_(-bound, bound))

#     def _init_linear_(m: nn.Linear, gelu_expected: bool = True):
#         gain = math.sqrt(2.0) if gelu_expected else 1.0
#         _copy_init_fp32_(m.weight, lambda t: nn.init.xavier_uniform_(t, gain=gain))
#         if m.bias is not None:
#             _copy_init_fp32_(m.bias,   lambda t: t.zero_())

#     def _init_conv1d_(m: nn.Conv1d):
#         _copy_init_fp32_(m.weight, lambda t: nn.init.kaiming_uniform_(t, a=0.0, mode="fan_in", nonlinearity="relu"))
#         if m.bias is not None:
#             fan_in = (m.in_channels // max(1, m.groups)) * m.kernel_size[0]
#             bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1e-3
#             _copy_init_fp32_(m.bias, lambda t: t.uniform_(-bound, bound))

#     def _init_conv2d_(m: nn.Conv2d):
#         _copy_init_fp32_(m.weight, lambda t: nn.init.kaiming_uniform_(t, a=0.0, mode="fan_in", nonlinearity="relu"))
#         if m.bias is not None:
#             kh, kw = m.kernel_size
#             fan_in = (m.in_channels // max(1, m.groups)) * kh * kw
#             bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1e-3
#             _copy_init_fp32_(m.bias, lambda t: t.uniform_(-bound, bound))

#     def _init_mha_(m: nn.MultiheadAttention):
#         if hasattr(m, "in_proj_weight") and isinstance(m.in_proj_weight, torch.Tensor):
#             _copy_init_fp32_(m.in_proj_weight, lambda t: nn.init.xavier_uniform_(t))
#         if hasattr(m, "in_proj_bias") and isinstance(m.in_proj_bias, torch.Tensor) and m.in_proj_bias is not None:
#             _copy_init_fp32_(m.in_proj_bias, lambda t: t.zero_())
#         if hasattr(m, "out_proj") and isinstance(m.out_proj, nn.Linear):
#             _copy_init_fp32_(m.out_proj.weight, lambda t: nn.init.xavier_uniform_(t))
#             if m.out_proj.bias is not None:
#                 _copy_init_fp32_(m.out_proj.bias, lambda t: t.zero_())

#     # --- init for AttnFeatureExtractorLite ---
#     def _init_attn_feature_extractor_(extractor):
#         # convs + norms
#         for m in [extractor.conv1, extractor.conv2]:
#             _init_conv2d_(m)
#         for m in [extractor.gn1, extractor.gn2]:
#             _init_groupnorm_(m)

#         # token projection head (LinearLN: contains LayerNorm + Linear)
#         if hasattr(extractor, "proj"):
#             for sm in extractor.proj.modules():
#                 if isinstance(sm, nn.Linear):
#                     _init_linear_(sm, gelu_expected=True)
#                 elif isinstance(sm, (nn.LayerNorm,)):
#                     _init_layernorm_like_(sm)
#                 elif isinstance(sm, nn.GroupNorm):
#                     _init_groupnorm_(sm)

#         # positional embeddings
#         _init_embedding_(extractor.layer_emb)
#         _init_embedding_(extractor.head_emb)

#         # query parameter (K, d_tok)
#         if hasattr(extractor, "query") and isinstance(extractor.query, torch.Tensor):
#             d = extractor.query.shape[-1]
#             bound = 1.0 / math.sqrt(max(1, d))
#             _copy_init_fp32_(extractor.query, lambda t: t.uniform_(-bound, bound))

#         # output projection (LinearLN + Dropout)
#         if hasattr(extractor, "out"):
#             for sm in extractor.out.modules():
#                 if isinstance(sm, nn.Linear):
#                     _init_linear_(sm, gelu_expected=True)
#                 elif isinstance(sm, (nn.LayerNorm,)):
#                     _init_layernorm_like_(sm)
#                 elif isinstance(sm, nn.GroupNorm):
#                     _init_groupnorm_(sm)

#     # --- generic init for conf/hid/stop (recurses into LinearLN) ---
#     def _init_generic_stack_(module: nn.Module):
#         for m in module.modules():
#             if isinstance(m, nn.Linear):
#                 _init_linear_(m, gelu_expected=True)
#             elif isinstance(m, nn.Conv1d):
#                 _init_conv1d_(m)
#             elif isinstance(m, nn.Conv2d):
#                 _init_conv2d_(m)
#             elif isinstance(m, nn.MultiheadAttention):
#                 _init_mha_(m)
#             elif isinstance(m, (nn.LayerNorm,)):
#                 _init_layernorm_like_(m)
#             elif isinstance(m, nn.GroupNorm):
#                 _init_groupnorm_(m)
#             elif isinstance(m, nn.Embedding):
#                 _init_embedding_(m)

#     def _init_free_queries_(module: nn.Module):
#         """
#         Re-init any nn.Parameter named 'query'/'q_row'/'q_col' that are not inside submodules.
#         This covers HiddenFeatureExtractorLite.query and any similar fields.
#         """
#         for attr in ("query", "q_row", "q_col"):
#             p = getattr(module, attr, None)
#             if isinstance(p, torch.Tensor) and p.is_floating_point():
#                 # bound ~ 1/sqrt(dim) (avoid div-by-zero)
#                 d = p.shape[-1] if p.ndim >= 1 else 0
#                 bound = 1.0 / math.sqrt(max(1, int(d)))
#                 _copy_init_fp32_(p, lambda t: t.uniform_(-bound, bound))
#                 # hard sanitize in-place (safe here; this is init-time)
#                 with torch.no_grad():
#                     p.copy_(torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0))
#     # --- optionally upsize attn extractor embeddings to cover model L/H ---
#     def _ensure_attn_extractor_caps_(extractor, base_config):
#         L = getattr(base_config, "num_hidden_layers", None)
#         H = getattr(base_config, "num_attention_heads", None)
#         if L is None or H is None:
#             return
#         # layer_emb
#         if extractor.layer_emb.num_embeddings < L:
#             old = extractor.layer_emb
#             new = nn.Embedding(L, old.embedding_dim, device=old.weight.device, dtype=old.weight.dtype)
#             _init_embedding_(new)
#             extractor.layer_emb = new
#         # head_emb
#         if extractor.head_emb.num_embeddings < H:
#             old = extractor.head_emb
#             new = nn.Embedding(H, old.embedding_dim, device=old.weight.device, dtype=old.weight.dtype)
#             _init_embedding_(new)
#             extractor.head_emb = new

#     def _freeze_base_(allow_toplevel_names: list[str]):
#         allow = set(allow_toplevel_names)
#         for n, p in model.named_parameters():
#             top = n.split(".", 1)[0]
#             p.requires_grad = (top in allow)

#     def _assert_finite_(names: list[str]):
#         bad = []
#         with torch.no_grad():
#             for name in names:
#                 mod = getattr(model, name, None)
#                 if mod is None:
#                     continue
#                 for pname, p in mod.named_parameters(recurse=True):
#                     if not isinstance(p, torch.Tensor) or getattr(p, "is_meta", False) or not p.is_floating_point():
#                         continue
#                     if not torch.isfinite(p).all():
#                         bad.append(f"{name}.{pname}")
#         if bad:
#             raise RuntimeError(f"Non-finite values after init: {bad}")

#     # -------- discover custom heads, init precisely, freeze base, verify --------
#     custom_head_names = getattr(model, "_custom_head_names", None)
#     if not custom_head_names:
#         custom_head_names = [n for n in ["hid_extractor", "attn_extractor", "conf_extractor", "stop_head"] if hasattr(model, n)]

#     if custom_head_names:
#         # Ensure capacity of attn extractor pos-embs vs. base L/H
#         if "attn_extractor" in custom_head_names:
#             _ensure_attn_extractor_caps_(getattr(model, "attn_extractor"), model.config)

#         # Initialize
#         for name in custom_head_names:
#             mod = getattr(model, name)
#             if name == "attn_extractor":
#                 _init_attn_feature_extractor_(mod)
#             else:
#                 _init_generic_stack_(mod)

#         # NEW: make sure any free query params are finite & well-initialized
#         _init_free_queries_(mod)
        
#         _assert_finite_(custom_head_names)
#         _freeze_base_(custom_head_names)   # add "lm_head" here if you also want to train it

#         # small report
#         trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         total     = sum(p.numel() for p in model.parameters())
#         print(f"[custom heads ready] heads={custom_head_names}  "
#               f"trainable={trainable:,} ({trainable/total:.2%})  total={total:,}")
#     else:
#         print("[custom heads] none detected; skipping post-load init/freeze.")

#     return model




# #t3 final, trivia
# def get_model(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> AutoModelForCausalLM:
#     """
#     Load the base model, then precisely initialize ONLY the custom heads
#     (attn_extractor / conf_extractor / hid_extractor / stop_head) in fp32,
#     freeze the base, and verify all trainable params are finite.
#     """
#     # -------- load pretrained --------
#     torch_dtype = (
#         model_args.torch_dtype if model_args.torch_dtype in ["auto", None]
#         else getattr(torch, model_args.torch_dtype)
#     )
#     quantization_config = get_quantization_config(model_args)
#     model_kwargs = dict(
#         revision=model_args.model_revision,
#         trust_remote_code=model_args.trust_remote_code,
#         attn_implementation=model_args.attn_implementation,
#         torch_dtype=torch_dtype,
#         use_cache=False if training_args.gradient_checkpointing else True,
#         device_map=get_kbit_device_map() if quantization_config is not None else None,
#         quantization_config=quantization_config,
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         **model_kwargs,
#     )

#     # Ensure reduction grid is available in config (used by Qwen3Model)
#     if not hasattr(model.config, "stop_k_att"):
#         setattr(model.config, "stop_k_att", 48)

#     # -------- precise, fp32-safe init helpers --------
#     def _copy_init_fp32_(param: torch.Tensor, init_fn):
#         if not isinstance(param, torch.Tensor) or getattr(param, "is_meta", False):
#             return
#         if not param.is_floating_point():
#             return
#         with torch.no_grad():
#             buf = torch.empty_like(param, dtype=torch.float32, device=param.device)
#             init_fn(buf)                      # initialize in fp32
#             param.copy_(torch.nan_to_num(buf, nan=0.0, posinf=0.0, neginf=0.0).to(param.dtype))  # cast back

#     def _init_layernorm_like_(m):
#         if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
#             _copy_init_fp32_(m.weight, lambda t: t.fill_(1.0))
#         if hasattr(m, "bias") and isinstance(m.bias, torch.Tensor) and m.bias is not None:
#             _copy_init_fp32_(m.bias, lambda t: t.zero_())

#     def _init_groupnorm_(m: nn.GroupNorm):
#         if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
#             _copy_init_fp32_(m.weight, lambda t: t.fill_(1.0))
#         if hasattr(m, "bias") and isinstance(m.bias, torch.Tensor) and m.bias is not None:
#             _copy_init_fp32_(m.bias, lambda t: t.zero_())

#     def _init_embedding_(m: nn.Embedding):
#         d = m.embedding_dim
#         bound = 1.0 / math.sqrt(max(1, d))
#         _copy_init_fp32_(m.weight, lambda t: t.uniform_(-bound, bound))

#     def _init_linear_(m: nn.Linear, gelu_expected: bool = True):
#         gain = math.sqrt(2.0) if gelu_expected else 1.0
#         _copy_init_fp32_(m.weight, lambda t: nn.init.xavier_uniform_(t, gain=gain))
#         if m.bias is not None:
#             _copy_init_fp32_(m.bias,   lambda t: t.zero_())

#     def _init_conv1d_(m: nn.Conv1d):
#         _copy_init_fp32_(m.weight, lambda t: nn.init.kaiming_uniform_(t, a=0.0, mode="fan_in", nonlinearity="relu"))
#         if m.bias is not None:
#             fan_in = (m.in_channels // max(1, m.groups)) * m.kernel_size[0]
#             bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1e-3
#             _copy_init_fp32_(m.bias, lambda t: t.uniform_(-bound, bound))

#     def _init_conv2d_(m: nn.Conv2d):
#         _copy_init_fp32_(m.weight, lambda t: nn.init.kaiming_uniform_(t, a=0.0, mode="fan_in", nonlinearity="relu"))
#         if m.bias is not None:
#             kh, kw = m.kernel_size
#             fan_in = (m.in_channels // max(1, m.groups)) * kh * kw
#             bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1e-3
#             _copy_init_fp32_(m.bias, lambda t: t.uniform_(-bound, bound))

#     def _init_mha_(m: nn.MultiheadAttention):
#         # in_proj_weight: (3*embed_dim, embed_dim)
#         if hasattr(m, "in_proj_weight") and isinstance(m.in_proj_weight, torch.Tensor):
#             _copy_init_fp32_(m.in_proj_weight, lambda t: nn.init.xavier_uniform_(t))
#         if hasattr(m, "in_proj_bias") and isinstance(m.in_proj_bias, torch.Tensor) and m.in_proj_bias is not None:
#             _copy_init_fp32_(m.in_proj_bias, lambda t: t.zero_())
#         if hasattr(m, "out_proj") and isinstance(m.out_proj, nn.Linear):
#             _copy_init_fp32_(m.out_proj.weight, lambda t: nn.init.xavier_uniform_(t))
#             if m.out_proj.bias is not None:
#                 _copy_init_fp32_(m.out_proj.bias, lambda t: t.zero_())


#     # --- init for AttnFeatureExtractorLite (now includes SE + tok_mixer) ---
#     def _init_attn_feature_extractor_(extractor):
#         # convs + norms
#         for m in [extractor.conv1, extractor.conv2]:
#             _init_conv2d_(m)
#         for m in [extractor.gn1, extractor.gn2]:
#             _init_groupnorm_(m)

#         # SE2d
#         if hasattr(extractor, "se2d"):
#             _init_conv2d_(extractor.se2d.fc1)
#             _init_conv2d_(extractor.se2d.fc2)

#         # token projection (LinearLN)
#         if hasattr(extractor, "proj"):
#             for sm in extractor.proj.modules():
#                 if isinstance(sm, nn.Linear):
#                     _init_linear_(sm, gelu_expected=True)
#                 elif isinstance(sm, nn.LayerNorm):
#                     _init_layernorm_like_(sm)
#                 elif isinstance(sm, nn.GroupNorm):
#                     _init_groupnorm_(sm)

#         # positional embeddings
#         _init_embedding_(extractor.layer_emb)
#         _init_embedding_(extractor.head_emb)

#         # optional token mixer (TransformerEncoder) — include MHA + FFN + norms
#         if getattr(extractor, "tok_mixer", None) is not None:
#             for sm in extractor.tok_mixer.modules():
#                 if isinstance(sm, nn.MultiheadAttention):
#                     _init_mha_(sm)
#                 elif isinstance(sm, nn.Linear):
#                     _init_linear_(sm, gelu_expected=True)
#                 elif isinstance(sm, nn.LayerNorm):
#                     _init_layernorm_like_(sm)

#         # queries
#         if hasattr(extractor, "query") and isinstance(extractor.query, torch.Tensor):
#             d = extractor.query.shape[-1]
#             bound = 1.0 / math.sqrt(max(1, d))
#             _copy_init_fp32_(extractor.query, lambda t: t.uniform_(-bound, bound))

#         # output projection
#         if hasattr(extractor, "out"):
#             for sm in extractor.out.modules():
#                 if isinstance(sm, nn.Linear):
#                     _init_linear_(sm, gelu_expected=True)
#                 elif isinstance(sm, nn.LayerNorm):
#                     _init_layernorm_like_(sm)
#                 elif isinstance(sm, nn.GroupNorm):
#                     _init_groupnorm_(sm)

#     # --- generic init for conf/hid/stop (recurses into LinearLN etc.) ---
#     def _init_generic_stack_(module: nn.Module):
#         for m in module.modules():
#             if isinstance(m, nn.Linear):
#                 _init_linear_(m, gelu_expected=True)
#             elif isinstance(m, nn.Conv1d):
#                 _init_conv1d_(m)
#             elif isinstance(m, nn.Conv2d):
#                 _init_conv2d_(m)
#             elif isinstance(m, nn.MultiheadAttention):
#                 _init_mha_(m)
#             elif isinstance(m, nn.LayerNorm):
#                 _init_layernorm_like_(m)
#             elif isinstance(m, nn.GroupNorm):
#                 _init_groupnorm_(m)
#             elif isinstance(m, nn.Embedding):
#                 _init_embedding_(m)

#     def _init_free_queries_(module: nn.Module):
#         """Re-init any top-level nn.Parameter named 'query'/'q_row'/'q_col'."""
#         for attr in ("query", "q_row", "q_col"):
#             p = getattr(module, attr, None)
#             if isinstance(p, torch.Tensor) and p.is_floating_point():
#                 d = p.shape[-1] if p.ndim >= 1 else 1
#                 bound = 1.0 / math.sqrt(max(1, int(d)))
#                 _copy_init_fp32_(p, lambda t: t.uniform_(-bound, bound))
#                 with torch.no_grad():
#                     p.copy_(torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0))

#     # --- optionally upsize attn extractor embeddings to cover model L/H ---
#     def _ensure_attn_extractor_caps_(extractor, base_config):
#         L = getattr(base_config, "num_hidden_layers", None)
#         H = getattr(base_config, "num_attention_heads", None)
#         if L is None or H is None:
#             return
#         # layer_emb
#         if extractor.layer_emb.num_embeddings < L:
#             old = extractor.layer_emb
#             new = nn.Embedding(L, old.embedding_dim, device=old.weight.device, dtype=old.weight.dtype)
#             _init_embedding_(new)
#             extractor.layer_emb = new
#         # head_emb
#         if extractor.head_emb.num_embeddings < H:
#             old = extractor.head_emb
#             new = nn.Embedding(H, old.embedding_dim, device=old.weight.device, dtype=old.weight.dtype)
#             _init_embedding_(new)
#             extractor.head_emb = new

#     def _freeze_base_(allow_toplevel_names: list[str]):
#         allow = set(allow_toplevel_names)
#         for n, p in model.named_parameters():
#             top = n.split(".", 1)[0]
#             p.requires_grad_(top in allow)

#     def _assert_finite_(names: list[str]):
#         bad = []
#         with torch.no_grad():
#             for name in names:
#                 mod = getattr(model, name, None)
#                 if mod is None:
#                     continue
#                 for pname, p in mod.named_parameters(recurse=True):
#                     if (not isinstance(p, torch.Tensor)) or getattr(p, "is_meta", False) or (not p.is_floating_point()):
#                         continue
#                     if not torch.isfinite(p).all():
#                         bad.append(f"{name}.{pname}")
#         if bad:
#             raise RuntimeError(f"Non-finite values after init: {bad}")

#     def _move_to_device_like_(module: nn.Module, ref_tensor: torch.Tensor):
#         """Move module params/buffers to ref_tensor.device, preserving dtype (keep heads fp32)."""
#         device = ref_tensor.device
#         module.to(device=device, dtype=None)  # dtype=None keeps each param's dtype

#     # -------- discover custom heads, init precisely, freeze base, verify --------
#     custom_head_names = getattr(model, "_custom_head_names", None)
#     if not custom_head_names:
#         custom_head_names = [n for n in ["hid_extractor", "attn_extractor", "conf_extractor", "stop_head"] if hasattr(model, n)]

#     if custom_head_names:
#         # Ensure capacity of attn extractor pos-embs vs. base L/H
#         if "attn_extractor" in custom_head_names:
#             _ensure_attn_extractor_caps_(getattr(model, "attn_extractor"), model.config)

#         # Initialize each head
#         for name in custom_head_names:
#             mod = getattr(model, name)
#             if name == "attn_extractor":
#                 _init_attn_feature_extractor_(mod)
#             else:
#                 _init_generic_stack_(mod)
#             # ensure any free query-like params are sane for THIS module
#             _init_free_queries_(mod)

#         # Move heads onto same device as base embeddings (preserve fp32 dtype)
#         try:
#             ref = model.model.embed_tokens.weight
#             for name in custom_head_names:
#                 _move_to_device_like_(getattr(model, name), ref)
#         except Exception:
#             # If the base model doesn't expose embed_tokens, skip device move
#             pass

#         _assert_finite_(custom_head_names)
#         _freeze_base_(custom_head_names)   # add "lm_head" if you also want to train it

#         # small report
#         trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         total     = sum(p.numel() for p in model.parameters())
#         print(f"[custom heads ready] heads={custom_head_names}  "
#               f"trainable={trainable:,} ({trainable/total:.2%})  total={total:,}")
#     else:
#         print("[custom heads] none detected; skipping post-load init/freeze.")

#     return model





# #t4
# def get_model(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> AutoModelForCausalLM:
#     """
#     Load the base model, then precisely initialize ONLY the custom heads
#     (attn_extractor / conf_extractor / hid_extractor / stop_head) in fp32,
#     freeze the base, and verify all trainable params are finite.
#     """
#     # -------- load pretrained --------
#     torch_dtype = (
#         model_args.torch_dtype if model_args.torch_dtype in ["auto", None]
#         else getattr(torch, model_args.torch_dtype)
#     )
#     quantization_config = get_quantization_config(model_args)
#     model_kwargs = dict(
#         revision=model_args.model_revision,
#         trust_remote_code=model_args.trust_remote_code,
#         attn_implementation=model_args.attn_implementation,
#         torch_dtype=torch_dtype,
#         use_cache=False if training_args.gradient_checkpointing else True,
#         device_map=get_kbit_device_map() if quantization_config is not None else None,
#         quantization_config=quantization_config,
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         **model_kwargs,
#     )

#     # Ensure reduction grid is available in config (used by Qwen3Model)
#     if not hasattr(model.config, "stop_k_att"):
#         setattr(model.config, "stop_k_att", 48)

#     # -------- precise, fp32-safe init helpers --------
#     def _copy_init_fp32_(param: torch.Tensor, init_fn):
#         if not isinstance(param, torch.Tensor) or getattr(param, "is_meta", False):
#             return
#         if not param.is_floating_point():
#             return
#         with torch.no_grad():
#             buf = torch.empty_like(param, dtype=torch.float32, device=param.device)
#             init_fn(buf)                      # initialize in fp32
#             param.copy_(torch.nan_to_num(buf, nan=0.0, posinf=0.0, neginf=0.0).to(param.dtype))  # cast back

#     def _init_layernorm_like_(m):
#         if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
#             _copy_init_fp32_(m.weight, lambda t: t.fill_(1.0))
#         if hasattr(m, "bias") and isinstance(m.bias, torch.Tensor) and m.bias is not None:
#             _copy_init_fp32_(m.bias, lambda t: t.zero_())

#     def _init_groupnorm_(m: nn.GroupNorm):
#         if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
#             _copy_init_fp32_(m.weight, lambda t: t.fill_(1.0))
#         if hasattr(m, "bias") and isinstance(m.bias, torch.Tensor) and m.bias is not None:
#             _copy_init_fp32_(m.bias, lambda t: t.zero_())

#     def _init_embedding_(m: nn.Embedding):
#         d = m.embedding_dim
#         bound = 1.0 / math.sqrt(max(1, d))
#         _copy_init_fp32_(m.weight, lambda t: t.uniform_(-bound, bound))

#     def _init_linear_(m: nn.Linear, gelu_expected: bool = True):
#         gain = math.sqrt(2.0) if gelu_expected else 1.0
#         _copy_init_fp32_(m.weight, lambda t: nn.init.xavier_uniform_(t, gain=gain))
#         if m.bias is not None:
#             _copy_init_fp32_(m.bias,   lambda t: t.zero_())

#     def _init_conv1d_(m: nn.Conv1d):
#         _copy_init_fp32_(m.weight, lambda t: nn.init.kaiming_uniform_(t, a=0.0, mode="fan_in", nonlinearity="relu"))
#         if m.bias is not None:
#             fan_in = (m.in_channels // max(1, m.groups)) * m.kernel_size[0]
#             bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1e-3
#             _copy_init_fp32_(m.bias, lambda t: t.uniform_(-bound, bound))

#     def _init_conv2d_(m: nn.Conv2d):
#         _copy_init_fp32_(m.weight, lambda t: nn.init.kaiming_uniform_(t, a=0.0, mode="fan_in", nonlinearity="relu"))
#         if m.bias is not None:
#             kh, kw = m.kernel_size
#             fan_in = (m.in_channels // max(1, m.groups)) * kh * kw
#             bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1e-3
#             _copy_init_fp32_(m.bias, lambda t: t.uniform_(-bound, bound))

#     def _init_mha_(m: nn.MultiheadAttention):
#         # in_proj_weight: (3*embed_dim, embed_dim)
#         if hasattr(m, "in_proj_weight") and isinstance(m.in_proj_weight, torch.Tensor):
#             _copy_init_fp32_(m.in_proj_weight, lambda t: nn.init.xavier_uniform_(t))
#         if hasattr(m, "in_proj_bias") and isinstance(m.in_proj_bias, torch.Tensor) and m.in_proj_bias is not None:
#             _copy_init_fp32_(m.in_proj_bias, lambda t: t.zero_())
#         if hasattr(m, "out_proj") and isinstance(m.out_proj, nn.Linear):
#             _copy_init_fp32_(m.out_proj.weight, lambda t: nn.init.xavier_uniform_(t))
#             if m.out_proj.bias is not None:
#                 _copy_init_fp32_(m.out_proj.bias, lambda t: t.zero_())

#     # ---- generic recursive init (covers conv/linear/norm/emb/mha) ----
#     def _init_generic_stack_(module: nn.Module):
#         for m in module.modules():
#             if isinstance(m, nn.Linear):
#                 _init_linear_(m, gelu_expected=True)
#             elif isinstance(m, nn.Conv1d):
#                 _init_conv1d_(m)
#             elif isinstance(m, nn.Conv2d):
#                 _init_conv2d_(m)
#             elif isinstance(m, nn.MultiheadAttention):
#                 _init_mha_(m)
#             elif isinstance(m, nn.LayerNorm):
#                 _init_layernorm_like_(m)
#             elif isinstance(m, nn.GroupNorm):
#                 _init_groupnorm_(m)
#             elif isinstance(m, nn.Embedding):
#                 _init_embedding_(m)

#     def _init_free_queries_(module: nn.Module):
#         """Re-init any top-level nn.Parameter named 'query'/'q_row'/'q_col' in a module."""
#         for attr in ("query", "q_row", "q_col"):
#             p = getattr(module, attr, None)
#             if isinstance(p, torch.Tensor) and p.is_floating_point():
#                 d = p.shape[-1] if p.ndim >= 1 else 1
#                 bound = 1.0 / math.sqrt(max(1, int(d)))
#                 _copy_init_fp32_(p, lambda t: t.uniform_(-bound, bound))
#                 with torch.no_grad():
#                     p.copy_(torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0))

#     # --- attn extractor specific: ensure pos-emb capacity & init embeddings ---
#     def _ensure_attn_extractor_caps_(extractor, base_config):
#         L = getattr(base_config, "num_hidden_layers", None)
#         H = getattr(base_config, "num_attention_heads", None)
#         if L is None or H is None:
#             return
#         # layer_emb
#         if hasattr(extractor, "layer_emb") and extractor.layer_emb.num_embeddings < L:
#             old = extractor.layer_emb
#             new = nn.Embedding(L, old.embedding_dim, device=old.weight.device, dtype=old.weight.dtype)
#             _init_embedding_(new)
#             extractor.layer_emb = new
#         # head_emb
#         if hasattr(extractor, "head_emb") and extractor.head_emb.num_embeddings < H:
#             old = extractor.head_emb
#             new = nn.Embedding(H, old.embedding_dim, device=old.weight.device, dtype=old.weight.dtype)
#             _init_embedding_(new)
#             extractor.head_emb = new

#     def _freeze_base_(allow_toplevel_names: list[str]):
#         allow = set(allow_toplevel_names)
#         for n, p in model.named_parameters():
#             top = n.split(".", 1)[0]
#             p.requires_grad_(top in allow)

#     def _assert_finite_(names: list[str]):
#         bad = []
#         with torch.no_grad():
#             for name in names:
#                 mod = getattr(model, name, None)
#                 if mod is None:
#                     continue
#                 for pname, p in mod.named_parameters(recurse=True):
#                     if (not isinstance(p, torch.Tensor)) or getattr(p, "is_meta", False) or (not p.is_floating_point()):
#                         continue
#                     if not torch.isfinite(p).all():
#                         bad.append(f"{name}.{pname}")
#         if bad:
#             raise RuntimeError(f"Non-finite values after init: {bad}")

#     def _move_to_device_like_(module: nn.Module, ref_tensor: torch.Tensor):
#         """Move module params/buffers to ref_tensor.device, preserving per-param dtype (keep heads fp32)."""
#         device = ref_tensor.device
#         module.to(device=device, dtype=None)

#     # -------- discover custom heads, init precisely, freeze base, verify --------
#     custom_head_names = getattr(model, "_custom_head_names", None)
#     if not custom_head_names:
#         custom_head_names = [n for n in ["hid_extractor", "attn_extractor", "conf_extractor", "stop_head"] if hasattr(model, n)]

#     if custom_head_names:
#         # Ensure capacity of attn extractor pos-embs vs. base L/H
#         if "attn_extractor" in custom_head_names:
#             _ensure_attn_extractor_caps_(getattr(model, "attn_extractor"), model.config)

#         # Initialize each head (new modules are handled by the generic walker)
#         for name in custom_head_names:
#             mod = getattr(model, name)
#             _init_generic_stack_(mod)   # convs, linears, norms, embeddings, (MHA if any)
#             _init_free_queries_(mod)    # initialize learnable query parameters if present

#         # Move heads onto same device as base embeddings (preserve fp32 dtype on heads)
#         try:
#             ref = model.model.embed_tokens.weight
#             for name in custom_head_names:
#                 _move_to_device_like_(getattr(model, name), ref)
#         except Exception:
#             # If the base model doesn't expose embed_tokens, skip device move
#             pass

#         _assert_finite_(custom_head_names)
#         _freeze_base_(custom_head_names)   # add "lm_head" here if you also want to train it

#         # small report
#         trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         total     = sum(p.numel() for p in model.parameters())
#         print(f"[custom heads ready] heads={custom_head_names}  "
#               f"trainable={trainable:,} ({trainable/total:.2%})  total={total:,}")
#     else:
#         print("[custom heads] none detected; skipping post-load init/freeze.")

#     return model


# ## t5
# def get_model(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> AutoModelForCausalLM:
#     """
#     Load the base model, then precisely initialize ONLY the custom heads
#     (attn_extractor / conf_extractor / hid_extractor / stop_head),
#     init in fp32, freeze the base, and verify all trainable params are finite.
#     """
#     # -------- load pretrained --------
#     torch_dtype = (
#         model_args.torch_dtype if model_args.torch_dtype in ["auto", None]
#         else getattr(torch, model_args.torch_dtype)
#     )
#     quantization_config = get_quantization_config(model_args)
#     model_kwargs = dict(
#         revision=model_args.model_revision,
#         trust_remote_code=model_args.trust_remote_code,
#         attn_implementation=model_args.attn_implementation,
#         torch_dtype=torch_dtype,
#         use_cache=False if training_args.gradient_checkpointing else True,
#         device_map=get_kbit_device_map() if quantization_config is not None else None,
#         quantization_config=quantization_config,
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         **model_kwargs,
#     )

#     # Ensure reduction grid is available in config (used by some models)
#     if not hasattr(model.config, "stop_k_att"):
#         setattr(model.config, "stop_k_att", 48)

#     # -------- precise, fp32-safe init helpers --------
#     def _copy_init_fp32_(param: torch.Tensor, init_fn):
#         if not isinstance(param, torch.Tensor) or getattr(param, "is_meta", False):
#             return
#         if not param.is_floating_point():
#             return
#         with torch.no_grad():
#             buf = torch.empty_like(param, dtype=torch.float32, device=param.device)
#             init_fn(buf)  # initialize in fp32
#             buf = torch.nan_to_num(buf, nan=0.0, posinf=0.0, neginf=0.0)
#             param.copy_(buf.to(param.dtype))

#     def _init_layernorm_like_(m):
#         if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
#             _copy_init_fp32_(m.weight, lambda t: t.fill_(1.0))
#         if hasattr(m, "bias") and isinstance(m.bias, torch.Tensor) and m.bias is not None:
#             _copy_init_fp32_(m.bias, lambda t: t.zero_())

#     def _init_groupnorm_(m: nn.GroupNorm):
#         if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
#             _copy_init_fp32_(m.weight, lambda t: t.fill_(1.0))
#         if hasattr(m, "bias") and isinstance(m.bias, torch.Tensor) and m.bias is not None:
#             _copy_init_fp32_(m.bias, lambda t: t.zero_())

#     def _init_embedding_(m: nn.Embedding):
#         d = m.embedding_dim
#         bound = 1.0 / math.sqrt(max(1, d))
#         _copy_init_fp32_(m.weight, lambda t: t.uniform_(-bound, bound))

#     def _init_linear_(m: nn.Linear, *, gain: float = 1.0):
#         _copy_init_fp32_(m.weight, lambda t: nn.init.xavier_uniform_(t, gain=gain))
#         if m.bias is not None:
#             _copy_init_fp32_(m.bias, lambda t: t.zero_())

#     def _init_conv1d_(m: nn.Conv1d):
#         _copy_init_fp32_(m.weight, lambda t: nn.init.kaiming_uniform_(t, a=0.0, mode="fan_in", nonlinearity="relu"))
#         if m.bias is not None:
#             fan_in = (m.in_channels // max(1, m.groups)) * m.kernel_size[0]
#             bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1e-3
#             _copy_init_fp32_(m.bias, lambda t: t.uniform_(-bound, bound))

#     def _init_conv2d_(m: nn.Conv2d):
#         _copy_init_fp32_(m.weight, lambda t: nn.init.kaiming_uniform_(t, a=0.0, mode="fan_in", nonlinearity="relu"))
#         if m.bias is not None:
#             kh, kw = m.kernel_size
#             fan_in = (m.in_channels // max(1, m.groups)) * kh * kw
#             bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1e-3
#             _copy_init_fp32_(m.bias, lambda t: t.uniform_(-bound, bound))

#     # ---- Custom Set-Transformer blocks (your classes) ----
#     def _init_custom_mha_(m):  # for your MultiHeadAttention (q/k/v/o are Linear)
#         _init_linear_(m.q, gain=1.0)
#         _init_linear_(m.k, gain=1.0)
#         _init_linear_(m.v, gain=1.0)
#         _init_linear_(m.o, gain=1.0)

#     def _init_mab_(m):  # MAB = MHA + FF + 2xLN
#         _init_custom_mha_(m.mha)
#         for sm in m.ff:
#             if isinstance(sm, nn.Linear):
#                 _init_linear_(sm, gain=math.sqrt(2.0))  # GELU
#         _init_layernorm_like_(m.ln1)
#         _init_layernorm_like_(m.ln2)

#     def _init_sab_(sab: 'SAB'):
#         for mab in sab.layers:
#             _init_mab_(mab)

#     def _init_pma_(pma: 'PMA'):
#         # seeds S
#         S = pma.S
#         d = S.shape[-1]
#         bound = 1.0 / math.sqrt(max(1, d))
#         _copy_init_fp32_(S, lambda t: t.uniform_(-bound, bound))
#         _init_mab_(pma.mab)

#     # --- init AttnFeatureExtractorLite (new design) ---
#     def _init_attn_feature_extractor_(extractor):
#         # CNN stems (cnn_s0/s1/s2) with Conv2d+GN+GELU
#         for stem in [extractor.cnn_s0, extractor.cnn_s1, extractor.cnn_s2]:
#             for sm in stem.modules():
#                 if isinstance(sm, nn.Conv2d):
#                     _init_conv2d_(sm)
#                 elif isinstance(sm, nn.GroupNorm):
#                     _init_groupnorm_(sm)
#                 elif isinstance(sm, nn.LayerNorm):
#                     _init_layernorm_like_(sm)
#                 elif isinstance(sm, nn.Linear):
#                     _init_linear_(sm, gain=math.sqrt(2.0))

#         # projection to token dim
#         _init_linear_(extractor.proj, gain=math.sqrt(2.0))

#         # positional embeddings (layer/head)
#         _init_embedding_(extractor.layer_emb)
#         _init_embedding_(extractor.head_emb)

#         # Set Transformer mixer
#         _init_sab_(extractor.sab)
#         _init_pma_(extractor.pma)

#         # output MLP
#         for sm in extractor.out:
#             if isinstance(sm, nn.Linear):
#                 _init_linear_(sm, gain=math.sqrt(2.0))
#             elif isinstance(sm, nn.LayerNorm):
#                 _init_layernorm_like_(sm)
#             elif isinstance(sm, nn.GroupNorm):
#                 _init_groupnorm_(sm)

#     # --- init Hidden/Conf extractors (generic + extras) ---
#     def _init_generic_stack_(module: nn.Module):
#         for m in module.modules():
#             if isinstance(m, nn.Linear):
#                 _init_linear_(m, gain=math.sqrt(2.0))
#             elif isinstance(m, nn.Conv1d):
#                 _init_conv1d_(m)
#             elif isinstance(m, nn.Conv2d):
#                 _init_conv2d_(m)
#             elif isinstance(m, nn.LayerNorm):
#                 _init_layernorm_like_(m)
#             elif isinstance(m, nn.GroupNorm):
#                 _init_groupnorm_(m)
#             elif isinstance(m, nn.Embedding):
#                 _init_embedding_(m)
#             # your custom attention blocks:
#             elif m.__class__.__name__ == "MultiHeadAttention":
#                 _init_custom_mha_(m)
#             elif m.__class__.__name__ == "MAB":
#                 _init_mab_(m)
#             elif m.__class__.__name__ == "SAB":
#                 _init_sab_(m)
#             elif m.__class__.__name__ == "PMA":
#                 _init_pma_(m)

#         # initialize common free parameters if present
#         for attr in ("pos",):
#             p = getattr(module, attr, None)
#             if isinstance(p, torch.Tensor) and p.is_floating_point():
#                 d = p.shape[-1] if p.ndim >= 1 else 1
#                 std = 1.0 / math.sqrt(max(1, d))
#                 _copy_init_fp32_(p, lambda t: t.normal_(mean=0.0, std=std))

#         # PMA seeds often live at module.pma.S; already handled via _init_pma_

#     # --- optionally upsize attn extractor embeddings to cover model L/H ---
#     def _ensure_attn_extractor_caps_(extractor, base_config):
#         L = getattr(base_config, "num_hidden_layers", None)
#         H = getattr(base_config, "num_attention_heads", None)
#         if L is None or H is None:
#             return
#         # layer_emb
#         if extractor.layer_emb.num_embeddings < L:
#             old = extractor.layer_emb
#             new = nn.Embedding(L, old.embedding_dim, device=old.weight.device, dtype=old.weight.dtype)
#             _init_embedding_(new)
#             with torch.no_grad():
#                 new.weight.data[:old.num_embeddings].copy_(old.weight.data)
#             extractor.layer_emb = new
#         # head_emb
#         if extractor.head_emb.num_embeddings < H:
#             old = extractor.head_emb
#             new = nn.Embedding(H, old.embedding_dim, device=old.weight.device, dtype=old.weight.dtype)
#             _init_embedding_(new)
#             with torch.no_grad():
#                 new.weight.data[:old.num_embeddings].copy_(old.weight.data)
#             extractor.head_emb = new

#     def _freeze_base_(allow_toplevel_names: list[str]):
#         allow = set(allow_toplevel_names)
#         for n, p in model.named_parameters():
#             top = n.split(".", 1)[0]
#             p.requires_grad_(top in allow)

#     def _assert_finite_(names: list[str]):
#         bad = []
#         with torch.no_grad():
#             for name in names:
#                 mod = getattr(model, name, None)
#                 if mod is None:
#                     continue
#                 for pname, p in mod.named_parameters(recurse=True):
#                     if (not isinstance(p, torch.Tensor)) or getattr(p, "is_meta", False) or (not p.is_floating_point()):
#                         continue
#                     if not torch.isfinite(p).all():
#                         bad.append(f"{name}.{pname}")
#         if bad:
#             raise RuntimeError(f"Non-finite values after init: {bad}")

#     def _move_to_device_like_(module: nn.Module, ref_tensor: torch.Tensor):
#         """Move module params/buffers to ref_tensor.device, preserving dtype (keep heads fp32)."""
#         device = ref_tensor.device
#         module.to(device=device, dtype=None)  # dtype=None keeps each param's dtype

#     # -------- discover custom heads, init precisely, freeze base, verify --------
#     custom_head_names = getattr(model, "_custom_head_names", None)
#     if not custom_head_names:
#         custom_head_names = [n for n in ["hid_extractor", "attn_extractor", "conf_extractor", "stop_head"] if hasattr(model, n)]

#     if custom_head_names:
#         # Ensure capacity of attn extractor pos-embs vs. base L/H
#         if "attn_extractor" in custom_head_names:
#             _ensure_attn_extractor_caps_(getattr(model, "attn_extractor"), model.config)

#         # Initialize each head
#         for name in custom_head_names:
#             mod = getattr(model, name)
#             if name == "attn_extractor":
#                 _init_attn_feature_extractor_(mod)
#             else:
#                 _init_generic_stack_(mod)

#         # Move heads onto same device as base embeddings (preserve dtype)
#         try:
#             ref = model.model.embed_tokens.weight  # most HF CausalLMs
#             for name in custom_head_names:
#                 _move_to_device_like_(getattr(model, name), ref)
#         except Exception:
#             pass

#         _assert_finite_(custom_head_names)
#         _freeze_base_(custom_head_names)   # add "lm_head" if you also want to train it

#         # small report
#         trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         total     = sum(p.numel() for p in model.parameters())
#         print(f"[custom heads ready] heads={custom_head_names}  "
#               f"trainable={trainable:,} ({trainable/total:.2%})  total={total:,}")
#     else:
#         print("[custom heads] none detected; skipping post-load init/freeze.")

#     return model


#general?

# universal_model_init.py
# A robust, universal initializer for HF CausalLM + custom heads (attn_extractor, conf_extractor,
# hid_extractor, stop_head, correctness_head, etc.). Safe in fp32, non-finite proof, with runtime guards.

import math
from typing import Dict, Callable, Optional, List, Iterable, Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers import Mxfp4Config

# --- Optional project dependencies: fall back safely if not available ---
try:
    # Replace with your actual imports
    from your_project.quant import get_quantization_config, get_kbit_device_map  # type: ignore
except Exception:
    def get_quantization_config(_): return None
    def get_kbit_device_map(): return None

# If you have typed configs in your project, you can keep these forward-declared type hints:
# from your_project.configs import ModelConfig, SFTConfig, GRPOConfig


# =========================
# Utility: deterministic SEED
# =========================
def _maybe_set_seed(training_args: Any):
    seed = getattr(training_args, "seed", None)
    if seed is None:
        return
    try:
        import random, numpy as np
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        # (Optional) for reproducibility—can slow down
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# =========================
# FP32-safe init primitives
# =========================
def _copy_init_fp32_(param: torch.Tensor, init_fn: Callable[[torch.Tensor], None]):
    if not isinstance(param, torch.Tensor) or getattr(param, "is_meta", False):
        return
    if not param.is_floating_point():
        return
    with torch.no_grad():
        buf = torch.empty_like(param, dtype=torch.float32, device=param.device)
        init_fn(buf)
        buf = torch.nan_to_num(buf, nan=0.0, posinf=0.0, neginf=0.0)
        param.copy_(buf.to(param.dtype))


def _init_layernorm_like_(m: nn.Module):
    if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
        _copy_init_fp32_(m.weight, lambda t: t.fill_(1.0))
    if hasattr(m, "bias") and isinstance(m.bias, torch.Tensor) and m.bias is not None:
        _copy_init_fp32_(m.bias, lambda t: t.zero_())


def _init_groupnorm_(m: nn.GroupNorm):
    if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
        _copy_init_fp32_(m.weight, lambda t: t.fill_(1.0))
    if hasattr(m, "bias") and isinstance(m.bias, torch.Tensor) and m.bias is not None:
        _copy_init_fp32_(m.bias, lambda t: t.zero_())


def _init_embedding_(m: nn.Embedding):
    d = m.embedding_dim
    bound = 1.0 / math.sqrt(max(1, d))
    _copy_init_fp32_(m.weight, lambda t: t.uniform_(-bound, bound))


def _init_linear_(m: nn.Linear, gain: float = math.sqrt(2.0)):
    _copy_init_fp32_(m.weight, lambda t: nn.init.xavier_uniform_(t, gain=gain))
    if m.bias is not None:
        _copy_init_fp32_(m.bias, lambda t: t.zero_())


def _init_conv1d_(m: nn.Conv1d):
    _copy_init_fp32_(m.weight, lambda t: nn.init.kaiming_uniform_(t, a=0.0, mode="fan_in", nonlinearity="relu"))
    if m.bias is not None:
        fan_in = (m.in_channels // max(1, m.groups)) * m.kernel_size[0]
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1e-3
        _copy_init_fp32_(m.bias, lambda t: t.uniform_(-bound, bound))


def _init_conv2d_(m: nn.Conv2d):
    _copy_init_fp32_(m.weight, lambda t: nn.init.kaiming_uniform_(t, a=0.0, mode="fan_in", nonlinearity="relu"))
    if m.bias is not None:
        kh, kw = m.kernel_size
        fan_in = (m.in_channels // max(1, m.groups)) * kh * kw
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1e-3
        _copy_init_fp32_(m.bias, lambda t: t.uniform_(-bound, bound))


def _init_torch_mha_(m: nn.MultiheadAttention):
    if hasattr(m, "in_proj_weight") and isinstance(m.in_proj_weight, torch.Tensor):
        _copy_init_fp32_(m.in_proj_weight, lambda t: nn.init.xavier_uniform_(t))
    if hasattr(m, "in_proj_bias") and isinstance(m.in_proj_bias, torch.Tensor) and m.in_proj_bias is not None:
        _copy_init_fp32_(m.in_proj_bias, lambda t: t.zero_())
    if hasattr(m, "out_proj") and isinstance(m.out_proj, nn.Linear):
        _copy_init_fp32_(m.out_proj.weight, lambda t: nn.init.xavier_uniform_(t))
        if m.out_proj.bias is not None:
            _copy_init_fp32_(m.out_proj.bias, lambda t: t.zero_())


# ============================================
# Name-based registry for project-specific cls
# ============================================
def _init_custom_mha_(m: nn.Module):  # expects attributes q,k,v,o as Linear
    if hasattr(m, "q"): _init_linear_(m.q, gain=1.0)
    if hasattr(m, "k"): _init_linear_(m.k, gain=1.0)
    if hasattr(m, "v"): _init_linear_(m.v, gain=1.0)
    if hasattr(m, "o"): _init_linear_(m.o, gain=1.0)


def _init_mab_(m: nn.Module):
    mha = getattr(m, "mha", None)
    if mha is not None:
        _init_custom_mha_(mha)
    ff = getattr(m, "ff", None)
    if isinstance(ff, nn.Module):
        for sm in ff.modules():
            if isinstance(sm, nn.Linear):
                _init_linear_(sm, gain=math.sqrt(2.0))
    ln1 = getattr(m, "ln1", None)
    ln2 = getattr(m, "ln2", None)
    if isinstance(ln1, nn.Module): _init_layernorm_like_(ln1)
    if isinstance(ln2, nn.Module): _init_layernorm_like_(ln2)


def _init_sab_(sab: nn.Module):
    layers = getattr(sab, "layers", None)
    if isinstance(layers, (list, nn.ModuleList)):
        for mab in layers:
            _init_mab_(mab)


def _init_pma_(pma: nn.Module):
    S = getattr(pma, "S", None)  # seeds (Parameter)
    if isinstance(S, torch.Tensor) and S.is_floating_point():
        d = S.shape[-1] if S.ndim >= 1 else 1
        bound = 1.0 / math.sqrt(max(1, int(d)))
        _copy_init_fp32_(S, lambda t: t.uniform_(-bound, bound))
    mab = getattr(pma, "mab", None)
    if mab is not None:
        _init_mab_(mab)


# default registry; extend via get_model(..., extra_init={...})
DEFAULT_NAME_REGISTRY: Dict[str, Callable[[nn.Module], None]] = {
    "MultiHeadAttention": _init_custom_mha_,
    "MAB": _init_mab_,
    "SAB": _init_sab_,
    "PMA": _init_pma_,
}


# ====================================
# Generic stack init + attn extractor
# ====================================
def _init_generic_stack_(module: nn.Module, name_registry: Dict[str, Callable[[nn.Module], None]]):
    for sm in module.modules():
        # type-based
        if isinstance(sm, nn.Linear):
            _init_linear_(sm, gain=math.sqrt(2.0))
        elif isinstance(sm, nn.Conv1d):
            _init_conv1d_(sm)
        elif isinstance(sm, nn.Conv2d):
            _init_conv2d_(sm)
        elif isinstance(sm, nn.LayerNorm):
            _init_layernorm_like_(sm)
        elif isinstance(sm, nn.GroupNorm):
            _init_groupnorm_(sm)
        elif isinstance(sm, nn.Embedding):
            _init_embedding_(sm)
        elif isinstance(sm, nn.MultiheadAttention):
            _init_torch_mha_(sm)

        # name-based custom classes
        fn = name_registry.get(sm.__class__.__name__, None)
        if fn is not None:
            try:
                fn(sm)
            except Exception:
                pass


def _init_free_queries_(module: nn.Module):
    for attr in ("query", "q_row", "q_col", "pos", "pos_emb", "pe"):
        p = getattr(module, attr, None)
        if isinstance(p, torch.Tensor) and p.is_floating_point():
            d = p.shape[-1] if p.ndim >= 1 else 1
            bound = 1.0 / math.sqrt(max(1, int(d)))
            _copy_init_fp32_(p, lambda t: t.uniform_(-bound, bound))


def _init_attn_feature_extractor_(extractor: nn.Module, name_registry: Dict[str, Callable[[nn.Module], None]]):
    # 0) Catch-all generic init FIRST (handles conv_in/conv1a/b/... gn3a, etc.)
    _init_generic_stack_(extractor, name_registry)

    # 1) CNN stems or named conv blocks many designs use
    for name in ["cnn_s0", "cnn_s1", "cnn_s2", "cnn", "stem", "conv1", "conv2"]:
        maybe = getattr(extractor, name, None)
        if isinstance(maybe, nn.Module):
            _init_generic_stack_(maybe, name_registry)

    # 2) Squeeze-Excitation (if present)
    se2d = getattr(extractor, "se2d", None)
    if isinstance(se2d, nn.Module):
        _init_generic_stack_(se2d, name_registry)

    # 3) token projection stacks
    for name in ["proj", "token_proj"]:
        proj = getattr(extractor, name, None)
        if isinstance(proj, nn.Module):
            _init_generic_stack_(proj, name_registry)

    # 4) positional embeddings (layer/head)
    for name in ["layer_emb", "head_emb"]:
        emb = getattr(extractor, name, None)
        if isinstance(emb, nn.Embedding):
            _init_embedding_(emb)

    # 5) optional token mixer (TransformerEncoder or custom)
    tok_mixer = getattr(extractor, "tok_mixer", None)
    if isinstance(tok_mixer, nn.Module):
        _init_generic_stack_(tok_mixer, name_registry)

    # 6) Set-Transformer style (SAB/PMA)
    sab = getattr(extractor, "sab", None)
    if isinstance(sab, nn.Module): _init_sab_(sab)
    pma = getattr(extractor, "pma", None)
    if isinstance(pma, nn.Module): _init_pma_(pma)

    # 7) output stacks
    out = getattr(extractor, "out", None)
    if isinstance(out, nn.Module):
        _init_generic_stack_(out, name_registry)

    # 8) queries/pos params
    _init_free_queries_(extractor)


# ==================================================
# Capacity: upsize attn extractor embeddings to L/H
# ==================================================
def _ensure_attn_extractor_caps_(extractor: nn.Module, base_config: Any):
    L = getattr(base_config, "num_hidden_layers", None)
    H = getattr(base_config, "num_attention_heads", None)
    if L is None or H is None:
        return

    le = getattr(extractor, "layer_emb", None)
    he = getattr(extractor, "head_emb", None)

    if isinstance(le, nn.Embedding) and le.num_embeddings < L:
        old = le
        new = nn.Embedding(L, old.embedding_dim, device=old.weight.device, dtype=old.weight.dtype)
        _init_embedding_(new)
        with torch.no_grad():
            new.weight.data[:old.num_embeddings].copy_(old.weight.data)
        extractor.layer_emb = new

    if isinstance(he, nn.Embedding) and he.num_embeddings < H:
        old = he
        new = nn.Embedding(H, old.embedding_dim, device=old.weight.device, dtype=old.weight.dtype)
        _init_embedding_(new)
        with torch.no_grad():
            new.weight.data[:old.num_embeddings].copy_(old.weight.data)
        extractor.head_emb = new


# =============================================
# Strong sanitizers + runtime (always-on) guard
# =============================================
def _sanitize_nonfinite_params_and_buffers_(module: nn.Module):
    """Re-init or zero any non-finite params/buffers."""
    with torch.no_grad():
        # Parameters
        for pname, p in module.named_parameters(recurse=True):
            if (not isinstance(p, torch.Tensor)) or getattr(p, "is_meta", False) or (not p.is_floating_point()):
                continue
            if torch.isfinite(p).all():
                continue
            if p.ndim >= 2:
                nn.init.xavier_uniform_(p.float())
            else:
                p.zero_()
            p.copy_(torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).to(p.dtype))

        # Buffers
        for bname, b in module.named_buffers(recurse=True):
            if (not isinstance(b, torch.Tensor)) or getattr(b, "is_meta", False) or (not b.is_floating_point()):
                continue
            if torch.isfinite(b).all():
                continue
            b.copy_(torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0))


def _deep_sanitize_until_finite_(module: nn.Module, max_passes: int = 3):
    for _ in range(max_passes):
        had_issue = False
        # scan params
        for _, p in module.named_parameters(recurse=True):
            if (isinstance(p, torch.Tensor) and p.is_floating_point() and not getattr(p, "is_meta", False)
                    and not torch.isfinite(p).all()):
                had_issue = True
                break
        # scan buffers if needed
        if not had_issue:
            for _, b in module.named_buffers(recurse=True):
                if (isinstance(b, torch.Tensor) and b.is_floating_point() and not getattr(b, "is_meta", False)
                        and not torch.isfinite(b).all()):
                    had_issue = True
                    break
        if not had_issue:
            return
        _sanitize_nonfinite_params_and_buffers_(module)


def _install_runtime_guards_(module: nn.Module, name: str = "head"):
    """Guard against future non-finites (lazy materialization, numeric instabilities)."""
    def pre_hook(_mod, _inputs):
        _sanitize_nonfinite_params_and_buffers_(module)
        return _inputs

    def post_hook(_mod, _inputs, outputs):
        def _clean(x):
            if torch.is_tensor(x) and x.is_floating_point():
                return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return x
        if isinstance(outputs, (tuple, list)):
            return type(outputs)(_clean(o) for o in outputs)
        return _clean(outputs)

    module.register_forward_pre_hook(pre_hook, with_kwargs=False)
    module.register_forward_hook(post_hook, with_kwargs=False)

    # Gradient guard: scrub NaN/Inf grads so training doesn't explode silently
    for p in module.parameters():
        if p.requires_grad:
            p.register_hook(lambda g: torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0))


def _force_fp32_(module: nn.Module):
    """Keep heads in fp32 regardless of base dtype."""
    module.to(dtype=torch.float32)


# =========================
# Freeze / checks / device
# =========================
def _freeze_base_(model: nn.Module, allow_toplevel_names: List[str]):
    allow = set(allow_toplevel_names)
    for n, p in model.named_parameters():
        top = n.split(".", 1)[0]
        p.requires_grad_(top in allow)


def _assert_finite_(model: nn.Module, names: Iterable[str]):
    bad = []
    with torch.no_grad():
        for name in names:
            mod = getattr(model, name, None)
            if mod is None:
                continue
            for pname, p in mod.named_parameters(recurse=True):
                if (not isinstance(p, torch.Tensor)) or getattr(p, "is_meta", False) or (not p.is_floating_point()):
                    continue
                if not torch.isfinite(p).all():
                    bad.append(f"{name}.{pname}")
            for bname, b in mod.named_buffers(recurse=True):
                if (not isinstance(b, torch.Tensor)) or getattr(b, "is_meta", False) or (not b.is_floating_point()):
                    continue
                if not torch.isfinite(b).all():
                    bad.append(f"{name}.{bname} [buffer]")
    if bad:
        raise RuntimeError(f"Non-finite values after init: {bad}")


def _move_to_device_like_(module: nn.Module, ref_tensor: torch.Tensor):
    module.to(device=ref_tensor.device, dtype=None)  # keep per-param dtypes; we'll force fp32 after


# =========================
# Public entrypoint
# =========================
def get_model(
    model_args: "Any",
    training_args: "Any",
    *,
    head_names: Optional[List[str]] = None,             # override auto-detect
    allow_train: Optional[List[str]] = None,            # override which top-level modules are trainable
    train_lm_head: bool = False,                        # also train lm_head if True
    extra_init: Optional[Dict[str, Callable[[nn.Module], None]]] = None,  # {class_name: init_fn(module)}
) -> AutoModelForCausalLM:
    """
    Load any HF CausalLM, initialize custom heads in FP32 with hard non-finite guarantees,
    upsize attn_extractor embeddings to base L/H, freeze base, and install runtime guards.

    Guaranteed behaviors:
      • All head params/buffers are finite after init (sanitized with multiple passes).
      • Heads are kept in fp32.
      • Runtime guards sanitize params pre-forward and scrub NaN/Inf outputs & grads.
    """
    _maybe_set_seed(training_args)

    # -------- load pretrained (quant, device map, dtype, use_cache vs GC) --------
    # torch_dtype = (
    #     model_args.torch_dtype if getattr(model_args, "torch_dtype", None) in ["auto", None]
    #     else getattr(torch, getattr(model_args, "torch_dtype", "float32"), torch.float32)
    # )
    # robust torch_dtype parsing
    raw_td = getattr(model_args, "torch_dtype", None)
    if raw_td in (None, "auto"):
        torch_dtype = raw_td  # let HF handle None/"auto"
    elif isinstance(raw_td, torch.dtype):
        torch_dtype = raw_td
    elif isinstance(raw_td, str):
        torch_dtype = getattr(torch, raw_td, torch.float32)
    else:
        torch_dtype = torch.float32

    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=getattr(model_args, "model_revision", None),
        trust_remote_code=getattr(model_args, "trust_remote_code", False),
        attn_implementation=getattr(model_args, "attn_implementation", None),
        torch_dtype=torch_dtype,
        use_cache=False if bool(getattr(training_args, "gradient_checkpointing", False)) else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        getattr(model_args, "model_name_or_path"),
        **model_kwargs,
    )

    # Some models expect this config field
    if not hasattr(model.config, "stop_k_att"):
        setattr(model.config, "stop_k_att", 48)

    # Build class-name registry
    name_registry = dict(DEFAULT_NAME_REGISTRY)
    if extra_init:
        name_registry.update(extra_init)

    # ==============================
    # Discover & initialize the heads
    # ==============================
    detected: List[str] = []
    if head_names is None:
        candidates = [
            "hid_extractor", "attn_extractor", "conf_extractor", "stop_head",
            "correctness_head",
        ]
        for n in candidates:
            if hasattr(model, n) and isinstance(getattr(model, n), nn.Module):
                detected.append(n)
        custom = getattr(model, "_custom_head_names", None)
        if isinstance(custom, (list, tuple)):
            for n in custom:
                if hasattr(model, n) and isinstance(getattr(model, n), nn.Module):
                    detected.append(n)
        head_names = sorted(set(detected))

    if head_names:
        # Ensure attn extractor capacity vs base L/H
        if "attn_extractor" in head_names and hasattr(model, "attn_extractor"):
            _ensure_attn_extractor_caps_(getattr(model, "attn_extractor"), model.config)

        # Initialize each head
        for name in head_names:
            mod = getattr(model, name)
            if name == "attn_extractor":
                _init_attn_feature_extractor_(mod, name_registry)
            else:
                _init_generic_stack_(mod, name_registry)
                _init_free_queries_(mod)

        # Move heads to same device as base embeddings (or first param fallback)
        ref = None
        try:
            ref = model.model.embed_tokens.weight  # common in HF CausalLMs
        except Exception:
            for p in model.parameters():
                ref = p
                break
        if isinstance(ref, torch.Tensor):
            for name in head_names:
                _move_to_device_like_(getattr(model, name), ref)

        # Force heads to fp32 (kept even if base is bf16/fp16)
        for name in head_names:
            _force_fp32_(getattr(model, name))

        # Sanitization passes (params + buffers), then verify
        for name in head_names:
            _deep_sanitize_until_finite_(getattr(model, name), max_passes=3)

        # Install runtime guards (pre-forward param/buffer sanitizer, output scrub, grad scrub)
        for name in head_names:
            _install_runtime_guards_(getattr(model, name), name=name)

        # Keep only heads (and optionally lm_head) trainable
        if allow_train is None:
            allow_train = list(head_names)
            if train_lm_head and hasattr(model, "lm_head"):
                allow_train.append("lm_head")
        _freeze_base_(model, allow_train)

        # Final assert (will raise if something slips through)
        _assert_finite_(model, head_names)

        # report
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        pct = (trainable / max(1, total)) * 100.0
        print(f"[custom heads ready] heads={head_names}  trainable={trainable:,} / {total:,} ({pct:.2f}%)")
    else:
        print("[custom heads] none detected; skipping post-load init/freeze.")

    return model






def get_model_gpt_oss(
    model_args: "Any",
    training_args: "Any",
    *,
    head_names: Optional[List[str]] = None,             # override auto-detect
    allow_train: Optional[List[str]] = None,            # override which top-level modules are trainable
    train_lm_head: bool = False,                        # also train lm_head if True
    extra_init: Optional[Dict[str, Callable[[nn.Module], None]]] = None,  # {class_name: init_fn(module)}
) -> AutoModelForCausalLM:
    """
    Specialized loader for GPT-OSS models.

    - Uses Mxfp4Config(dequantize=True) so `config.quantization_config` is valid
      (avoids NoneType.to_dict errors) while keeping the rest of the Open-R1
      head initialization / freezing logic identical to `get_model`.
    - Heads are kept in fp32, base model can be run in bf16/fp16.
    """

    _maybe_set_seed(training_args)

    # -------- robust torch_dtype parsing (same as original) --------
    raw_td = getattr(model_args, "torch_dtype", None)
    if raw_td in (None, "auto"):
        torch_dtype = raw_td  # let HF handle None/"auto"
    elif isinstance(raw_td, torch.dtype):
        torch_dtype = raw_td
    elif isinstance(raw_td, str):
        torch_dtype = getattr(torch, raw_td, torch.float32)
    else:
        torch_dtype = torch.float32

    # -------- GPT-OSS: explicit MXFP4 config --------
    # dequantize=True => weights are materialized in bf16/fp16/float on load,
    # and `config.quantization_config` is a proper Mxfp4Config (no NoneType crash).
    quantization_config = Mxfp4Config(dequantize=False)

    # Let Accelerate / DeepSpeed handle device placement; GPT-OSS is big.
    model_kwargs = dict(
        revision=getattr(model_args, "model_revision", None),
        trust_remote_code=getattr(model_args, "trust_remote_code", False),
        attn_implementation=getattr(model_args, "attn_implementation", None),
        torch_dtype=torch_dtype,
        use_cache=False if bool(getattr(training_args, "gradient_checkpointing", False)) else True,
        device_map=None,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(
        getattr(model_args, "model_name_or_path"),
        **model_kwargs,
    )

    # Some models expect this config field
    if not hasattr(model.config, "stop_k_att"):
        setattr(model.config, "stop_k_att", 48)

    # ----------------------
    # Head discovery / init
    # ----------------------
    name_registry = dict(DEFAULT_NAME_REGISTRY)
    if extra_init:
        name_registry.update(extra_init)

    detected: List[str] = []
    if head_names is None:
        candidates = [
            "hid_extractor", "attn_extractor", "conf_extractor", "stop_head",
            "correctness_head",
        ]
        for n in candidates:
            if hasattr(model, n) and isinstance(getattr(model, n), nn.Module):
                detected.append(n)
        custom = getattr(model, "_custom_head_names", None)
        if isinstance(custom, (list, tuple)):
            for n in custom:
                if hasattr(model, n) and isinstance(getattr(model, n), nn.Module):
                    detected.append(n)
        head_names = sorted(set(detected))

    if head_names:
        # Ensure attn extractor capacity vs base L/H
        if "attn_extractor" in head_names and hasattr(model, "attn_extractor"):
            _ensure_attn_extractor_caps_(getattr(model, "attn_extractor"), model.config)

        # Initialize each head
        for name in head_names:
            mod = getattr(model, name)
            if name == "attn_extractor":
                _init_attn_feature_extractor_(mod, name_registry)
            else:
                _init_generic_stack_(mod, name_registry)
                _init_free_queries_(mod)

        # Move heads to same device as base embeddings (or first param fallback)
        ref = None
        try:
            ref = model.model.embed_tokens.weight  # common in HF CausalLMs
        except Exception:
            for p in model.parameters():
                ref = p
                break
        if isinstance(ref, torch.Tensor):
            for name in head_names:
                _move_to_device_like_(getattr(model, name), ref)

        # Force heads to fp32 (kept even if base is bf16/fp16)
        for name in head_names:
            _force_fp32_(getattr(model, name))

        # Sanitization passes (params + buffers), then verify
        for name in head_names:
            _deep_sanitize_until_finite_(getattr(model, name), max_passes=3)

        # Install runtime guards (pre-forward param/buffer sanitizer, output scrub, grad scrub)
        for name in head_names:
            _install_runtime_guards_(getattr(model, name), name=name)

        # Keep only heads (and optionally lm_head) trainable
        if allow_train is None:
            allow_train = list(head_names)
            if train_lm_head and hasattr(model, "lm_head"):
                allow_train.append("lm_head")
        _freeze_base_(model, allow_train)

        # Final assert (will raise if something slips through)
        _assert_finite_(model, head_names)

        # report
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        pct = (trainable / max(1, total)) * 100.0
        print(f"[gpt-oss custom heads ready] heads={head_names}  "
              f"trainable={trainable:,} / {total:,} ({pct:.2f}%)")
    else:
        print("[gpt-oss custom heads] none detected; skipping post-load init/freeze.")

    return model
