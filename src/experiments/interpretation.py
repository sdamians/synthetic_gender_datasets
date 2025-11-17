from typing import Tuple
from torch import Tensor
import torch as t
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer
from transformer_lens import ActivationCache
import numpy as np

def head_focused_or_diffuse(heads: list[tuple[int,int]],
                            q_pos:int,
                            cache: ActivationCache,
                            model: HookedTransformer,
                            name_pos: int,
                            verb_pos: int,
                            last_pos: int,
                            example_prompt_ids: list,
                            topk_thresholds=(0.4,0.8)):
  results = {}
  for layer, head in heads:
    attn_pattern = cache["pattern", layer][:, head, q_pos] # batch, seqK

    #Removemos el token inicial, sumamos todos los valores del batch y posteriormente normalizamos
    attn_pattern = attn_pattern[:, 1:].sum(dim=0)
    attn_pattern = attn_pattern / attn_pattern.sum()

    seq_len = len(attn_pattern)
    entropy_max = t.log(t.tensor(seq_len)).item()

    # entropia alta >1.5 = el pattern es focused
    entropy = -t.sum(attn_pattern * t.log(attn_pattern + 1e-12)).item()
    entropy = entropy / entropy_max

    # se extraen el top 2 tokens que más se les ponen atención
    sorted_vals, sorted_idx = t.sort(attn_pattern, descending=True)
    sorted_idx = sorted_idx + 1
    top1_frac = sorted_vals[0].item()
    top2_frac = sorted_vals[:2].sum().item()

    # --- Clasificación ---
    if top1_frac > topk_thresholds[0]:
        tipo = "focused"
        posiciones_relevantes = [sorted_idx[0].item()]
        valores_relevantes = [sorted_vals[0].item()]
    elif top2_frac > topk_thresholds[0]:
        tipo = "bi-focused"
        posiciones_relevantes = sorted_idx[:2].tolist()
        valores_relevantes = sorted_vals[:2].tolist()
    else:
        tipo = "diffuse"
        posiciones_relevantes = sorted_idx[:3].tolist()
        valores_relevantes = sorted_vals[:3].tolist()

    key_pos = { name_pos: "name", verb_pos: "verb", last_pos: "last" }
    rel_pos = [ key_pos[p] if p in key_pos else model.to_string(example_prompt_ids[p]).strip() for p in posiciones_relevantes ] # type: ignore

    results[(layer, head)] = {
        "tipo": tipo,
        "entropy": entropy,
        "top1_frac": top1_frac,
        "top2_frac": top2_frac,
        "posiciones_relevantes": rel_pos,
        "valores_relevantes": valores_relevantes,
    }
  return results

def calculate_entropy(val_components: Float[Tensor, "seq"]):
  entropy_max = t.log(t.tensor(len(val_components))).item()
  entropy = -t.sum(val_components * t.log(val_components + 1e-12)).item()
  return np.round(entropy / entropy_max, 6)

