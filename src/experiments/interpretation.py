from typing import Tuple
from torch import Tensor
import torch as t
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer
from transformer_lens import ActivationCache
import numpy as np

def head_focused_or_diffuse(heads: list[tuple[int,int]],
                            q_pos: Float[Tensor, "batch"],
                            cache: ActivationCache,
                            model: HookedTransformer,
                            name_pos: Float[Tensor, "batch"],
                            verb_pos: Float[Tensor, "batch"],
                            last_pos: Float[Tensor, "batch"],
                            topk_threshold=0.4):
  results = {}
  for layer, head in heads:
    org_attn_pattern = cache["pattern", layer][t.arange(q_pos.size(0)), head, q_pos] # batch, seqK

    #Removemos el token inicial, sumamos todos los valores del batch y posteriormente normalizamos
    attn_pattern = org_attn_pattern[:, 1:]
    # Sum without the first token
    den = attn_pattern.sum(dim=-1, keepdim=True)
    # Avoid division by zero
    den = den + 1e-12
    # Renormalize
    attn_renorm = attn_pattern / den

    # attn para name, verb, last
    name_attn_pattern = attn_renorm[t.arange(name_pos.size(0)), name_pos-1] # batch
    verb_attn_pattern = attn_renorm[t.arange(verb_pos.size(0)), verb_pos-1] # batch
    last_attn_pattern = attn_renorm[t.arange(last_pos.size(0)), last_pos-1] # batch

    positions = ["name", "verb", "last"]
    attn = t.tensor([ name_attn_pattern.mean().item(), verb_attn_pattern.mean().item(), last_attn_pattern.mean().item() ])
    
    # se extraen el top 2 tokens que más se les ponen atención
    sorted_vals, sorted_idx = t.sort(attn, descending=True)
    
    top1_frac = sorted_vals[0].item()
    top2_frac = sorted_vals[:2].sum().item()

    # --- Clasificación ---
    if top1_frac > topk_threshold:
        tipo = "focused"
        position = positions[sorted_idx[0]]

    elif top2_frac > topk_threshold:
        tipo = "bi-focused"
        position = f"{positions[sorted_idx[0]]}-{positions[sorted_idx[1]]}"
    else:
        tipo = "diffuse"
        position = "na"
    
    results[(layer, head)] = {
        "type": tipo,
        "top1_frac": top1_frac,
        "top2_frac": top2_frac,
        "position": position,
    }

  return results

def calculate_entropy(val_components: Float[Tensor, "seq"]):
  entropy_max = t.log(t.tensor(len(val_components))).item()
  entropy = -t.sum(val_components * t.log(val_components + 1e-12)).item()
  return np.round(entropy / entropy_max, 6)

