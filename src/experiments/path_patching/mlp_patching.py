from itertools import product
from tqdm import tqdm
from jaxtyping import Float
from typing import Callable, List
from torch import Tensor
import torch as t
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer
from transformer_lens import ActivationCache
import transformer_lens.utils as utils
from functools import partial
import numpy as np

################ Path patching implementation for MLPs

def patch_or_freeze_mlp_vectors(
    orig_mlp_vector: Float[Tensor, "batch pos d_mlp"],
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    neurons_to_patch: tuple[int, list[int]],
) -> Float[Tensor, "batch pos d_mlp"]:
    """
    This helps implement step 2 of path patching. We freeze all head outputs (i.e. set them to their values in
    orig_cache), except for head_to_patch (if it's in this layer) which we patch with the value from new_cache.

    head_to_patch: tuple of (layer, head)
    """
    # Setting using ..., otherwise changing orig_mlp_vector will edit cache value too
    orig_mlp_vector[...] = orig_cache[hook.name][...]
    # Patching head
    if neurons_to_patch[0] == hook.layer():
      orig_mlp_vector[:, :, neurons_to_patch[1]] = new_cache[hook.name][:, :, neurons_to_patch[1]] # We patched the original with the new (corrupted)

    return orig_mlp_vector


def get_path_patch_mlp_to_final_resid_post(
    model: HookedTransformer,
    patching_metric: Callable,
    neuron_batch_size: int,
    new_cache: ActivationCache | None,
    orig_cache: ActivationCache | None,
    answer_tokens: Float[Tensor, "batch 2"],
    corrupted_logit_diff: float,
    clean_logit_diff: float,
    seq: Float[Tensor, "batch"],
    corr_tokens: List | None,
    tokens: List | None,
) -> Float[Tensor, "layer head"]:
    """
    Performs path patching (see algorithm in appendix B of IOI paper), with:

        sender head = (each head, looped through, one at a time)
        receiver node = final value of residual stream

    Returns:
        tensor of metric values for every possible sender head
    """
    model.reset_hooks()
    results = t.zeros(model.cfg.n_layers, model.cfg.d_mlp // neuron_batch_size, device="cuda", dtype=t.float32)

    resid_post_hook_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1) # getting last layer
    resid_post_name_filter = lambda name: name == resid_post_hook_name
    mlp_name_filter = lambda name: name.endswith("hook_post") # z means attn_pattern * W_V

    # ========== Step 1 ==========
    # Gather activations on x_orig and x_new

    # Note the use of names_filter for the run_with_cache function. Using it means we
    # only cache the things we need (in this case, just attn head outputs).

    if new_cache is None:
        _, new_cache = model.run_with_cache(corr_tokens, names_filter=mlp_name_filter, return_type=None) # corrupted_cache
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(tokens, names_filter=mlp_name_filter, return_type=None) # clean_cache

    # Looping over every possible sender head (the receiver is always the final resid_post)
    for sender_layer, sender_neurons in tqdm(list(product(range(model.cfg.n_layers), range(0, model.cfg.d_mlp, neuron_batch_size)))):
        # ========== Step 2 ==========
        # Run on x_orig, with sender head patched from x_new, every other head frozen

        hook_fn = partial(
            patch_or_freeze_mlp_vectors,
            new_cache=new_cache,
            orig_cache=orig_cache,
            neurons_to_patch=(sender_layer, [n for n in range(sender_neurons, sender_neurons + neuron_batch_size)]),
        )
        model.add_hook(mlp_name_filter, hook_fn, level=1) # type: ignore

        _, patched_cache = model.run_with_cache(
            tokens, names_filter=resid_post_name_filter, return_type=None
        )

        #assert set(patched_cache.keys()) == {resid_post_hook_name}

        # ========== Step 3 ==========
        # Unembed the final residual stream value, to get our patched logits

        patched_logits = model.unembed(model.ln_final(patched_cache[resid_post_hook_name]))

        # Save the results
        results[sender_layer, sender_neurons // neuron_batch_size] = patching_metric(patched_logits,
                                                                                     answer_tokens,
                                                                                     corrupted_logit_diff,
                                                                                     clean_logit_diff,
                                                                                     seq)

    return results


### Para verificar por neurona
def get_path_patch_neuron_mlp_to_final_resid_post(
    model: HookedTransformer,
    patching_metric: Callable,
    neuron_block: tuple[int, int],
    new_cache: ActivationCache | None,
    orig_cache: ActivationCache | None,
    answer_tokens: Float[Tensor, "batch 2"],
    corrupted_logit_diff: float,
    clean_logit_diff: float,
    seq: Float[Tensor, "batch"],
    corr_tokens: List | None,
    tokens: List | None,
    neuron_size: int = 96
) -> Float[Tensor, "layer head"]:
    """
    Performs path patching (see algorithm in appendix B of IOI paper), with:

        sender head = (each head, looped through, one at a time)
        receiver node = final value of residual stream

    Returns:
        tensor of metric values for every possible sender head
    """
    model.reset_hooks()
    results = t.zeros(neuron_size, device="cuda", dtype=t.float32)

    resid_post_hook_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1) # getting last layer
    resid_post_name_filter = lambda name: name == resid_post_hook_name
    mlp_name_filter = lambda name: name.endswith("hook_post") # z means attn_pattern * W_V

    # ========== Step 1 ==========
    # Gather activations on x_orig and x_new

    # Note the use of names_filter for the run_with_cache function. Using it means we
    # only cache the things we need (in this case, just attn head outputs).

    if new_cache is None:
        _, new_cache = model.run_with_cache(corr_tokens, names_filter=mlp_name_filter, return_type=None) # corrupted_cache
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(tokens, names_filter=mlp_name_filter, return_type=None) # clean_cache

    init_range = neuron_block[1] * neuron_size
    end_range  = (neuron_block[1] + 1) * neuron_size
    sender_layer = neuron_block[0]

    # Looping over every possible sender head (the receiver is always the final resid_post)
    for sender_neuron in tqdm(range(init_range, end_range)):
        # ========== Step 2 ==========
        # Run on x_orig, with sender head patched from x_new, every other head frozen

        hook_fn = partial(
            patch_or_freeze_mlp_vectors,
            new_cache=new_cache,
            orig_cache=orig_cache,
            neurons_to_patch=(sender_layer, [sender_neuron]),
        )
        model.add_hook(mlp_name_filter, hook_fn, level=1) # type: ignore

        _, patched_cache = model.run_with_cache(
            tokens, names_filter=resid_post_name_filter, return_type=None
        )

        #assert set(patched_cache.keys()) == {resid_post_hook_name}

        # ========== Step 3 ==========
        # Unembed the final residual stream value, to get our patched logits

        patched_logits = model.unembed(model.ln_final(patched_cache[resid_post_hook_name]))

        # Save the results
        results[sender_neuron - (neuron_block[1] * neuron_size)] = patching_metric(patched_logits,
                                                                                     answer_tokens,
                                                                                     corrupted_logit_diff,
                                                                                     clean_logit_diff,
                                                                                     seq)

    return results

def head_focused_or_diffuse(heads: list[tuple[int,int]],
                            q_pos:int,
                            cache: ActivationCache,
                            model: HookedTransformer,
                            tokens: List,
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