from jaxtyping import Float
from typing import Callable, List
from torch import Tensor
import torch as t
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer
from transformer_lens import ActivationCache
from itertools import product
import transformer_lens.utils as utils
from functools import partial
from tqdm import tqdm

from .mlp_patching import patch_or_freeze_mlp_vectors
from .head_patching import patch_or_freeze_head_vectors

################ Path patching implementation for attention heads and MLPs (component to component)

def patch_head_input(
    orig_activation: Float[Tensor, "batch pos head_idx d_head"],
    hook: HookPoint,
    patched_cache: ActivationCache,
    head_list: list[tuple[int, int]],
) -> Float[Tensor, "batch pos head_idx d_head"]:
    """
    Function which can patch any combination of heads in layers,
    according to the heads in head_list.
    """
    heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
    orig_activation[:, :, heads_to_patch] = patched_cache[hook.name][:, :, heads_to_patch]
    return orig_activation

def patch_mlp_input(
    orig_activation: Float[Tensor, "batch pos d_mlp"],
    hook: HookPoint,
    patched_cache: ActivationCache,
    mlp_list: list[tuple[int, int]],
) -> Float[Tensor, "batch pos d_mlp"]:
    """
    Function which can patch any combination of heads in layers,
    according to the heads in mlp_list.
    """
    neurons_to_patch = [head for layer, head in mlp_list if layer == hook.layer()]
    orig_activation[:, :, neurons_to_patch] = patched_cache[hook.name][:, :, neurons_to_patch]
    return orig_activation

def get_path_patch_component_to_component(
    from_component: str,
    to_component: str,
    receiver_components: list[tuple[int, int]],
    receiver_input: str,
    model: HookedTransformer,
    patching_metric: Callable,
    answer_tokens: Float[Tensor, "batch 2"],
    corrupted_logit_diff: float,
    clean_logit_diff: float,
    seq: Float[Tensor, "batch"],
    corr_tokens: List | None,
    tokens: List | None,
    neuron_batch: int = 96,
    new_cache: ActivationCache | None = None,
    orig_cache: ActivationCache | None = None,
) -> Float[Tensor, "layer head"]:
    """
    Returns:
        tensor of metric values for every possible sender head
    """
    model.reset_hooks()
    assert receiver_input in ("k", "q", "v", "mlp_post")
    assert from_component in ("head", "mlp")
    assert to_component in ("head", "mlp")

    # Extract all layers to be analyzed
    receiver_layers = set(next(zip(*receiver_components)))
    # Extract all hook names per layer i.e. 'hook.8.v'
    receiver_hook_names = [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
    # Filter by all these v components (all v-components in all selected layers)
    receiver_hook_names_filter = lambda name: name in receiver_hook_names

    if from_component == "head":
      results = t.zeros(max(receiver_layers), model.cfg.n_heads, device="cuda", dtype=t.float32)
    else:
      results = t.zeros(max(receiver_layers), model.cfg.d_mlp // neuron_batch, device="cuda", dtype=t.float32)

    # ========== Step 1 ==========
    # Gather activations on x_orig and x_new

    # Note the use of names_filter for the run_with_cache function. Using it means we
    # only cache the things we need (in this case, just attn head outputs).
    name_filter = lambda name: name.endswith("z") if from_component == "head" else name.endswith("hook_post")

    if new_cache is None:
        _, new_cache = model.run_with_cache(corr_tokens, names_filter=name_filter, return_type=None)
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(tokens, names_filter=name_filter, return_type=None)

    if from_component == "head":
      components = list(product(range(max(receiver_layers)), range(model.cfg.n_heads)))
    else:
      components = list(product(range(max(receiver_layers)), range(0, model.cfg.d_mlp, neuron_batch)))

    # Note, the sender layer will always be before the final receiver layer, otherwise there will
    # be no causal effect from sender -> receiver. So we only need to loop this far.
    for sender_layer, sender_component in tqdm(components):
        # ========== Step 2 ==========
        # Run on x_orig, with sender head patched from x_new, every other head frozen

        if from_component == "head":
          hook_fn = partial(
              patch_or_freeze_head_vectors,
              new_cache=new_cache,
              orig_cache=orig_cache,
              head_to_patch=(sender_layer, sender_component),
          )
        else:
          hook_fn = partial(
              patch_or_freeze_mlp_vectors,
              new_cache=new_cache,
              orig_cache=orig_cache,
              neurons_to_patch=(sender_layer, [n for n in range(sender_component, sender_component + neuron_batch)]),
          )

        model.add_hook(name_filter, hook_fn, level=1) # type: ignore

        _, patched_cache = model.run_with_cache(
            tokens, names_filter=receiver_hook_names_filter, return_type=None
        )
        # model.reset_hooks(including_permanent=True)
        assert set(patched_cache.keys()) == set(receiver_hook_names)

        # ========== Step 3 ==========
        # Run on x_orig, patching in the receiver node(s) from the previously cached value

        if to_component == "head":
          hook_fn = partial(
            patch_head_input,
            patched_cache=patched_cache,
            head_list=receiver_components,
          )
        else:
          hook_fn = partial(
              patch_mlp_input,
              patched_cache=patched_cache,
              mlp_list=receiver_components,
          )

        patched_logits = model.run_with_hooks(
            tokens, fwd_hooks=[(receiver_hook_names_filter, hook_fn)], return_type="logits"
        )

        # Save the results
        if from_component == "head":
          results[sender_layer, sender_component] = patching_metric(patched_logits, answer_tokens, corrupted_logit_diff, clean_logit_diff, seq )
        else:
          results[sender_layer, sender_component // neuron_batch] = patching_metric(patched_logits, answer_tokens, corrupted_logit_diff, clean_logit_diff, seq )

    return results
