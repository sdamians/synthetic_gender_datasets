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

################ Path patching implementation for attention heads

def patch_or_freeze_head_vectors(
    orig_head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: tuple[int, int],
) -> Float[Tensor, "batch pos head_index d_head"]:
    """
    This helps implement step 2 of path patching. We freeze all head outputs (i.e. set them to their values in
    orig_cache), except for head_to_patch (if it's in this layer) which we patch with the value from new_cache.

    head_to_patch: tuple of (layer, head)
    """
    # Setting using ..., otherwise changing orig_head_vector will edit cache value too
    orig_head_vector[...] = orig_cache[hook.name][...]
    # Patching head
    if head_to_patch[0] == hook.layer():
        orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]] # We patched the original with the new (corrupted)
    return orig_head_vector


def get_path_patch_head_to_final_resid_post(
    model: HookedTransformer,
    patching_metric: Callable,
    new_cache: ActivationCache | None,
    orig_cache: ActivationCache | None,
    answer_tokens: Float[Tensor, "batch 2"],
    corrupted_logit_diff: float,
    clean_logit_diff: float,
    seq: Float[Tensor, "batch"],
    corr_tokens: List | None,
    tokens: List | None
) -> Float[Tensor, "layer head"]:
    """
    Performs path patching (see algorithm in appendix B of IOI paper), with:

        sender head = (each head, looped through, one at a time)
        receiver node = final value of residual stream

    Returns:
        tensor of metric values for every possible sender head
    """
    model.reset_hooks()
    results = t.zeros(model.cfg.n_layers, model.cfg.n_heads, device="cuda", dtype=t.float32)

    resid_post_hook_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1) # getting last layer
    resid_post_name_filter = lambda name: name == resid_post_hook_name
    z_name_filter = lambda name: name.endswith("z") # z means attn_pattern * W_V

    # ========== Step 1 ==========
    # Gather activations on x_orig and x_new

    # Note the use of names_filter for the run_with_cache function. Using it means we
    # only cache the things we need (in this case, just attn head outputs).

    if new_cache is None:
        _, new_cache = model.run_with_cache(corr_tokens, names_filter=z_name_filter, return_type=None) # corrupted_cache
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(tokens, names_filter=z_name_filter, return_type=None) # clean_cache

    # Looping over every possible sender head (the receiver is always the final resid_post)
    for sender_layer, sender_head in tqdm(list(product(range(model.cfg.n_layers), range(model.cfg.n_heads)))):
        # ========== Step 2 ==========
        # Run on x_orig, with sender head patched from x_new, every other head frozen

        hook_fn = partial(
            patch_or_freeze_head_vectors,
            new_cache=new_cache,
            orig_cache=orig_cache,
            head_to_patch=(sender_layer, sender_head),
        )
        model.add_hook(z_name_filter, hook_fn, level=1) # type: ignore

        _, patched_cache = model.run_with_cache(
            tokens, names_filter=resid_post_name_filter, return_type=None
        )

        #assert set(patched_cache.keys()) == {resid_post_hook_name}

        # ========== Step 3 ==========
        # Unembed the final residual stream value, to get our patched logits

        patched_logits = model.unembed(model.ln_final(patched_cache[resid_post_hook_name]))

        # Save the results
        results[sender_layer, sender_head] = patching_metric(patched_logits,
                                                             answer_tokens,
                                                             corrupted_logit_diff,
                                                             clean_logit_diff,
                                                             seq)

    return results