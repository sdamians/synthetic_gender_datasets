from typing import Tuple, List
from torch import Tensor
import torch as t
from jaxtyping import Float, Bool
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer
from transformer_lens import ActivationCache
import transformer_lens.utils as utils
import einops
from functools import partial

# =====================================================================================
# Mean ablation implementation for heads
# =====================================================================================

def get_heads_and_posns_to_keep(
    dataset_len: int,
    max_len: int,
    model: HookedTransformer,
    circuit: dict[str, list[tuple[int, int]]],
    seq_pos_to_keep: dict[str, Float[Tensor, "batch"]],
    inverse_circuit: bool
) -> dict[int, Bool[Tensor, "batch seq head"]]:
    """
    Returns a dictionary mapping layers to a boolean mask giving the indices of the z output which *shouldn't* be
    mean-ablated.

    The output of this function will be used for the hook function that does ablation.
    """
    heads_and_posns_to_keep = {}
    batch, seq, n_heads = dataset_len, max_len, model.cfg.n_heads

    for layer in range(model.cfg.n_layers):
        mask = t.zeros(size=(batch, seq, n_heads))

        for head_type, head_list in circuit.items():
            indices = seq_pos_to_keep[head_type]
            # indices = means_dataset.word_idx[seq_pos]
            for layer_idx, head_idx in head_list:
                if layer_idx == layer:
                    mask[:, indices, head_idx] = 1

        heads_and_posns_to_keep[layer] = mask.bool()
        if inverse_circuit:
            heads_and_posns_to_keep[layer] = ~heads_and_posns_to_keep[layer]

    return heads_and_posns_to_keep


def hook_fn_mask_z(
    z: Float[Tensor, "batch seq head d_head"],
    hook: HookPoint,
    heads_and_posns_to_keep: dict[int, Bool[Tensor, "batch seq head"]],
    means: Float[Tensor, "layer batch seq head d_head"]
) -> Float[Tensor, "batch seq head d_head"]:
    """
    Hook function which masks the z output of a transformer head.

    heads_and_posns_to_keep
        dict created with the get_heads_and_posns_to_keep function. This tells us where to mask.

    means
        Tensor of mean z values of the means_dataset over each group of prompts with the same template. This tells us
        what values to mask with.
    """
    # Get the mask for this layer, and add d_head=1 dimension so it broadcasts correctly
    mask_for_this_layer = heads_and_posns_to_keep[hook.layer()].unsqueeze(-1).to(z.device)

    # Set z values to the mean
    # Retrieve z if condition satisfies, else retrieve means
    z = t.where(mask_for_this_layer, z, means[hook.layer()])

    return z


def compute_means_by_template(
    prompt_type_indices: Float[Tensor, "idx"], # type: ignore
    tokens: Float[Tensor, "batch seq"],
    dataset_len: int,
    max_len: int,
    model: HookedTransformer
) -> Float[Tensor, "layer batch seq head_idx d_head"]:
    """
    Returns the mean of each head's output over the means dataset. This mean is computed separately for each group of
    prompts with the same template (these are given by means_dataset.groups).
    """
    # Cache the outputs of every head
    _, means_cache = model.run_with_cache(
        tokens,
        return_type=None,
        names_filter=lambda name: name.endswith("z"),
    )
    # Create tensor to store means
    n_layers, n_heads, d_head = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head
    batch, seq_len = dataset_len, max_len
    means = t.zeros(size=(n_layers, batch, seq_len, n_heads, d_head), device=model.cfg.device)

    # Get set of different templates for this data
    for layer in range(model.cfg.n_layers):
        z_for_this_layer = means_cache[utils.get_act_name("z", layer)]  # [batch seq head d_head]
        for prompt_type_idx in prompt_type_indices:
            z_for_this_template = z_for_this_layer[prompt_type_idx]
            z_means_for_this_template = einops.reduce(
                z_for_this_template, "batch seq head d_head -> seq head d_head", "mean"
            )
            means[layer, prompt_type_idx] = z_means_for_this_template

    return means # [n_layers, prompt_type, seq, head, d_head]


def add_mean_ablation_hook(
    model: HookedTransformer,
    prompt_type_indices: List[Tensor, "idx"], # type: ignore
    tokens: Float[Tensor, "batch seq"],
    dataset_len: int,
    max_len: int,
    circuit: dict[str, list[tuple[int, int]]],
    seq_pos_to_keep: dict[str, str],
    inverse_circuit = False,
    is_permanent: bool = True,
    reset_model: bool = True
) -> HookedTransformer:
    """
    Adds a permanent hook to the model, which ablates according to the circuit and seq_pos_to_keep dictionaries.

    In other words, when the model is run on ioi_dataset, every head's output will be replaced with the mean over
    means_dataset for sequences with the same template, except for a subset of heads and sequence positions as specified
    by the circuit and seq_pos_to_keep dicts.
    """
    if reset_model:
      model.reset_hooks(including_permanent=True)

    # Compute the mean of each head's output on the ABC dataset, grouped by template
    means = compute_means_by_template(prompt_type_indices, tokens, dataset_len, max_len, model) # type: ignore

    # Convert this into a boolean map
    heads_and_posns_to_keep = get_heads_and_posns_to_keep(dataset_len, max_len, model, circuit, seq_pos_to_keep, inverse_circuit) # type: ignore

    # Get a hook function which will patch in the mean z values for each head, at
    # all positions which aren't important for the circuit
    hook_fn = partial(hook_fn_mask_z, heads_and_posns_to_keep=heads_and_posns_to_keep, means=means)

    # Apply hook
    model.add_hook(lambda name: name.endswith("z"), hook_fn, is_permanent=is_permanent) # type: ignore

    return model


# =====================================================================================
# Mean ablation implementation for MLPs
# =====================================================================================

def get_neurons_and_posns_to_keep(
    dataset_len: int,
    max_len: int,
    model: HookedTransformer,
    circuit: dict[str, list[tuple[int, int]]],
    mlp_seq_pos_to_keep: dict[str, Float[Tensor, "batch"]],
    inverse_circuit: bool
) -> dict[int, Bool[Tensor, "batch seq"]]:
    """
    Returns a dictionary mapping layers to a boolean mask giving the indices of the z output which *shouldn't* be
    mean-ablated.

    The output of this function will be used for the hook function that does ablation.
    """
    neurons_and_posns_to_keep = {}
    batch, seq, d_mlp = dataset_len, max_len, model.cfg.d_mlp

    for layer in range(model.cfg.n_layers):
        mask = t.zeros(size=(batch, seq, d_mlp))

        for mlp_type, neuron_list in circuit.items():
            indices = mlp_seq_pos_to_keep[mlp_type]
            # indices = means_dataset.word_idx[seq_pos]
            for layer_idx, neuron_idx in neuron_list:
                if layer_idx == layer:
                    mask[:, indices, neuron_idx] = 1

        neurons_and_posns_to_keep[layer] = mask.bool()
        if inverse_circuit:
            neurons_and_posns_to_keep[layer] = ~neurons_and_posns_to_keep[layer]

    return neurons_and_posns_to_keep


def hook_fn_mask_mlp_post(
    mlp_post: Float[Tensor, "batch seq d_mlp"],
    hook: HookPoint,
    neurons_and_posns_to_keep: dict[int, Bool[Tensor, "batch seq"]],
    means: Float[Tensor, "layer batch seq d_mlp"]
) -> Float[Tensor, "batch seq d_mlp"]:
    """
    Hook function which masks the mlp_post output of a transformer head.

    neurons_and_posns_to_keep
        dict created with the get_neurons_and_posns_to_keep function. This tells us where to mask.

    means
        Tensor of mean d_mlp values of the means_dataset over each group of prompts with the same template. This tells us
        what values to mask with.
    """
    # Get the mask for this layer, and add d_mlp=1 dimension so it broadcasts correctly
    mask_for_this_layer = neurons_and_posns_to_keep[hook.layer()].to(mlp_post.device)

    # Set mlp_post values to the mean
    # Retrieve mlp_post if condition satisfies, else retrieve means
    mlp_post = t.where(mask_for_this_layer, mlp_post, means[hook.layer()])

    return mlp_post


def compute_means_by_template_for_mlp(
    prompt_type_indices: Float[Tensor, "idx"], # type: ignore
    tokens: Float[Tensor, "batch seq"],
    dataset_len: int,
    max_len: int,
    model: HookedTransformer
) -> Float[Tensor, "layer batch seq d_mlp"]:
    """
    Returns the mean of each neurons's output over the means dataset. This mean is computed separately for each group of
    prompts with the same template (these are given by means_dataset.groups).
    """
    # Cache the outputs of every head
    _, means_cache = model.run_with_cache(
        tokens,
        return_type=None,
        names_filter=lambda name: name.endswith("hook_post"),
    )
    # Create tensor to store means
    n_layers, d_mlp = model.cfg.n_layers, model.cfg.d_mlp
    batch, seq_len = dataset_len, max_len
    means = t.zeros(size=(n_layers, batch, seq_len, d_mlp), device=model.cfg.device)

    # Get set of different templates for this data
    for layer in range(n_layers):
        z_for_this_layer = means_cache[utils.get_act_name("mlp_post", layer)]  # [batch seq d_mlp]
        for prompt_type_idx in prompt_type_indices:
            z_for_this_template = z_for_this_layer[prompt_type_idx]
            z_means_for_this_template = einops.reduce(
                z_for_this_template, "batch seq d_mlp -> seq d_mlp", "mean"
            )
            means[layer, prompt_type_idx] = z_means_for_this_template

    return means # [n_layers, seq, d_mlp]


def add_mean_ablation_hook_for_mlp(
    model: HookedTransformer,
    prompt_type_indices: List[Tensor, "idx"], # type: ignore
    tokens: Float[Tensor, "batch seq"],
    dataset_len: int,
    max_len: int,
    circuit: dict[str, list[tuple[int, int]]],
    seq_pos_to_keep: dict[str, str],
    inverse_circuit = False,
    is_permanent: bool = True,
    reset_model: bool = True
) -> HookedTransformer:
    """
    Adds a permanent hook to the model, which ablates according to the circuit and seq_pos_to_keep dictionaries.

    In other words, when the model is run on ioi_dataset, every head's output will be replaced with the mean over
    means_dataset for sequences with the same template, except for a subset of heads and sequence positions as specified
    by the circuit and seq_pos_to_keep dicts.
    """
    if reset_model:
      model.reset_hooks(including_permanent=True)

    # Compute the mean of each head's output on the ABC dataset, grouped by template
    means = compute_means_by_template_for_mlp(prompt_type_indices, tokens, dataset_len, max_len, model) # type: ignore

    # Convert this into a boolean map
    neurons_and_posns_to_keep = get_neurons_and_posns_to_keep(dataset_len, max_len, model, circuit, seq_pos_to_keep, inverse_circuit) # type: ignore

    # Get a hook function which will patch in the mean z values for each head, at
    # all positions which aren't important for the circuit
    hook_fn = partial(hook_fn_mask_mlp_post, neurons_and_posns_to_keep=neurons_and_posns_to_keep, means=means)

    # Apply hook
    model.add_hook(lambda name: name.endswith("hook_post"), hook_fn, is_permanent=is_permanent) # type: ignore

    return model


# =====================================================================================
# Functions for Faithfulness experiments
# =====================================================================================

def patch_head_vector(
    corrupted_head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    head_index: int,
    name: str,
    seq_pos_to_keep: dict[str, Float[Tensor, "batch"]],
    clean_cache: ActivationCache,
) -> Float[Tensor, "batch pos head_index d_head"]:
    """
    Patches the output of a given head (before it's added to the residual stream) at every sequence position, using the
    value from the clean cache.
    """
    positions = seq_pos_to_keep[name]
    corrupted_head_vector[:, positions, head_index] = clean_cache[hook.name][:, positions, head_index]
    return corrupted_head_vector

def patch_mlp_single_neuron(
    corrupted_mlp_vector: Float[Tensor, "batch pos d_mlp"],
    hook,
    neuron_idx: int,
    clean_cache: ActivationCache,
    name: str,
    mlp_seq_pos_to_keep: dict[str, Float[Tensor, "batch"]],
) -> Float[Tensor, "batch pos d_mlp"]:
    """
    Patches a subset of MLP neurons from the clean cache into the corrupted run.
    """
    positions = mlp_seq_pos_to_keep[name]
    if len(positions) == 0:
      corrupted_mlp_vector[:, :, neuron_idx] = clean_cache[hook.name][:, :, neuron_idx]
    else:
      for pos in positions:
        corrupted_mlp_vector[:, positions, neuron_idx] = clean_cache[hook.name][:, positions, neuron_idx]

    return corrupted_mlp_vector

def patching_circuit(
    model: HookedTransformer,
    corrupted_tokens: Float[Tensor, "batch pos"],
    clean_cache: ActivationCache,
    mlp_circuit: dict[str, list[tuple[int, int]]], 
    circuit: dict[str, list[tuple[int, int]]],
    seq_pos_to_keep:  dict[str, Float[Tensor, "batch"]],
    mlp_seq_pos_to_keep:  dict[str, Float[Tensor, "batch"]]
) -> Float[Tensor, "layer head"]:

    model.reset_hooks(including_permanent=True)

    hooks = []

    # Creating hooks for attn heads
    if circuit is not None:
      for key in circuit:
        for layer, head in circuit[key]:
          hook_fn = partial(patch_head_vector, head_index=head, clean_cache=clean_cache, name=key, seq_pos_to_keep=seq_pos_to_keep)
          hooks.append((utils.get_act_name("z", layer), hook_fn))

    # Creating hooks for mlp neurons
    if mlp_circuit is not None:
      for key in mlp_circuit:
        for layer, neuron in mlp_circuit[key]:
          hook_fn = partial(patch_mlp_single_neuron, neuron_idx=neuron, clean_cache=clean_cache, name=key, mlp_seq_pos_to_keep=mlp_seq_pos_to_keep)
          hooks.append((utils.get_act_name("mlp_post", layer), hook_fn))

    patched_logits = model.run_with_hooks(
          corrupted_tokens,
          fwd_hooks=hooks,
          return_type="logits"
      )

    return patched_logits