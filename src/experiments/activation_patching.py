import torch as t
import transformer_lens.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from transformer_lens import HookedTransformer
from transformer_lens import ActivationCache
from functools import partial
from jaxtyping import Float
from typing import Callable
from torch import Tensor
from tqdm import tqdm


def patch_mlp_neurons(
    corrupted_mlp_vector: Float[Tensor, "batch pos d_mlp"],
    hook,
    neuron_indices: list[int],
    clean_cache: ActivationCache,
    pos_by_layer: dict[int, list[int]]
) -> Float[Tensor, "batch pos d_mlp"]:
    """
    Patches a subset of MLP neurons from the clean cache into the corrupted run.
    """
    positions = pos_by_layer[hook.layer] if hook.layer in pos_by_layer else []
    if len(positions) == 0:
      corrupted_mlp_vector[:, :, neuron_indices] = clean_cache[hook.name][:, :, neuron_indices]
    else:
      for pos in positions:
        corrupted_mlp_vector[:, pos, neuron_indices] = clean_cache[hook.name][:, pos, neuron_indices]
    return corrupted_mlp_vector


def get_act_patch_mlp_neurons(
    model: HookedTransformer,
    corrupted_tokens: Float[Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Callable,
    dataset: dict,
    clean_logit_diff: float,
    corrupted_logit_diff: float,
    component: str = "mlp_post",
    neuron_step: int = 96
) -> Float[Tensor, "layer"]:
    """
    Patches subsets of MLP neurons across all layers.
    If neuron_indices=None, patches all neurons (equivalent to full MLP patch).
    Returns results[layer] = effect of patching on metric.
    """
    dim = model.cfg.d_mlp if component == "mlp_post" else model.cfg.d_model

    results = t.zeros((model.cfg.n_layers, dim // neuron_step), dtype=t.float32)
    model.reset_hooks()

    for mlp_layer in range(model.cfg.n_layers):
      step = 0
      for neuron_idx in tqdm(range(0, dim, neuron_step)):

          hook_fn = partial(
              patch_mlp_neurons,
              neuron_indices=list(range(neuron_idx, neuron_idx + neuron_step)),
              clean_cache=clean_cache,
          )

          patched_logits = model.run_with_hooks(
              corrupted_tokens,
              fwd_hooks=[(utils.get_act_name(component, mlp_layer), hook_fn)],
              return_type="logits",
          )

          results[mlp_layer, step] = patching_metric(patched_logits,
                                                     dataset["answer_ids"],
                                                     corrupted_logit_diff,
                                                     clean_logit_diff,
                                                     dataset["last_token_pos"])
          step += 1
          model.reset_hooks()

    return results

def patch_mlp_single_neuron(
    corrupted_mlp_vector: Float[Tensor, "batch pos d_mlp"],
    hook,
    neuron_idx: int,
    clean_cache: ActivationCache,
    pos_by_layer: dict[int, list[int]]
) -> Float[Tensor, "batch pos d_mlp"]:
    """
    Patches a subset of MLP neurons from the clean cache into the corrupted run.
    """
    positions = pos_by_layer[hook.layer] if hook.layer in pos_by_layer else []
    if len(positions) == 0:
      corrupted_mlp_vector[:, :, neuron_idx] = clean_cache[hook.name][:, :, neuron_idx]
    else:
      for pos in positions:
        corrupted_mlp_vector[:, pos, neuron_idx] = clean_cache[hook.name][:, pos, neuron_idx]

    return corrupted_mlp_vector


def get_act_patch_mlp_by_neuron(
    model: HookedTransformer,
    corrupted_tokens: Float[Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Callable,
    dataset: dict,
    clean_logit_diff: float,
    corrupted_logit_diff: float,
    layer: int,
    neuron_idx_list: list = []
) -> Float[Tensor, "layer"]:
    """
    Patches subsets of MLP neurons across all layers.
    If neuron_indices=None, patches all neurons (equivalent to full MLP patch).
    Returns results[layer] = effect of patching on metric.
    """
    results = t.zeros(len(neuron_idx_list), dtype=t.float32)
    model.reset_hooks()

    for idx, neuron_idx in tqdm(enumerate(neuron_idx_list)):
      hook_fn = partial(
          patch_mlp_single_neuron,
          neuron_idx=neuron_idx,
          clean_cache=clean_cache,
      )

      patched_logits = model.run_with_hooks(
          corrupted_tokens,
          fwd_hooks=[(utils.get_act_name("mlp_post", layer), hook_fn)],
          return_type="logits",
      )

      results[idx] = patching_metric(patched_logits, dataset["answer_ids"], corrupted_logit_diff, clean_logit_diff, dataset["last_token_pos"])

      model.reset_hooks()

    return results

def get_most_important_neurons(neuron_blocks: list,
                               model: HookedTransformer,
                               corrupted_tokens: Float[Tensor, "batch pos"],
                               clean_cache: ActivationCache,
                               patching_metric: Callable,
                               dataset: dict,
                               clean_logit_diff: float,
                               corrupted_logit_diff: float,
                               plot_heatmap: bool = True
                               ) -> dict:
    """
    Identifies and returns the most important neurons in specified MLP blocks of a transformer model using activation patching.
    Args:
        neuron_blocks (list): List of tuples (layer, block) specifying which MLP blocks to analyze.
        model (HookedTransformer): The transformer model to analyze.
        corrupted_tokens (Float[Tensor, "batch pos"]): Input tokens with corruption applied.
        clean_cache (ActivationCache): Activation cache from a clean (uncorrupted) forward pass.
        patching_metric (Callable): Function to compute the metric for activation patching.
        dataset (dict): Dataset used for the analysis.
        clean_logit_diff (float): Logit difference for the clean input.
        corrupted_logit_diff (float): Logit difference for the corrupted input.
        plot_heatmap (bool, optional): Whether to plot a heatmap of neuron importance. Defaults to True.
    Returns:
        dict: A dictionary mapping (layer, neuron_index) to their corresponding importance values, as determined by activation patching.
    """
    results = {}

    for layer, block in neuron_blocks:
        neuron_idx_list = [nidx for nidx in range( (3072 // 32) * block, (3072 // 32) * (block + 1) )]

        mlp_single_activations = get_act_patch_mlp_by_neuron(
            model=model,
            corrupted_tokens=corrupted_tokens,
            clean_cache=clean_cache,
            patching_metric=patching_metric,
            dataset=dataset,
            clean_logit_diff=clean_logit_diff,
            corrupted_logit_diff=corrupted_logit_diff,
            layer=layer,
            neuron_idx_list=neuron_idx_list
        )

        mlp_single_activations = mlp_single_activations.unsqueeze(0).cpu().detach().numpy()
        values = np.round(mlp_single_activations[0], 2)
        indices = np.nonzero(values)[0]
        for i in indices:
            results[ (layer, neuron_idx_list[i]) ] = values[i]
        
        if plot_heatmap:
            print(f"layer:{layer} - block:{block}")
            plt.figure(figsize=(18, 2))
            sns.heatmap(mlp_single_activations, annot=False, fmt=".2f", 
                        cmap="RdBu", center=0, xticklabels=neuron_idx_list, yticklabels=[layer]) # type: ignore
            plt.title("Activation Patching by MLP neuron")
            plt.xlabel("Neuron subset")
            plt.xticks(rotation=-90)
            plt.ylabel("Layer Number")
            plt.tight_layout()
            plt.show()

    return results

def project_neurons_to_logit_diff_dir(neuron_set, cache, pos_by_layer, model: HookedTransformer):
    """
    Project neurons' post-MLP activations into vocabulary logit-space for selected sequence positions
    and print the top-k most influenced tokens per neuron/position.
    Parameters
    ----------
    neuron_set : Mapping[tuple[int, int], int]
        Mapping from (layer, neuron_index) -> sign indicator (any value; treated as positive if > 0,
        negative otherwise). The function iterates over the keys (layer, neuron_index) to select neurons.
    cache : Mapping[str, torch.Tensor]
        Activation cache (as produced by a HookedTransformer run). Expected to contain the
        'mlp_post' activations for each layer under the key returned by utils.get_act_name('mlp_post', layer).
        Each activation tensor is expected to have shape [batch, seq_len, n_neurons].
    pos_by_layer : Mapping[int, Sequence[int]]
        Mapping from layer index -> sequence indices of interest. For each neuron (layer, neuron_index),
        the routine inspects the positions listed in pos_by_layer[layer].
    model : HookedTransformer
        A model instance that must expose:
            - blocks[layer].mlp.W_out (and blocks[layer].mlp.b_out)
            - W_U (the unembedding matrix of shape [d_model, vocab_size])
            - to_string(token_id) for decoding token ids to strings
    Behavior
    --------
    For each neuron identified in neuron_set and for each sequence position s in pos_by_layer[layer]:
        1. Read the neuron's post-MLP activation across the batch at position s from cache.
            (neuron_vec_seq has shape [batch]).
        2. Read the corresponding MLP output projection vector W_out[neuron_index]
            (shape [d_model]) and the MLP output bias b_out (shape [d_model]).
        3. Compute the neuron's contribution to logits via:
            logits_direction = (neuron_activation[:, None] @ neuron_w_out[None, :] + b_out) @ W_U
            which yields a tensor of shape [batch, vocab_size].
        4. For each batch item, take the top-k token indices by value (k=10), decode them
            with model.to_string, and accumulate them into a set for the tuple
            (sign, layer, neuron_index, seq).
        5. After processing all batches, print one summary line per (sign, layer, neuron_index, seq)
            containing the union of the top tokens across the batch.
    Returns
    -------
    None
        The function prints the results to stdout and does not return the constructed mapping.
        Internally a mapping of results is created (neuron_results), but it is not returned.
    Notes
    -----
    - The function relies on torch's topk operation for selecting top tokens.
    - Expected tensor shapes:
        - cache[utils.get_act_name('mlp_post', layer)]: [batch, seq_len, n_neurons]
        - neuron_vec_seq: [batch]
        - neuron_w_out_vec: [d_model]
        - W_U: [d_model, vocab_size]
        - logits_direction: [batch, vocab_size]
    - Top-k is currently fixed at k=10.
    - Side effects: prints lines of the form:
        "{POS|NEG} - L:{layer}.{neuron_index} - Seq:{s} - Tokens: {set_of_tokens}"
    """

    neurons = [ ("POS" if neuron_set[key] > 0 else "NEG", key[0], key[1], pos_by_layer[key[0]]) for key in neuron_set ]

    neuron_results = {}

    for positive, layer, neuron_idx, seq in neurons:
        # 1. Obtener W_out de la MLP
        W_out = model.blocks[layer].mlp.W_out # type: ignore

        # 2. Unembedding
        W_U = model.W_U  # shape [d_model, vocab_size]

        # 3. Obtener la neurona
        neuron_vec = cache[utils.get_act_name('mlp_post', layer)][:, :, neuron_idx]  # shape [batch seq]

        for s in seq:
            # 4. selecciona la seq de interés
            neuron_vec_seq = neuron_vec[:, s]  # shape [batch]

            # 5. Vector de salida de la neurona
            neuron_w_out_vec = W_out[neuron_idx]  # type: ignore # shape [d_model]

            # 6. Proyectar neurona -> vocab
            logits_direction = ((neuron_vec_seq.unsqueeze(1) @ neuron_w_out_vec.unsqueeze(0)) + model.blocks[layer].mlp.b_out) @ W_U  # type: ignore # shape [batch, vocab_size]

            for batch_idx in range(logits_direction.shape[0]):

                # 7. Obtener tokens top-k más influidos
                topk_vals, topk_idx = t.topk(logits_direction[batch_idx], k=10)

                # Decodificar tokens
                top_tokens = set([model.to_string(tid.item()) for tid in topk_idx]) # type: ignore

                if (positive, layer, neuron_idx, s) not in neuron_results:
                    neuron_results[(positive, layer, neuron_idx, s)] = top_tokens
                else:
                    neuron_results[(positive, layer, neuron_idx, s)] = neuron_results[(positive, layer, neuron_idx, s)].union(top_tokens)

        for (positive, layer, neuron_idx, s), top_tokens in neuron_results.items():
            print(f"{positive} - L:{layer}.{neuron_idx} - Seq:{s} - Tokens: {top_tokens}")

