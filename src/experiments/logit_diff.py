from jaxtyping import Float
from typing import Tuple, Any
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens import ActivationCache
import einops

def get_logit_diff_direction(answer_ids: Float[Tensor, "batch 2"], 
                             model: HookedTransformer
) -> Float[Tensor, "batch d_model"]:
  # Getting expected and unexpected token values from W_U (the unembedding matrix)
  answer_residual_directions = model.tokens_to_residual_directions(answer_ids) # [batch 2 d_model]

  # Split them into correct and incorrect
  correct_residual_directions, incorrect_residual_directions = answer_residual_directions.split(answer_ids.size(-1) // 2, dim=1)

  # Getting their difference (logit difference)
  logit_diff_directions = correct_residual_directions.sum(dim=1) - incorrect_residual_directions.sum(dim=1)  # [batch d_model]

  return logit_diff_directions

def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"],
    cache: ActivationCache,
    logit_diff_directions: Float[Tensor, "batch d_model"],
) -> Float[Tensor, "..."]:
    """
    Gets the avg logit difference between the correct and incorrect answer for a given stack of components in the
    residual stream.
    """
    # Getting batch size
    batch_size = residual_stack.size(-2)
    # Multiplying the residual stream value by the normalization step
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    # X_i * W_U
    return (
        einops.einsum(scaled_residual_stack, logit_diff_directions, "... batch d_model, batch d_model -> ...")
        / batch_size
    )

def get_accumulated_logit_diffs(cache: ActivationCache, 
                                logit_diff_directions: Float[Tensor, "batch d_model"]
) -> Tuple[Float[Tensor, "component"], Any]:
    """
    Calculates accumulated logit differences for each component in the residual stream.
    This function processes the accumulated values of the residual stream and computes
    logit differences based on given directions.
    Args:
        cache: A cache object containing accumulated residual values from the model.
        logit_diff_directions: Directions used to calculate logit differences.
    Returns:
        tuple:
            - logit_lens_logit_diffs (Tensor): Logit differences per component
            - labels: Labels associated with the accumulated residuals
    Shapes:
        - accumulated_residual: [number of components, batch, d_model]
        - logit_lens_logit_diffs: [component]
    """
    # Getting all accumulated values of the residual stream
    # accumulated_residual shape: [number of components, batch, d_model]
    accumulated_residual, labels = cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)

    # Getting the logit diff per component
    logit_lens_logit_diffs: Float[Tensor, "component"] = residual_stack_to_logit_diff(accumulated_residual, cache, logit_diff_directions)
    return logit_lens_logit_diffs, labels

def get_logit_diffs_per_layer(cache: ActivationCache, 
                              logit_diff_directions: Float[Tensor, "batch d_model"]
) -> Tuple[Float[Tensor, "batch d_model"], Any]:
    """
    Calculates the logit differences for each layer based on the provided activation cache and logit difference directions.
    Args:
        cache (ActivationCache): An instance of ActivationCache containing the residuals and other necessary data.
        logit_diff_directions (Float[Tensor, "batch d_model"]): A tensor representing the directions for logit differences.
    Returns:
        Tuple[Float[Tensor, "batch d_model"], Any]: A tuple containing:
            - per_layer_logit_diffs (Float[Tensor, "batch d_model"]): The computed logit differences for each layer.
            - labels (Any): The labels associated with the computed logit differences.
    """
    per_layer_residual, labels = cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
    per_layer_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, cache, logit_diff_directions)

    return per_layer_logit_diffs, labels

def get_logit_diffs_per_head(cache: ActivationCache, 
                             logit_diff_directions: Float[Tensor, "batch d_model"], 
                             n_layers: int
) -> Tuple[Float[Tensor, "layer head batch"], Any]:
    """
    Calculates logit differences per attention head using cached model outputs.

    This function processes cached attention head outputs to compute logit differences
    based on provided directions. It stacks and reshapes the residual outputs from
    each head and computes the corresponding logit differences.

    Parameters
    ----------
    cache : TransformerCache
        Cache object containing model's intermediate attention outputs
    logit_diff_directions : torch.Tensor
        Tensor containing the directions for computing logit differences
    n_layers : int
        Number of transformer layers in the model

    Returns
    -------
    tuple
        - per_head_logit_diffs (torch.Tensor): Computed logit differences for each head
        - labels (torch.Tensor): Corresponding labels from the cache
    """
    per_head_residual, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
    per_head_residual = einops.rearrange(per_head_residual, "(layer head) ... -> layer head ...", layer=n_layers)
    per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual, cache, logit_diff_directions)
    return per_head_logit_diffs, labels