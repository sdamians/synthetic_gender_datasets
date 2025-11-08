from jaxtyping import Float
from torch import Tensor
import torch as t


def from_logits_to_avg_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"],
    seq: Float[Tensor, "batch"],
    per_prompt: bool = False,
) -> Float[Tensor, "*batch"]:
    """
    Calculate the logit difference between correct and incorrect answer tokens.

    Args:
        logits (Tensor): Model output logits with shape [batch, sequence, vocabulary]
        answer_tokens (Tensor): Pairs of tokens (correct/incorrect) with shape [batch, 2]
        seq (Tensor): Sequence positions to evaluate with shape [batch]
        per_prompt (bool): If True, return per-example differences instead of mean

    Returns:
        Tensor: Average logit differences if per_prompt=False, otherwise per-example differences
    """
    # Extract logits for answer tokens at specified sequence positions
    answer_logits = logits[t.arange(len(logits))[:, None], seq[:, None], answer_tokens[None, :]].squeeze()
    # Split into correct and incorrect answer logits
    correct_logits, incorrect_logits = answer_logits.split(answer_tokens.size(-1) // 2, dim=-1)


    # Calculate the difference
    diff_logits = correct_logits.sum(dim=-1) - incorrect_logits.sum(dim=-1)
    return diff_logits if per_prompt else diff_logits.mean()

def activation_patching_metric(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"],
    corrupted_logit_diff: float,
    clean_logit_diff: float,
    last_token_pos: Float[Tensor, "batch"]
) -> Float[Tensor, ""]:
    """
    Linear function of logit diff, calibrated so that it equals 0 when performance is same as on corrupted input, and 1
    when performance is same as on clean input.
    """
    patched_logit_diff = from_logits_to_avg_logit_diff(logits, answer_tokens, last_token_pos)
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

def path_patching_metric(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"],
    corrupted_logit_diff: float,
    clean_logit_diff: float,
    seq: Float[Tensor, "batch"]
) -> Float[Tensor, ""]:
    """
    Linear function of logit diff, calibrated so that it equals 0 when performance isn't harmed (same as clean dataset) and -1
    when performance has been destroyed (same as corrupted dataset). If it was destroyed it means the component is important (its influence alters the performance negatively).
    It could also happen the component affected positively, which could mean the original component acted in the opposite way.
    """
    patched_logit_diff = from_logits_to_avg_logit_diff(logits, answer_tokens, seq)
    return (patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
