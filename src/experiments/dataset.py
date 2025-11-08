import torch as t
from datasets import Dataset
from typing import Any, Dict, List, Optional, Union

ans_tokens_dict: Dict[str, int] = {" she": 673, " he": 339}
ans_token_id_dict: Dict[int, str] = {673: " she", 339: " he"}


def get_nested_feature_values(dataset: Dataset, levels: List[str]) -> List[Any]:
  """
  Return the nested feature values from a HuggingFace Dataset by following the given levels.

  Parameters:
  - dataset (Dataset): A HuggingFace Dataset or a nested mapping returned from it.
  - levels (List[str]): Sequence of keys to index into the nested structure (e.g. ['prompts', 'org_prompt']).

  Returns:
  - List[Any]: The extracted values for all rows at the specified nested key.
  """
  for level in levels:
    dataset = dataset[level] # type: ignore

  return dataset[: len(dataset)] # type: ignore


def get_opposite_gender(value: Union[str, int], return_id: bool = True) -> Union[int, str]:
  """
  Return the opposite gender token/id.

  Parameters:
  - value (str|int): Either a token string ('she'/'he' or with leading space) or an integer token id.
  - return_id (bool): If True, return the token id (int). If False, return the token string (str).

  Returns:
  - int or str: The opposite token id or token string depending on return_id.
  """
  id_map = {673: 339, 339: 673}
  token_map = {"she": "he", "he": "she", " she": " he", " he": " she"}

  if return_id:
    if isinstance(value, int):
      return id_map.get(value, value)
    # try to resolve from token string to id
    val_str = str(value)
    # accept with or without leading space
    if val_str in ("she", "he"):
      val_str = " " + val_str
    return ans_tokens_dict.get(val_str, value)
  else:
    if isinstance(value, int):
      return ans_token_id_dict.get(value, value)
    return token_map.get(str(value), value)


def tokenize_function(instance: Dict[str, Any]) -> Dict[str, Any]:
  """
  Compute positions and expected/unexpected answer tokens for a single dataset instance.

  The function:
  - Determines the expected and unexpected token ids and token strings.
  - Computes the verb token position based on the prompt structure.
  - Returns a dict with:
    'verb_token_pos', 'ans_token_ids', 'ans_tokens', and 'gender'.

  Parameters:
  - instance (Dict[str, Any]): A dataset row containing keys like 'expected_token_id',
    'prompts', 'end', and 'subject'.

  Returns:
  - Dict[str, Any]: Computed fields for the instance.
  """
  exp_tok_id = int(instance["expected_token_id"])
  unexp_tok_id = int(get_opposite_gender(exp_tok_id, return_id=True))

  ans_token_ids = [exp_tok_id, unexp_tok_id]
  ans_tokens = [ans_token_id_dict[exp_tok_id], ans_token_id_dict[unexp_tok_id]]

  verb_token_pos = (
    int(instance["end"]["pos"]) - 1
    if "(" in instance["prompts"]["org_prompt"]
    else int(instance["subject"]["pos"][0]) + 1
  )

  return {
    "verb_token_pos": verb_token_pos,
    "ans_token_ids": ans_token_ids,
    "ans_tokens": ans_tokens,
    "gender": "female" if exp_tok_id == 673 else "male",
  }


def get_dataset(
  subset: Dataset,
  device: Union[t.device, str],
  gender_study: Optional[str] = None,
  prompt_type_list: Optional[List[int]] = None,
  data_prop: Optional[float] = None,
) -> Dict[str, Any]:
  """
  Prepare and return tensors and prompt/answer lists extracted from a Dataset.

  Parameters:
  - subset (Dataset): The dataset to extract from.
  - device (torch.device | str): Device to move tensors to.
  - gender_study (Optional[str]): If provided, filter by this gender ('female'/'male').
  - prompt_type_list (Optional[List[int]]): If provided, keep only rows whose prompt_type is in this list.
    If None, defaults to [1].
  - data_prop (Optional[float]): If provided, perform a train_test_split and keep the 'test' subset
    of size data_prop (stratified by prompt_type).

  Returns:
  - Dict[str, Any]: A mapping with prompts, answers, token positions and tensors moved to device.
  """
  if data_prop is not None:
    subset = subset.train_test_split(test_size=data_prop, stratify_by_column="prompt_type", seed=43)["test"]

  if prompt_type_list is None:
    prompt_type_list = [1]
  subset = subset.filter(lambda x: x["prompt_type"] in prompt_type_list)

  if gender_study is not None:
    subset = subset.filter(lambda x: x["gender"] == gender_study)

  return {
    "prompts": get_nested_feature_values(subset, ["prompts", "org_prompt"]),
    "c_prompts": get_nested_feature_values(subset, ["prompts", "corr_prompt"]),
    "a_prompts": get_nested_feature_values(subset, ["prompts", "ablated_prompt"]),
    "answers": get_nested_feature_values(subset, ["ans_tokens"]),
    "prompt_type": get_nested_feature_values(subset, ["prompt_type"]),
    "answer_ids": t.tensor(subset["ans_token_ids"]).to(device),
    "last_token_pos": t.tensor(get_nested_feature_values(subset, ["end", "pos"])).to(device),
    "subject_pos": t.tensor(get_nested_feature_values(subset, ["subject", "pos"])).squeeze().to(device),
    "verb_pos": t.tensor(subset["verb_token_pos"]).squeeze().to(device),
  }
