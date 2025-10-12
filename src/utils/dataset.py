import pandas as pd
from transformer_lens import HookedTransformer

def gather_data(filename: str, 
                name_col: str, 
                gender_col: str, 
                count_col: str, 
                new_names: list)-> pd.DataFrame:
    """
    Groups data by name and gender to count the most popular names.
    Args:
        filename: Full path to the raw data CSV file
        name_col: Name column
        gender_col: Gender column
        count_col: Count column
        new_names: List of new column names
    Returns: 
        pd.DataFrame: DataFrame grouped by name-gender, sorted in descending order
    """
    df = pd.read_csv(filename)
    df.drop_duplicates(inplace=True)
    df[name_col] = df[name_col].apply(lambda x: x.capitalize())
    df[gender_col] = df[gender_col].apply(lambda x: x[0])
    
    df_agg = (df
          .groupby([name_col, gender_col])
          .agg({ count_col: "sum" })
          .reset_index()
          .sort_values(by=count_col, ascending=False)
         )
    
    if new_names is not None and len(new_names) == len(df_agg.columns):
        df_agg.columns = new_names

    return df_agg 


def to_token_ids(model: HookedTransformer, name: str, prepend_bos=False) -> str:
    """
    Converts a given name string into a comma-separated string of token IDs using the provided model.

    Args:
        model (HookedTransformer): The language model used for tokenization.
        name (str): The name to be tokenized.

    Returns:
        str: A comma-separated string of token IDs representing the input name.
    """
    tokens = model.to_tokens(f" {name}", prepend_bos=prepend_bos)[0].tolist()
    return ",".join([str(t) for t in tokens])