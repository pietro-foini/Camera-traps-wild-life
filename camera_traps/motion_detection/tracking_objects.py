import pandas as pd
import numpy as np


def flatten_concatenated_mapping(key, value, dictionary):
    if dictionary.get(value) is not None:
        new_value = dictionary.get(value)
        dictionary[key] = new_value
        return flatten_concatenated_mapping(key, new_value, dictionary)
    else:
        return dictionary


def tracking(df: pd.DataFrame, distance_limit: float = 10):
    """
    Given a DataFrame containing boxes centroids for a set of id frames, the algorithm retrieve the unique IDs of the
    boxes over time (consecutive frames) based on distance condition of the relative centroids. The provided DataFrame
    must contain at least the following columns:

        | id_frame | centroid |
        |----------|----------|
        |  ------  |  ------  |

    :param df: the DataFrame containing boxes' centroid (centroid) and the frame IDs (id_frame)
    :param distance_limit:
    :return:
    """
    # Create unique row id.
    df_centroid = df[["id_frame", "centroid"]].copy()
    df_centroid["index"] = np.arange(0, len(df_centroid))
    # Get all the combination boxes between consecutive frames.
    traker = pd.merge(df_centroid, df_centroid, how="cross", suffixes=(" (t)", " (t-1)"))
    traker.query("`id_frame (t)` - `id_frame (t-1)` == 1", inplace=True)
    # Consider the boxes to be the same if their centroids have a distance smaller than a provided threshold.
    traker["distance"] = traker.apply(lambda x: x["centroid (t)"].distance(x["centroid (t-1)"]), axis=1)
    traker.query("distance <= @distance_limit", inplace=True)

    # Get a dictionary de-concatenating the inner mapping.
    mapping = traker.set_index("index (t)")["index (t-1)"].to_dict()
    for k, v in mapping.items():
        mapping = flatten_concatenated_mapping(k, v, mapping)

    tracking_index = df_centroid["index"].map(mapping).fillna(df_centroid["index"]).astype(int)
    ascending_mapping = dict(zip(tracking_index.unique(), np.arange(0, tracking_index.nunique())))
    tracking_index = tracking_index.map(ascending_mapping)

    return tracking_index
