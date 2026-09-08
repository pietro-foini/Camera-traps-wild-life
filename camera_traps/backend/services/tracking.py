import pandas as pd


def smooth_labels(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Consolidate tracking labels by selecting the most confident class per track.

    | frame_id | tracker_id | xmin  | ymin  | xmax  | ymax  | confidence | label |
    |----------|------------|-------|-------|-------|-------|------------|-------|
    |   ----   |    ----    | ----- | ----- | ----- | ----- | ---------- | ----- |

    :param df: DataFrame containing raw frame-by-frame tracking detections
    :type df: pd.DataFrame
    :param threshold: minimum confidence threshold for score aggregation
    :type threshold: pd.DataFrame
    :return: processed DataFrame with consolidated labels and re-indexed tracker IDs
    :rtype: pd.DataFrame
    """

    if df.empty:
        return df

    labels = (
        df[df["confidence"] >= threshold]
        .groupby(["tracker_id", "label"])["confidence"]
        .sum()
        .groupby(level=0)
        .idxmax()
        .str[1]
    )

    df["label"] = df["tracker_id"].map(labels)
    df.dropna(subset=["label"], inplace=True)

    # Remove non-relevant/background class.
    df = df[df["label"] != "None_of_the_above"]

    if df.empty:
        return df

    # Reset tracker IDs to sequential integers (0, 1, 2, ...).
    tracker_ids = {old: new for new, old in enumerate(df["tracker_id"].unique())}
    df["tracker_id"] = df["tracker_id"].map(tracker_ids)

    return df
