"""Data loading utilities."""
import pandas as pd


def load_data(filename, target_column='ram', ignore_columns=None):
    """
    Load data from CSV file.

    Args:
        filename: Path to CSV file
        target_column: Name of the target column to predict
        ignore_columns: List of column names to drop from the dataset

    Returns:
        DataFrame with data

    Raises:
        Exception: If file cannot be loaded or target column not found
    """
    if ignore_columns is None:
        ignore_columns = []

    # Load CSV
    df = pd.read_csv(filename)

    # Verify target column exists
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in data. Available columns: {list(df.columns)}")

    # Drop ignored columns
    columns_to_drop = [col for col in ignore_columns if col in df.columns and col != target_column]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        print(f"Ignored columns: {columns_to_drop}")

    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Target column: '{target_column}'")

    return df
