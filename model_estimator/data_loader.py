"""Data loading utilities."""
import pandas as pd


def load_data(filename, target_column='ram', ignore_columns=None):
    """
    Load data from CSV, Excel, or ODS file.

    Args:
        filename: Path to data file (.csv, .xlsx, .ods)
        target_column: Name of the target column to predict
        ignore_columns: List of column names to drop from the dataset

    Returns:
        DataFrame with data, or None on error
    """
    if ignore_columns is None:
        ignore_columns = []
    try:
        # Determine file type and load accordingly
        if filename.endswith('.csv'):
            df = pd.read_csv(filename)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(filename, engine='openpyxl')
        elif filename.endswith('.ods'):
            df = pd.read_excel(filename, engine='odf')
        else:
            print(f"Error: Unsupported file format. Use .csv, .xlsx, or .ods")
            return None

        # Verify target column exists
        if target_column not in df.columns:
            print(f"Error: Column '{target_column}' not found in data")
            print(f"Available columns: {list(df.columns)}")
            return None

        # Drop ignored columns
        columns_to_drop = [col for col in ignore_columns if col in df.columns and col != target_column]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            print(f"Ignored columns: {columns_to_drop}")

        print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        print(f"Target column: '{target_column}'")

        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None
