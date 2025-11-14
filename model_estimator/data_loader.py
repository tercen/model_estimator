"""Data loading utilities."""
import pandas as pd


def load_data(filename, target_column='ram'):
    """
    Load data from CSV, Excel, or ODS file.

    Args:
        filename: Path to data file (.csv, .xlsx, .ods)
        target_column: Name of the target column to predict

    Returns:
        DataFrame with data, or None on error
    """
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

        print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        print(f"Target column: '{target_column}'")

        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None
