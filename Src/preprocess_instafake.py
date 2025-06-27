import pandas as pd
import numpy as np
import os
from pathlib import Path

# Get relative path to the root directory
ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = Path(ROOT).parent  # Step up to the parent directory
DATA_DIR = ROOT / "Dataset" / "InstaFake" / "raw"

# Set seed for reproducibility
np.random.seed(42)


def check_path(file_path: Path):
    """
    Check if the file exists and load it into a DataFrame.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")


def find_json_files(directory: Path, name: str) -> str:
    """
    Find JSON files in the specified directory that match the given name.
    """
    json_files = list(directory.rglob(f"{name}"))
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found with name '{name}' in {directory}."
        )
    return str(json_files[0])  # Return the first matching file path


def load_json(file_path: str, safe_load: bool = True) -> pd.DataFrame:
    """
    Load a JSON file into a DataFrame, handling errors gracefully.
    """
    file_path = Path(file_path)
    if safe_load:
        check_path(file_path)
    try:
        df = pd.read_json(file_path, lines=False)
        print(f"Data loaded successfully with {len(df)} records.")
        return df
    except ValueError as e:
        print(f"Error loading data: {e}")
        raise e


def preprocess_data(df: pd.DataFrame) -> None:
    # TODO: Implement preprocessing steps
    pass


if __name__ == "__main__":
    print("Checking for dependency files...")
    # Check if the data directory exists
    check_path(DATA_DIR)
    print(f"Data directory exists: {DATA_DIR}")
    # Declaring dependency files
    FAKE_ACCOUNT_FILES = "fakeAccountData.json"
    REAL_ACCOUNT_FILES = "realAccountData.json"

    print("Loading fake account data...")
    fake_file_path = find_json_files(DATA_DIR, FAKE_ACCOUNT_FILES)
    fake_account_df = load_json(fake_file_path, safe_load=True)

    print("Loading real account data...")
    real_file_path = find_json_files(DATA_DIR, REAL_ACCOUNT_FILES)
    real_account_df = load_json(real_file_path, safe_load=True)

    print("Combining datasets...")
    combined_df = pd.concat([fake_account_df, real_account_df], ignore_index=True)
    # Shuffle the combined DataFrame to ensure randomness
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    print(f"Combined dataset has {len(combined_df)} records.")
    print('Saving combined dataset to "Dataset/interim/combined_account_data.csv"...')

    # Check if the interim directory exists, create it if not
    interim_dir = ROOT / "Dataset" / "InstaFake" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(interim_dir / "combined_account_data.csv", index=False)
    print("Combined dataset saved successfully.")
