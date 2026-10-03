"""
Dataset Loader and Streamer for the VeReMi Vehicular Misbehavior Dataset.
"""

import os
from typing import Optional, Iterator
import pandas as pd

class VeReMiDatasetLoader:
    """Streamlined loader for the 7.47 GB VeReMi CSV dataset."""
    def __init__(self, dataset_path: str):
        self.path = dataset_path
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"VeReMi dataset not found at: {dataset_path}")

    def load_samples(self, n_rows: int = 100000) -> pd.DataFrame:
        """Loads a fixed number of rows from the dataset head."""
        return pd.read_csv(self.path, nrows=n_rows)

    def stream_chunks(self, chunk_size: int = 50000) -> Iterator[pd.DataFrame]:
        """Yields chunks of the dataset for memory-efficient out-of-core processing."""
        for chunk in pd.read_csv(self.path, chunksize=chunk_size):
            yield chunk

    def get_dataset_info(self) -> dict:
        """Returns dataset file size and column schema."""
        size_bytes = os.path.getsize(self.path)
        sample = pd.read_csv(self.path, nrows=5)
        return {
            "path": self.path,
            "size_gb": size_bytes / (1024**3),
            "columns": list(sample.columns),
            "num_features": len(sample.columns)
        }
