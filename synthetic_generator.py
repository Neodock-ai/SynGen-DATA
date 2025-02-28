"""
synthetic_generator.py

This module provides an industry-level function to generate synthetic data
using an AI-based method (CTGAN from the SDV library, version 1.x).
The function `generate_synthetic_data` accepts an original dataset as a pandas DataFrame
and a target number of synthetic rows, then returns a new DataFrame containing
synthetic data that mirrors the original dataset's structure and distribution.
"""

import logging
import time
import pandas as pd
from sdv.single_table import CTGAN  # Updated import for SDV 1.x

# Configure logging to capture detailed production-level information.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def generate_synthetic_data(data: pd.DataFrame, num_rows: int) -> pd.DataFrame:
    """
    Generate synthetic data using CTGAN.

    This function analyzes the input DataFrame, trains a CTGAN model to capture
    the underlying distribution, and then samples synthetic data that resembles 
    the original data.

    Parameters:
        data (pd.DataFrame): The original dataset as a pandas DataFrame.
        num_rows (int): The desired number of synthetic rows to generate.

    Returns:
        pd.DataFrame: A new DataFrame containing synthetic data.

    Raises:
        ValueError: If the input parameters are invalid.
        Exception: For any errors encountered during model training or sampling.
    """
    
    # Validate input parameters
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("Input DataFrame is empty. Please provide valid data.")
    if not isinstance(num_rows, int) or num_rows <= 0:
        raise ValueError("Number of synthetic rows must be a positive integer.")
    
    try:
        start_time = time.time()
        logging.info("Starting synthetic data generation using CTGAN.")
        
        # Initialize the CTGAN model.
        # Adjust 'epochs' as needed based on your dataset's complexity.
        model = CTGAN(epochs=300, verbose=True)
        logging.info("CTGAN model initialized. Beginning training...")

        # Train the CTGAN model on the original data.
        model.fit(data)
        elapsed = time.time() - start_time
        logging.info("CTGAN model training completed in %.2f seconds.", elapsed)
        
        # Sample the synthetic data.
        synthetic_data = model.sample(num_rows)
        logging.info("Synthetic data generated with shape: %s", synthetic_data.shape)
        
        return synthetic_data

    except Exception as ex:
        logging.error("Error during synthetic data generation: %s", ex)
        raise ex
