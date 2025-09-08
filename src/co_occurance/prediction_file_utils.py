#!/usr/bin/env python3
"""
Utility functions for loading prediction files with FDR format preference.

This module provides helper functions to automatically try FDR-calibrated predictions
first, then fallback to original uncertainty interval predictions if needed.
"""

import os
import pickle
import logging
from typing import Dict, Any, Optional


def get_prediction_file_path(base_dir: str, ontology_subdir: str, 
                           prefer_fdr: bool = True, logger: Optional[logging.Logger] = None) -> str:
    """Get the path to the best available prediction file."""
    fdr_file = os.path.join(base_dir, ontology_subdir, 'predicted_terms_fdr.pkl')
    original_file = os.path.join(base_dir, ontology_subdir, 'predicted_terms.pkl')
    
    if prefer_fdr and os.path.exists(fdr_file):
        if logger:
            logger.info(f"Using FDR-calibrated predictions: {fdr_file}")
        return fdr_file
    elif os.path.exists(original_file):
        if logger:
            if prefer_fdr:
                logger.warning(f"FDR predictions not found, using original format: {original_file}")
            else:
                logger.info(f"Using original predictions: {original_file}")
        return original_file
    else:
        raise FileNotFoundError(f"No prediction files found in {base_dir}/{ontology_subdir}/")


def load_predictions_with_fdr_preference(base_dir: str, ontology_subdir: str,
                                       prefer_fdr: bool = True, 
                                       logger: Optional[logging.Logger] = None) -> Dict[Any, Any]:
    """Load predictions with preference for FDR format."""
    pred_file = get_prediction_file_path(base_dir, ontology_subdir, prefer_fdr, logger)
    
    with open(pred_file, 'rb') as f:
        predictions = pickle.load(f)
    
    # Log format information
    if logger:
        if 'fdr' in os.path.basename(pred_file):
            logger.info(f"Loaded FDR predictions: {len(predictions)} FDR levels")
            sample_keys = sorted(list(predictions.keys()))[:5]
            logger.info(f"FDR levels available: {sample_keys}...")
        else:
            logger.info(f"Loaded confidence threshold predictions: {len(predictions)} threshold levels")
            sample_keys = sorted(list(predictions.keys()))[:5]
            logger.info(f"Confidence thresholds available: {sample_keys}...")
    
    return predictions


def get_ontology_prediction_files(predictions_dir: str, prefer_fdr: bool = True,
                                logger: Optional[logging.Logger] = None) -> Dict[str, str]:
    """Get prediction file paths for all ontologies (BP, MF, CC)."""
    ontology_mapping = {
        'BP': 'process_test_predictions_and_FDR_NEW/bpo',
        'MF': 'function_test_predictions_and_FDR_NEW/mfo', 
        'CC': 'component_test_predictions_and_FDR_NEW/cco'
    }
    
    prediction_files = {}
    
    for ontology, subdir in ontology_mapping.items():
        try:
            pred_file = get_prediction_file_path(predictions_dir, subdir, prefer_fdr, logger)
            prediction_files[ontology] = pred_file
        except FileNotFoundError as e:
            if logger:
                logger.error(f"Failed to find predictions for {ontology}: {e}")
            raise
    
    return prediction_files


# Backward compatibility function names
def get_best_prediction_file(base_dir: str, ontology_subdir: str, 
                           logger: Optional[logging.Logger] = None) -> str:
    """Backward compatibility alias for get_prediction_file_path with FDR preference."""
    return get_prediction_file_path(base_dir, ontology_subdir, prefer_fdr=True, logger=logger)


if __name__ == "__main__":
    print("Prediction file utilities for co-occurrence analysis")
    print("Provides automatic FDR format preference with fallback to original format")
