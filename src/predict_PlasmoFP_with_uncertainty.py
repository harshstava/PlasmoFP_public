#!/usr/bin/env python3
"""
FDR thresholding uses a CLOSEST LOWER FDR approach:
- All threshold selection uses utils_corrected.py with the closest lower FDR logic
- At a given FDR level (e.g., 10%), if no exact threshold exists, use the threshold 
  from the closest lower FDR level (e.g., 9%, 8%, etc.)
- FDR thresholds are applied to effective scores (median - MAD) for uncertainty-aware predictions

EXAMPLE USAGE:

# Predict with both probability and FDR thresholds
python predict_with_uncertainty_and_thresholds.py --fasta proteins.fasta --output_dir predictions \
    --bp_tsv bp_test.tsv --cc_tsv cc_test.tsv --mf_tsv mf_test.tsv

# Use custom discrete FDR levels
python predict_with_uncertainty_and_thresholds.py --fasta proteins.fasta --output_dir predictions \
    --mf_tsv mf_test.tsv --fdr_levels 0.01 0.05 0.10 0.20

# Use continuous FDR range (0.01 to 0.5 in steps of 0.01)
python predict_with_uncertainty_and_thresholds.py --fasta proteins.fasta --output_dir predictions \
    --mf_tsv mf_test.tsv --fdr_range_start 0.01 --fdr_range_end 0.5 --fdr_range_step 0.01

# Skip original probability thresholding, only use FDR
python predict_with_uncertainty_and_thresholds.py --fasta proteins.fasta --output_dir predictions \
    --mf_tsv mf_test.tsv --skip_prob_thresholds
"""

import argparse
import os
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score

from utils_corrected import (
    PFP, predict_without_MCD, store_predicted_terms, 
    calculate_sharpness_coverage_and_fdr, process_GO_data,
    load_kfold_ensembles, compute_ensemble_probs, chunked_mad_over_runs,
    evaluate_annotations, threshold_performance_metrics, calculate_aupr_micro,
    get_term_threshold, build_pred_annots_dict, compute_effective_scores
)

warnings.filterwarnings('ignore')


def setup_logging(output_dir: str, verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logger = logging.getLogger('prediction')
    logger.setLevel(log_level)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    log_file = os.path.join(output_dir, 'prediction_log.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_mlb(ontology_name: str) -> MultiLabelBinarizer:
    mlb_path = f'{ontology_name}_mlb.pkl'
    if not os.path.exists(mlb_path):
        raise FileNotFoundError(f"MultiLabelBinarizer not found: {mlb_path}")
    
    with open(mlb_path, 'rb') as f:
        mlb = pickle.load(f)
    return mlb


def load_ic_dict(ic_path: Optional[str], logger: logging.Logger) -> Optional[Dict[str, float]]:
    if ic_path is None:
        ic_path = 'IA_all.tsv'  # Default IC file
    
    if not os.path.exists(ic_path):
        logger.warning(f"IC file not found: {ic_path}. CAFA metrics will be limited.")
        return None
    
    try:
        ic_df = pd.read_csv(ic_path, sep='\t', header=None, names=['GO_term', 'IC'])
        
        ic_dict = dict(zip(ic_df['GO_term'], ic_df['IC']))
        
        logger.info(f"Loaded IC dictionary with {len(ic_dict)} terms from {ic_path}")
        return ic_dict
    except Exception as e:
        logger.warning(f"Failed to load IC dictionary from {ic_path}: {e}. CAFA metrics will be limited.")
        return None


def load_effective_thresholds(ontology_name: str, logger: logging.Logger) -> Optional[Dict[str, Dict[str, float]]]:
    """Load effective thresholds for given ontology."""
    threshold_file = f'effective_thresholds_ensemble20_{ontology_name}.json'
    
    if not os.path.exists(threshold_file):
        logger.warning(f"Effective thresholds file not found: {threshold_file}")
        return None
    
    try:
        with open(threshold_file, 'r') as f:
            effective_thresholds = json.load(f)
        
        logger.info(f"Loaded effective thresholds for {len(effective_thresholds)} terms from {threshold_file}")
        return effective_thresholds
    except Exception as e:
        logger.warning(f"Failed to load effective thresholds from {threshold_file}: {e}")
        return None


def extract_fdr_thresholds(effective_thresholds: Dict[str, Dict[str, float]], 
                          fdr_level: float) -> Dict[str, float]:
    """Extract thresholds for a specific FDR level using the corrected utils approach."""
    term_thresholds = {}
    
    for term in effective_thresholds.keys():
        threshold = get_term_threshold(term, effective_thresholds, fdr_level)
        if threshold is not None:
            term_thresholds[term] = threshold
    
    return term_thresholds


def apply_fdr_thresholds(effective_scores: np.ndarray, mlb: MultiLabelBinarizer, 
                        fdr_thresholds: Dict[str, float]) -> np.ndarray:
    """Apply term-specific FDR thresholds to effective scores (median - MAD)."""
    predictions = np.zeros_like(effective_scores, dtype=bool)
    
    for i, go_term in enumerate(mlb.classes_):
        if go_term in fdr_thresholds:
            threshold = fdr_thresholds[go_term]
            predictions[:, i] = effective_scores[:, i] >= threshold
    
    return predictions


def evaluate_fdr_predictions(predictions: np.ndarray, true_labels: np.ndarray, 
                           mlb: MultiLabelBinarizer, fdr_level: float, 
                           fdr_thresholds: Dict[str, float], test_GO_list: List[List[str]],
                           ic_dict: Optional[Dict[str, float]]) -> Dict[str, Any]:
        
    n_samples, n_classes = predictions.shape
    
    real_annots_dict = {}
    pred_annots_dict = {}
    
    for i in range(n_samples):
        protein_id = f"protein_{i}"
        
        if i < len(test_GO_list):
            real_annots_dict[protein_id] = set(test_GO_list[i])
        else:
            real_annots_dict[protein_id] = set()
        
        predicted_terms = set()
        for j, is_predicted in enumerate(predictions[i]):
            if is_predicted:
                predicted_terms.add(mlb.classes_[j])
        pred_annots_dict[protein_id] = predicted_terms
    
    if ic_dict is not None:
        f_macro, p_macro, r_macro, s_min, ru, mi, f_micro, p_micro, r_micro, tp_global, fp_global, fn_global = \
            evaluate_annotations(ic_dict, real_annots_dict, pred_annots_dict)
    else:
        precision_micro = precision_score(true_labels, predictions, average='micro', zero_division=0)
        recall_micro = recall_score(true_labels, predictions, average='micro', zero_division=0)
        f1_micro = f1_score(true_labels, predictions, average='micro', zero_division=0)
        
        precision_macro = precision_score(true_labels, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(true_labels, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
        
        f_macro, p_macro, r_macro = f1_macro, precision_macro, recall_macro
        f_micro, p_micro, r_micro = f1_micro, precision_micro, recall_micro
        s_min = ru = mi = None
        tp_global = fp_global = fn_global = None
    
    terms_with_thresholds = len(fdr_thresholds)
    terms_predicted = np.sum(predictions.sum(axis=0) > 0)  # Terms with at least one prediction
    avg_predictions_per_protein = predictions.sum(axis=1).mean()
    
    tp_per_term = ((predictions == 1) & (true_labels == 1)).sum(axis=0)
    fp_per_term = ((predictions == 1) & (true_labels == 0)).sum(axis=0)
    fn_per_term = ((predictions == 0) & (true_labels == 1)).sum(axis=0)
    
    term_precision = np.divide(tp_per_term, tp_per_term + fp_per_term, 
                              out=np.zeros_like(tp_per_term, dtype=float), 
                              where=(tp_per_term + fp_per_term) != 0)
    term_recall = np.divide(tp_per_term, tp_per_term + fn_per_term,
                           out=np.zeros_like(tp_per_term, dtype=float),
                           where=(tp_per_term + fn_per_term) != 0)
    term_f1 = np.divide(2 * term_precision * term_recall, term_precision + term_recall,
                       out=np.zeros_like(term_precision, dtype=float),
                       where=(term_precision + term_recall) != 0)
    
    return {
        'fdr_level': fdr_level,
        'f_macro': f_macro,
        'precision_macro': p_macro,
        'recall_macro': r_macro,
        's_min': s_min,
        'remaining_uncertainty': ru,
        'misinformation': mi,
        'f_micro': f_micro,
        'precision_micro': p_micro,
        'recall_micro': r_micro,
        'tp_global': tp_global,
        'fp_global': fp_global,
        'fn_global': fn_global,
        'terms_with_thresholds': terms_with_thresholds,
        'terms_predicted': int(terms_predicted),
        'total_terms': n_classes,
        'avg_predictions_per_protein': avg_predictions_per_protein,
        'total_predictions': int(predictions.sum()),
        'avg_term_precision': term_precision.mean(),
        'avg_term_recall': term_recall.mean(),
        'avg_term_f1': term_f1.mean()
    }


def load_model_config(ontology: str, config_file: str = 'TUNED_MODEL_ARCHS.json') -> Dict[str, Any]:
    """Load model architecture configuration."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        tuned_model_archs = json.load(f)
    
    if ontology not in tuned_model_archs:
        raise KeyError(f"Ontology {ontology} not found in config file")
    
    return tuned_model_archs[ontology]


def load_ensemble_models(models_dir: str, ontology: str, device: torch.device, 
                        logger: logging.Logger) -> List[torch.nn.Module]:
    """Load 20-fold ensemble models for given ontology."""
    logger.info(f"Loading 20-fold ensemble for {ontology} from {models_dir}")
    
    arch_config = load_model_config(ontology)
    
    model_kwargs = {
        'input_dim': 512,
        'hidden_dims': arch_config['CHOSEN_CONFIG_ARCH'],
        'output_dim': arch_config['NUM_CLASSES'],
        'dropout_rate': 0.3
    }
    
    kf_root = os.path.join(models_dir, "k_folds")
    if not os.path.exists(kf_root):
        raise FileNotFoundError(f"K-fold directory not found: {kf_root}")
    
    ensemble_dict = load_kfold_ensembles(
        kf_root=kf_root,
        fold_options=[20],
        model_cls=PFP,
        model_kwargs=model_kwargs,
        device=device
    )
    
    if 20 not in ensemble_dict:
        raise ValueError(f"20-fold ensemble not found for {ontology}")
    
    models = [model for (_, model) in ensemble_dict[20]]
    logger.info(f"Loaded {len(models)} models for {ontology}")
    
    return models


def generate_embeddings_from_fasta(fasta_file: str, output_dir: str, 
                                  logger: logging.Logger) -> str:
    """Generate embeddings from FASTA file using generate_embeddings.py"""
    logger.info(f"Generating embeddings from {fasta_file}")
    
    if not os.path.exists('generate_embeddings.py'):
        raise FileNotFoundError("generate_embeddings.py not found in current directory")
    
    embeddings_file = os.path.join(output_dir, 'generated_embeddings.npy')
    
    import subprocess
    cmd = [
        'python', 'generate_embeddings.py', 
        '--fasta', fasta_file,
        '--output', embeddings_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Embedding generation completed successfully")
        logger.debug(f"Embedding generation output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Embedding generation failed: {e.stderr}")
        raise
    
    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(f"Expected embeddings file not created: {embeddings_file}")
    
    return embeddings_file


def compute_ensemble_predictions(models: List[torch.nn.Module], data_loader: DataLoader, 
                               device: torch.device, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ensemble predictions and MAD uncertainty."""
    logger.info("Computing ensemble predictions...")
    
    all_probs = []
    
    for i, model in enumerate(tqdm(models, desc="Ensemble inference")):
        model.eval()
        probs = predict_without_MCD(model, data_loader, device)
        all_probs.append(probs)
    
    ensemble_probs = np.stack(all_probs, axis=0)
    
    median_probs = np.median(ensemble_probs, axis=0)
    mad_uncertainty = chunked_mad_over_runs(ensemble_probs, chunk_size=50)
    
    logger.info(f"Ensemble predictions computed: {median_probs.shape}")
    
    return median_probs, mad_uncertainty


def evaluate_predictions_cafa(median_probs: np.ndarray, mad_uncertainty: np.ndarray, 
                             tsv_file: str, mlb: MultiLabelBinarizer, 
                             thresholds: np.ndarray, ic_dict: Optional[Dict[str, float]],
                             logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict, Dict]:
    """Evaluate predictions against ground truth annotations using CAFA-style metrics with three approaches."""
    logger.info(f"Evaluating predictions against {tsv_file} using CAFA-style metrics")
    
    n_samples = median_probs.shape[0]
    dummy_embeddings = np.zeros((n_samples, 512))  # Match prediction sample count
    try:
        test_GO_df, _, test_GO_list, _ = process_GO_data(tsv_file, dummy_embeddings)
    except Exception as e:
        logger.error(f"Failed to process TSV file {tsv_file}: {e}")
        raise
    
    if len(test_GO_list) != median_probs.shape[0]:
        logger.warning(f"Sample count mismatch: TSV has {len(test_GO_list)}, predictions have {median_probs.shape[0]}")
        min_len = min(len(test_GO_list), median_probs.shape[0])
        test_GO_list = test_GO_list[:min_len]
        median_probs = median_probs[:min_len]
        mad_uncertainty = mad_uncertainty[:min_len]
    
    real_annots_dict = {}
    for i, go_terms in enumerate(test_GO_list):
        protein_id = f"protein_{i}"
        real_annots_dict[protein_id] = set(go_terms)
    
    if ic_dict is None:
        logger.warning("No IC dictionary available. Creating dummy IC values (all = 1.0)")
        ic_dict = {go_term: 1.0 for go_term in mlb.classes_}
    
    logger.info("=== APPROACH 1: Regular Median Thresholding ===")
    
    pred_annots_dict_median = {}
    for i in range(len(median_probs)):
        protein_id = f"protein_{i}"
        go_scores = {}
        for j, go_term in enumerate(mlb.classes_):
            score = median_probs[i, j]
            go_scores[go_term] = score
        pred_annots_dict_median[protein_id] = go_scores
    
    logger.info("Computing threshold-based metrics (median only)...")
    smin_median, fmax_median, best_threshold_s_median, best_threshold_f_median, s_at_fmax_median, metrics_df_median = \
        threshold_performance_metrics(ic_dict, real_annots_dict, pred_annots_dict_median, threshold_range=thresholds)
    
    logger.info("Adding coverage statistics (median only)...")
    coverage_stats_median = []
    
    for threshold in thresholds:
        binary_predictions = median_probs >= threshold
        
        terms_predicted = np.sum(binary_predictions.sum(axis=0) > 0)
        avg_predictions_per_protein = binary_predictions.sum(axis=1).mean()
        total_predictions = int(binary_predictions.sum())
        
        coverage_stats_median.append({
            'threshold': threshold,
            'terms_predicted': int(terms_predicted),
            'avg_predictions_per_protein': avg_predictions_per_protein,
            'total_predictions': total_predictions,
            'total_terms': len(mlb.classes_)
        })
    
    coverage_df_median = pd.DataFrame(coverage_stats_median)
    metrics_df_median = metrics_df_median.merge(coverage_df_median, left_on='n', right_on='threshold', how='left')
    
    logger.info("Computing AUPR micro (median only)...")
    aupr_micro_median = calculate_aupr_micro(real_annots_dict, pred_annots_dict_median)
    
    fmax_micro_median = metrics_df_median['f_micro'].max()
    best_threshold_f_micro_idx_median = metrics_df_median['f_micro'].idxmax()
    best_threshold_f_micro_median = metrics_df_median.loc[best_threshold_f_micro_idx_median, 'n']
    
    logger.info("=== APPROACH 2: Median-MAD (Effective Scores) Thresholding ===")
    
    effective_scores = median_probs - mad_uncertainty
    
    pred_annots_dict_effective = {}
    for i in range(len(effective_scores)):
        protein_id = f"protein_{i}"
        go_scores = {}
        for j, go_term in enumerate(mlb.classes_):
            score = effective_scores[i, j]
            go_scores[go_term] = score
        pred_annots_dict_effective[protein_id] = go_scores
    
    logger.info("Computing threshold-based metrics (median-MAD)...")
    smin_effective, fmax_effective, best_threshold_s_effective, best_threshold_f_effective, s_at_fmax_effective, metrics_df_effective = \
        threshold_performance_metrics(ic_dict, real_annots_dict, pred_annots_dict_effective, threshold_range=thresholds)
    
    logger.info("Adding coverage statistics (median-MAD)...")
    coverage_stats_effective = []
    
    for threshold in thresholds:
        binary_predictions = effective_scores >= threshold
        
        terms_predicted = np.sum(binary_predictions.sum(axis=0) > 0)
        avg_predictions_per_protein = binary_predictions.sum(axis=1).mean()
        total_predictions = int(binary_predictions.sum())
        
        coverage_stats_effective.append({
            'threshold': threshold,
            'terms_predicted': int(terms_predicted),
            'avg_predictions_per_protein': avg_predictions_per_protein,
            'total_predictions': total_predictions,
            'total_terms': len(mlb.classes_)
        })
    
    coverage_df_effective = pd.DataFrame(coverage_stats_effective)
    metrics_df_effective = metrics_df_effective.merge(coverage_df_effective, left_on='n', right_on='threshold', how='left')
    
    logger.info("Computing AUPR micro (median-MAD)...")
    aupr_micro_effective = calculate_aupr_micro(real_annots_dict, pred_annots_dict_effective)
    
    fmax_micro_effective = metrics_df_effective['f_micro'].max()
    best_threshold_f_micro_idx_effective = metrics_df_effective['f_micro'].idxmax()
    best_threshold_f_micro_effective = metrics_df_effective.loc[best_threshold_f_micro_idx_effective, 'n']
    
    predicted_terms = store_predicted_terms(median_probs, mad_uncertainty, thresholds, mlb)
    
    summary_metrics_median = {
        'approach': 'median_only',
        'fmax_macro': fmax_median,
        'fmax_micro': fmax_micro_median,
        'smin': smin_median,
        'aupr_micro': aupr_micro_median,
        'best_threshold_fmax_macro': best_threshold_f_median,
        'best_threshold_fmax_micro': best_threshold_f_micro_median,
        'best_threshold_smin': best_threshold_s_median,
        's_at_fmax': s_at_fmax_median,
        'total_proteins': len(real_annots_dict),
        'total_go_terms': len(mlb.classes_)
    }
    
    summary_metrics_effective = {
        'approach': 'median_minus_mad',
        'fmax_macro': fmax_effective,
        'fmax_micro': fmax_micro_effective,
        'smin': smin_effective,
        'aupr_micro': aupr_micro_effective,
        'best_threshold_fmax_macro': best_threshold_f_effective,
        'best_threshold_fmax_micro': best_threshold_f_micro_effective,
        'best_threshold_smin': best_threshold_s_effective,
        's_at_fmax': s_at_fmax_effective,
        'total_proteins': len(real_annots_dict),
        'total_go_terms': len(mlb.classes_)
    }
    
    logger.info(f"MEDIAN-ONLY Evaluation completed:")
    logger.info(f"  F-max macro: {fmax_median:.4f} @ threshold {best_threshold_f_median}")
    logger.info(f"  F-max micro: {fmax_micro_median:.4f} @ threshold {best_threshold_f_micro_median}")
    logger.info(f"  S-min: {smin_median:.4f} @ threshold {best_threshold_s_median}")
    logger.info(f"  AUPR micro: {aupr_micro_median:.4f}")
    
    logger.info(f"MEDIAN-MAD Evaluation completed:")
    logger.info(f"  F-max macro: {fmax_effective:.4f} @ threshold {best_threshold_f_effective}")
    logger.info(f"  F-max micro: {fmax_micro_effective:.4f} @ threshold {best_threshold_f_micro_effective}")
    logger.info(f"  S-min: {smin_effective:.4f} @ threshold {best_threshold_s_effective}")
    logger.info(f"  AUPR micro: {aupr_micro_effective:.4f}")
    
    return metrics_df_median, metrics_df_effective, summary_metrics_median, summary_metrics_effective, predicted_terms


def evaluate_fdr_thresholds(median_probs: np.ndarray, mad_uncertainty: np.ndarray, 
                           tsv_file: str, mlb: MultiLabelBinarizer,
                           effective_thresholds: Dict[str, Dict[str, float]], 
                           fdr_levels: List[float], ic_dict: Optional[Dict[str, float]], 
                           logger: logging.Logger) -> pd.DataFrame:
    """Evaluate predictions using FDR thresholds applied to effective scores (median - MAD) at multiple FDR levels."""
    logger.info(f"Evaluating FDR thresholds at levels: {fdr_levels}")
    
    n_samples = median_probs.shape[0]
    dummy_embeddings = np.zeros((n_samples, 512))
    test_GO_df, _, test_GO_list, _ = process_GO_data(tsv_file, dummy_embeddings)
    
    if len(test_GO_list) != median_probs.shape[0]:
        min_len = min(len(test_GO_list), median_probs.shape[0])
        test_GO_list = test_GO_list[:min_len]
        median_probs = median_probs[:min_len]
        mad_uncertainty = mad_uncertainty[:min_len]
    
    effective_scores = median_probs - mad_uncertainty
    
    true_labels = mlb.transform(test_GO_list)
    
    fdr_results = []
    
    for fdr_level in fdr_levels:
        logger.info(f"Evaluating FDR level: {fdr_level}")
        
        fdr_thresholds = extract_fdr_thresholds(effective_thresholds, fdr_level)
        
        if not fdr_thresholds:
            logger.warning(f"No thresholds found for FDR level {fdr_level}")
            continue
        
        logger.info(f"FDR {fdr_level}: Found thresholds for {len(fdr_thresholds)} terms")
        if fdr_thresholds:
            threshold_values = list(fdr_thresholds.values())
            logger.info(f"FDR {fdr_level}: Threshold range [{min(threshold_values):.4f}, {max(threshold_values):.4f}]")
        
        predictions = apply_fdr_thresholds(effective_scores, mlb, fdr_thresholds)
        
        metrics = evaluate_fdr_predictions(predictions, true_labels, mlb, fdr_level, fdr_thresholds, test_GO_list, ic_dict)
        fdr_results.append(metrics)
        
        if metrics['s_min'] is not None:
            logger.info(f"FDR {fdr_level}: F-macro={metrics['f_macro']:.4f}, "
                       f"F-micro={metrics['f_micro']:.4f}, "
                       f"S-min={metrics['s_min']:.4f}, "
                       f"Terms with thresholds={metrics['terms_with_thresholds']}")
        else:
            logger.info(f"FDR {fdr_level}: F-macro={metrics['f_macro']:.4f}, "
                       f"F-micro={metrics['f_micro']:.4f}, "
                       f"Terms with thresholds={metrics['terms_with_thresholds']}")
    
    fdr_df = pd.DataFrame(fdr_results)
    
    if len(fdr_df) > 1:
        logger.info("Checking FDR behavior (closest lower FDR approach):")
        hierarchy_violations = 0
        for i in range(len(fdr_df) - 1):
            current_fdr = fdr_df.iloc[i]['fdr_level']
            next_fdr = fdr_df.iloc[i + 1]['fdr_level']
            current_predictions = fdr_df.iloc[i]['total_predictions']
            next_predictions = fdr_df.iloc[i + 1]['total_predictions']
            
            if next_predictions >= current_predictions:
                logger.info(f"FDR {current_fdr} -> {next_fdr}: {current_predictions} -> {next_predictions} predictions")
            else:
                hierarchy_violations += 1
                logger.info(f"FDR {current_fdr} -> {next_fdr}: {current_predictions} -> {next_predictions} predictions (decreased)")
        
        if hierarchy_violations > 0:
            logger.info(f"  Note: {hierarchy_violations} hierarchy violations found.")
    
    return fdr_df


def save_results(ontology: str, median_probs: np.ndarray, mad_uncertainty: np.ndarray,
                predicted_terms: Dict, metrics_df_median: Optional[pd.DataFrame], 
                metrics_df_effective: Optional[pd.DataFrame],
                summary_metrics_median: Optional[Dict], summary_metrics_effective: Optional[Dict],
                fdr_metrics_df: Optional[pd.DataFrame], output_dir: str, logger: logging.Logger):
    """Save all results to output directory."""
    logger.info(f"Saving results for {ontology}")
    
    ontology_dir = os.path.join(output_dir, ontology.lower())
    os.makedirs(ontology_dir, exist_ok=True)
    
    np.save(os.path.join(ontology_dir, 'median_probabilities.npy'), median_probs)
    
    np.save(os.path.join(ontology_dir, 'mad_uncertainty.npy'), mad_uncertainty)
    
    predicted_terms_file = os.path.join(ontology_dir, 'predicted_terms.pkl')
    if not os.path.exists(predicted_terms_file):
        with open(predicted_terms_file, 'wb') as f:
            pickle.dump(predicted_terms, f)
    
    if metrics_df_median is not None:
        metrics_df_median.to_csv(os.path.join(ontology_dir, 'detailed_metrics_median_only.csv'), index=False)
        logger.info(f"Saved median-only metrics to {ontology_dir}/detailed_metrics_median_only.csv")
    
    if metrics_df_effective is not None:
        metrics_df_effective.to_csv(os.path.join(ontology_dir, 'detailed_metrics_median_mad.csv'), index=False)
        logger.info(f"Saved median-MAD metrics to {ontology_dir}/detailed_metrics_median_mad.csv")
    
    if summary_metrics_median is not None or summary_metrics_effective is not None:
        combined_summary = {}
        if summary_metrics_median is not None:
            combined_summary['median_only'] = summary_metrics_median
        if summary_metrics_effective is not None:
            combined_summary['median_minus_mad'] = summary_metrics_effective
        
        with open(os.path.join(ontology_dir, 'summary_metrics.json'), 'w') as f:
            json.dump(combined_summary, f, indent=2)
        logger.info(f"Saved combined summary metrics to {ontology_dir}/summary_metrics.json")
    
    if fdr_metrics_df is not None:
        fdr_metrics_df.to_csv(os.path.join(ontology_dir, 'fdr_threshold_metrics.csv'), index=False)
        logger.info(f"Saved FDR threshold metrics to {ontology_dir}/fdr_threshold_metrics.csv")
    
    logger.info(f"Results saved to {ontology_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Protein Function Prediction with Uncertainty Quantification and FDR Thresholds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--fasta', help='Input FASTA file with protein sequences')
    input_group.add_argument('--embeddings', help='Pre-computed embeddings (.npy file)')
    
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    
    parser.add_argument('--bp_models_dir', default='bp_publish', help='Directory containing BPO models')
    parser.add_argument('--cc_models_dir', default='cc_publish', help='Directory containing CCO models') 
    parser.add_argument('--mf_models_dir', default='mf_publish', help='Directory containing MFO models')
    
    parser.add_argument('--bp_tsv', help='BPO annotations TSV file for evaluation')
    parser.add_argument('--cc_tsv', help='CCO annotations TSV file for evaluation')
    parser.add_argument('--mf_tsv', help='MFO annotations TSV file for evaluation')
    
    parser.add_argument('--ic_file', default='IA_all.tsv', 
                       help='Information Content TSV file for all ontologies (default: IA_all.tsv)')
    
    parser.add_argument('--fdr_levels', nargs='+', type=float, 
                       default=[0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                       help='FDR levels to evaluate (default: 0.01 0.05 0.10 0.15 0.20 0.25 0.30)')
    
    parser.add_argument('--fdr_range_start', type=float, default=None,
                       help='Starting FDR value for continuous range (e.g., 0.01)')
    parser.add_argument('--fdr_range_end', type=float, default=None,
                       help='Ending FDR value for continuous range (e.g., 0.5)')
    parser.add_argument('--fdr_range_step', type=float, default=0.01,
                       help='Step size for FDR range (default: 0.01)')
    
    parser.add_argument('--config_file', default='TUNED_MODEL_ARCHS.json',
                       help='JSON file containing model architectures')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for inference')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Device to use for inference')
    
    parser.add_argument('--skip_metrics', action='store_true',
                       help='Skip evaluation metrics computation')
    parser.add_argument('--skip_prob_thresholds', action='store_true',
                       help='Skip original probability threshold evaluation')
    parser.add_argument('--ontologies', nargs='+', choices=['BPO', 'CCO', 'MFO'], 
                       default=['BPO', 'CCO', 'MFO'], help='Ontologies to predict')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir, args.verbose)
    
    if args.fdr_range_start is not None and args.fdr_range_end is not None:
        if args.fdr_range_start >= args.fdr_range_end:
            raise ValueError("fdr_range_start must be less than fdr_range_end")
        if args.fdr_range_step <= 0:
            raise ValueError("fdr_range_step must be positive")
        if args.fdr_range_start < 0 or args.fdr_range_end > 1:
            raise ValueError("FDR range values must be between 0 and 1")
        
        fdr_levels = np.arange(args.fdr_range_start, args.fdr_range_end + args.fdr_range_step/2, args.fdr_range_step)
        fdr_levels = np.round(fdr_levels, 3)  # Round to avoid floating point precision issues
        logger.info(f"Using continuous FDR range: {args.fdr_range_start} to {args.fdr_range_end} (step: {args.fdr_range_step})")
        logger.info(f"Generated {len(fdr_levels)} FDR levels")
    elif args.fdr_range_start is not None or args.fdr_range_end is not None:
        raise ValueError("Both fdr_range_start and fdr_range_end must be specified when using FDR range")
    else:
        fdr_levels = args.fdr_levels
        logger.info(f"Using discrete FDR levels: {fdr_levels}")
    
    if any(fdr < 0 or fdr > 1 for fdr in fdr_levels):
        raise ValueError("All FDR levels must be between 0 and 1")
    
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    
    try:
        if args.fasta:
            logger.info(f"Processing FASTA file: {args.fasta}")
            embeddings_file = generate_embeddings_from_fasta(args.fasta, args.output_dir, logger)
        else:
            logger.info(f"Using pre-computed embeddings: {args.embeddings}")
            embeddings_file = args.embeddings
        
        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        
        embeddings = np.load(embeddings_file)
        logger.info(f"Loaded embeddings shape: {embeddings.shape}")
        
        dataset = TensorDataset(torch.tensor(embeddings, dtype=torch.float32))
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        
        ic_dict = load_ic_dict(args.ic_file, logger)
        
        ontology_config = {
            'BPO': {'models_dir': args.bp_models_dir, 'tsv': args.bp_tsv, 'name': 'process'},
            'CCO': {'models_dir': args.cc_models_dir, 'tsv': args.cc_tsv, 'name': 'component'},
            'MFO': {'models_dir': args.mf_models_dir, 'tsv': args.mf_tsv, 'name': 'function'}
        }
        
        overall_results = {}
        all_fdr_results = {}
        
        for ontology in args.ontologies:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {ontology}")
            logger.info(f"{'='*50}")
            
            config = ontology_config[ontology]
            
            models = load_ensemble_models(config['models_dir'], ontology, device, logger)
            
            mlb = load_mlb(config['name'])
            logger.info(f"Loaded MLBs for {len(mlb.classes_)} classes")
            
            effective_thresholds = load_effective_thresholds(config['name'], logger)
            
            median_probs, mad_uncertainty = compute_ensemble_predictions(
                models, data_loader, device, logger
            )
            
            from generate_fdr_predictions import save_both_prediction_formats
            logger.info("Generating predictions in both original and FDR formats...")
            predicted_terms, predicted_terms_fdr = save_both_prediction_formats(
                ontology, median_probs, mad_uncertainty, mlb, thresholds, 
                effective_thresholds, fdr_levels, args.output_dir
            )
            
            if predicted_terms:
                logger.info(f"Generated original format: predicted_terms.pkl with {len(predicted_terms)} confidence levels")
            if predicted_terms_fdr:
                logger.info(f"Generated FDR format: predicted_terms_fdr.pkl with {len(predicted_terms_fdr)} FDR levels")
            elif effective_thresholds is None:
                logger.warning("No effective thresholds available - FDR format not generated")
            
            metrics_df_median = None
            metrics_df_effective = None
            summary_metrics_median = None
            summary_metrics_effective = None
            
            if config['tsv'] and not args.skip_metrics and not args.skip_prob_thresholds:
                try:
                    logger.info("=== Probability Threshold Evaluation (Median vs Median-MAD) ===")
                    metrics_df_median, metrics_df_effective, summary_metrics_median, summary_metrics_effective, _ = evaluate_predictions_cafa(
                        median_probs, mad_uncertainty, config['tsv'], mlb, thresholds, ic_dict, logger
                    )
                    overall_results[f"{ontology}_median_only"] = summary_metrics_median
                    overall_results[f"{ontology}_median_mad"] = summary_metrics_effective
                except Exception as e:
                    logger.error(f"Original evaluation failed for {ontology}: {e}")
            
            fdr_metrics_df = None
            if config['tsv'] and not args.skip_metrics and effective_thresholds:
                try:
                    logger.info("=== FDR Threshold Evaluation ===")
                    fdr_metrics_df = evaluate_fdr_thresholds(
                        median_probs, mad_uncertainty, config['tsv'], mlb, effective_thresholds, fdr_levels, ic_dict, logger
                    )
                    all_fdr_results[ontology] = fdr_metrics_df
                except Exception as e:
                    logger.error(f"FDR evaluation failed for {ontology}: {e}")
            
            save_results(ontology, median_probs, mad_uncertainty, predicted_terms,
                        metrics_df_median, metrics_df_effective, summary_metrics_median, 
                        summary_metrics_effective, fdr_metrics_df, args.output_dir, logger)
        
        if overall_results:
            with open(os.path.join(args.output_dir, 'overall_summary.json'), 'w') as f:
                json.dump(overall_results, f, indent=2)
        
        if all_fdr_results:
            combined_fdr_df = pd.DataFrame()
            for ontology, fdr_df in all_fdr_results.items():
                fdr_df_copy = fdr_df.copy()
                fdr_df_copy['ontology'] = ontology
                combined_fdr_df = pd.concat([combined_fdr_df, fdr_df_copy], ignore_index=True)
            
            cols = ['ontology'] + [col for col in combined_fdr_df.columns if col != 'ontology']
            combined_fdr_df = combined_fdr_df[cols]
            
            combined_fdr_df.to_csv(os.path.join(args.output_dir, 'fdr_threshold_metrics_all_ontologies.csv'), index=False)
            logger.info(f"Saved combined FDR metrics to {args.output_dir}/fdr_threshold_metrics_all_ontologies.csv")
        
        logger.info(f"\n{'='*60}")
        logger.info("PREDICTION COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*60}")
        logger.info(f"Input: {args.fasta if args.fasta else args.embeddings}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Processed ontologies: {args.ontologies}")
        logger.info(f"Number of proteins: {embeddings.shape[0]}")
        if args.fdr_range_start is not None:
            logger.info(f"FDR range: {args.fdr_range_start} to {args.fdr_range_end} (step: {args.fdr_range_step}, {len(fdr_levels)} levels)")
        else:
            logger.info(f"FDR levels evaluated: {fdr_levels}")
        
        if overall_results:
            logger.info("\nOriginal Performance Summary (CAFA-style metrics):")
            
            ontology_groups = {}
            for key, metrics in overall_results.items():
                if '_median_only' in key:
                    ontology = key.replace('_median_only', '')
                    if ontology not in ontology_groups:
                        ontology_groups[ontology] = {}
                    ontology_groups[ontology]['median_only'] = metrics
                elif '_median_mad' in key:
                    ontology = key.replace('_median_mad', '')
                    if ontology not in ontology_groups:
                        ontology_groups[ontology] = {}
                    ontology_groups[ontology]['median_mad'] = metrics
            
            for ontology, approaches in ontology_groups.items():
                logger.info(f"  {ontology}:")
                
                if 'median_only' in approaches:
                    metrics = approaches['median_only']
                    logger.info(f"    MEDIAN-ONLY:")
                    logger.info(f"      F-max macro: {metrics['fmax_macro']:.4f} @ {metrics['best_threshold_fmax_macro']:.2f}")
                    logger.info(f"      F-max micro: {metrics['fmax_micro']:.4f} @ {metrics['best_threshold_fmax_micro']:.2f}")
                    logger.info(f"      S-min: {metrics['smin']:.4f} @ {metrics['best_threshold_smin']:.2f}")
                    logger.info(f"      AUPR micro: {metrics['aupr_micro']:.4f}")
                
                if 'median_mad' in approaches:
                    metrics = approaches['median_mad']
                    logger.info(f"    MEDIAN-MAD:")
                    logger.info(f"      F-max macro: {metrics['fmax_macro']:.4f} @ {metrics['best_threshold_fmax_macro']:.2f}")
                    logger.info(f"      F-max micro: {metrics['fmax_micro']:.4f} @ {metrics['best_threshold_fmax_micro']:.2f}")
                    logger.info(f"      S-min: {metrics['smin']:.4f} @ {metrics['best_threshold_smin']:.2f}")
                    logger.info(f"      AUPR micro: {metrics['aupr_micro']:.4f}")
        
        if all_fdr_results:
            logger.info("\nFDR Threshold Performance Summary:")
            for ontology, fdr_df in all_fdr_results.items():
                logger.info(f"  {ontology}:")
                best_f_micro_idx = fdr_df['f_micro'].idxmax()
                best_f_macro_idx = fdr_df['f_macro'].idxmax()
                logger.info(f"    Best F-micro: {fdr_df.loc[best_f_micro_idx, 'f_micro']:.4f} @ FDR {fdr_df.loc[best_f_micro_idx, 'fdr_level']}")
                logger.info(f"    Best F-macro: {fdr_df.loc[best_f_macro_idx, 'f_macro']:.4f} @ FDR {fdr_df.loc[best_f_macro_idx, 'fdr_level']}")
                
                if 's_min' in fdr_df.columns and fdr_df['s_min'].notna().any():
                    best_s_min_idx = fdr_df['s_min'].idxmin()
                    logger.info(f"    Best S-min: {fdr_df.loc[best_s_min_idx, 's_min']:.4f} @ FDR {fdr_df.loc[best_s_min_idx, 'fdr_level']}")
        
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 