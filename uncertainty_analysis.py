#!/usr/bin/env python3
"""
Uncertainty Analysis for Protein Function Prediction Models

This script performs comprehensive uncertainty quantification analysis across different
ontologies (BPO, CCO, MFO) using multiple uncertainty estimation methods.

EXAMPLE USAGE CASES:

# Basic usage - BP uncertainty analysis with default settings
python uncertainty_analysis.py --ontology BPO --models_dir bp_publish

# CC analysis with custom output directory
python uncertainty_analysis.py --ontology CCO --models_dir cc_publish --output_dir cc_uncertainty_results

# MF analysis with extended save and custom uncertainty metrics
python uncertainty_analysis.py --ontology MFO --models_dir mf_publish \
    --save_extended --uncertainty_metric std --central_tendency mean

# Custom MC Dropout passes and threshold range
python uncertainty_analysis.py --ontology BPO --models_dir bp_publish \
    --mcd_passes 10 20 50 --thresholds 0.05 0.1 0.2 0.3 0.5

# Batch analysis with robustness testing
python uncertainty_analysis.py --ontology BPO --models_dir bp_publish \
    --noise_levels 20 --save_extended --generate_report

# Quick analysis without robustness testing
python uncertainty_analysis.py --ontology CCO --models_dir cc_publish \
    --skip_robustness --uncertainty_metric mad

# Custom data directory and configuration
python uncertainty_analysis.py --ontology MFO --models_dir mf_publish \
    --data_dir custom_processed_data --config_file custom_archs.json

# Analysis with specific k-fold ensembles only
python uncertainty_analysis.py --ontology BPO --models_dir bp_publish \
    --ensemble_folds 10 20 --skip_mcd --skip_temporal

# Save plot data for custom replotting
python uncertainty_analysis.py --ontology BPO --models_dir bp_publish \
    --save_plot_data --uncertainty_metric mad

# Use smart defaults (median for MC Dropout/Ensembles, mean for temporal)
python uncertainty_analysis.py --ontology BPO --models_dir bp_publish \
    --central_tendency auto
"""

import argparse
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.stats import shapiro

from utils_corrected import (
    PFP, process_GO_data, mc_dropout_inference, predict_without_MCD,
    store_predicted_terms, calculate_sharpness_coverage_and_fdr,
    compute_effective_scores, chunked_mad_over_runs, chunked_median_over_runs,
    load_kfold_ensembles, compute_ensemble_probs
)


def setup_logging(output_dir: str, verbose: bool = False) -> logging.Logger:
    log_level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger('uncertainty_analysis')
    logger.setLevel(log_level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    log_file = os.path.join(output_dir, 'run_log.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_ontology_config(ontology: str) -> Dict[str, str]:
    config = {
        'BPO': {'name': 'process', 'type': 'BPO'},
        'CCO': {'name': 'component', 'type': 'CCO'}, 
        'MFO': {'name': 'function', 'type': 'MFO'}
    }
    if ontology not in config:
        raise ValueError(f"Unknown ontology: {ontology}. Choose from {list(config.keys())}")
    return config[ontology]


def load_test_data(ontology_name: str, data_dir: str, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List, MultiLabelBinarizer]:
    logger.info(f"Loading test data for {ontology_name}")
    
    test_embeddings_path = os.path.join(data_dir, f'{ontology_name}_test.npy')
    test_tsv_path = os.path.join(data_dir, f'{ontology_name}_test.tsv')
    mlb_path = f'{ontology_name}_mlb.pkl'
    
    for path in [test_embeddings_path, test_tsv_path, mlb_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
    
    test_embeddings = np.load(test_embeddings_path)
    logger.info(f"Loaded embeddings shape: {test_embeddings.shape}")
    
    test_GO_df, test_embeddings, test_GO_list, test_GO_annotated = process_GO_data(
        test_tsv_path, test_embeddings
    )
    logger.info(f"Processed GO data - samples: {len(test_GO_df)}")
    
    with open(mlb_path, 'rb') as f:
        mlb = pickle.load(f)
    
    test_labels = mlb.transform(test_GO_list)
    logger.info(f"Test labels shape: {test_labels.shape}")
    
    return test_embeddings, test_labels, test_GO_df, test_GO_list, mlb


def load_models(models_dir: str, ontology: str, config_file: str, device: torch.device, logger: logging.Logger) -> Dict[str, Any]:
    logger.info(f"Loading models from {models_dir}")
    
    with open(config_file, 'r') as f:
        tuned_model_archs = json.load(f)
    
    if ontology not in tuned_model_archs:
        raise KeyError(f"Ontology {ontology} not found in config file")
    
    arch_config = tuned_model_archs[ontology]
    
    base_kwargs = {
        'input_dim': 512,
        'hidden_dims': arch_config['CHOSEN_CONFIG_ARCH'],
        'output_dim': arch_config['NUM_CLASSES'],
        'dropout_rate': 0.3  # Default gamma
    }
    
    models = {
        'config': arch_config,
        'base_kwargs': base_kwargs
    }
    
    best_model_path = os.path.join(models_dir, f"{ontology}_best_model.pt")
    if os.path.exists(best_model_path):
        best_model = PFP(**base_kwargs).to(device)
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
        best_model.eval()
        models['best_model'] = best_model
        logger.info("Loaded best deterministic model")
    
    epoch_states_path = os.path.join(models_dir, f"{ontology}_epoch_states.pt")
    if os.path.exists(epoch_states_path):
        all_states = torch.load(epoch_states_path, map_location=device)
        models['epoch_states'] = all_states
        logger.info(f"Loaded {len(all_states)} epoch states")
    
    kf_root = os.path.join(models_dir, "k_folds")
    if os.path.exists(kf_root):
        fold_options = [5, 10, 20]
        ensemble = load_kfold_ensembles(
            kf_root=kf_root,
            fold_options=fold_options,
            model_cls=PFP,
            model_kwargs=base_kwargs,
            device=device
        )
        models['ensembles'] = {'k_folds': ensemble}
        logger.info(f"Loaded k-fold ensembles: {list(ensemble.keys())}")
    
    return models


def compute_uncertainty_metrics(predictions_array: np.ndarray, method: str = 'mad', 
                              central: str = 'median', method_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute central tendency and uncertainty metrics.
    
    Args:
        predictions_array: shape (n_runs, n_samples, n_classes)
        method: 'mad', 'std', 'var'
        central: 'median', 'mean', 'auto'
        method_name: name of the uncertainty method for auto selection
    """
    if central == 'auto':
        if method_name and 'temporal' in method_name.lower():
            actual_central = 'mean'
        else:
            actual_central = 'median'
    else:
        actual_central = central
    
    if actual_central == 'median':
        central_values = chunked_median_over_runs(predictions_array, chunk_size=50)
    else:
        central_values = np.mean(predictions_array, axis=0)
    
    if method == 'mad':
        uncertainty = chunked_mad_over_runs(predictions_array, chunk_size=50)
    elif method == 'std':
        uncertainty = np.std(predictions_array, axis=0)
    elif method == 'var':
        uncertainty = np.var(predictions_array, axis=0)
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")
    
    return central_values, uncertainty


def compute_uncertainties(models: Dict, test_loader: DataLoader, mcd_passes: List[int], 
                         uncertainty_metric: str, central_tendency: str, 
                         device: torch.device, logger: logging.Logger,
                         skip_mcd: bool = False, skip_temporal: bool = False, 
                         skip_ensemble: bool = False) -> Dict[str, Any]:
    """Run all uncertainty methods: MCD, temporal, ensemble."""
    logger.info("Computing uncertainties...")
    
    results = {}
    
    if not skip_mcd and 'best_model' in models:
        logger.info("Running Monte Carlo Dropout inference...")
        mcd_results = {}
        
        for passes in mcd_passes:
            logger.info(f"  MC Dropout with {passes} passes")
            mcd_probs = mc_dropout_inference(models['best_model'], test_loader, passes, device)
            method_name = f'mcd_{passes}'
            central, uncertainty = compute_uncertainty_metrics(
                mcd_probs, uncertainty_metric, central_tendency, method_name
            )
            actual_central = 'median' if central_tendency in ['auto', 'median'] else central_tendency
            logger.info(f"    Using {actual_central} for central tendency")
            mcd_results[method_name] = {
                'probs': mcd_probs,
                'central': central,
                'uncertainty': uncertainty
            }
        
        results['mcd'] = mcd_results
    
    if not skip_temporal and 'epoch_states' in models:
        logger.info("Running temporal ensemble...")
        last_epochs = sorted(models['epoch_states'].keys())[-5:]
        
        temporal_probs = []
        for epoch in last_epochs:
            temp_model = PFP(**models['base_kwargs']).to(device)
            temp_model.load_state_dict(models['epoch_states'][epoch])
            temp_model.eval()
            probs = predict_without_MCD(temp_model, test_loader, device)
            temporal_probs.append(probs)
        
        temporal_array = np.stack(temporal_probs, axis=0)
        method_name = 'temporal'
        central, uncertainty = compute_uncertainty_metrics(
            temporal_array, uncertainty_metric, central_tendency, method_name
        )
        actual_central = 'mean' if central_tendency == 'auto' else central_tendency
        logger.info(f"  Using {actual_central} for temporal ensemble central tendency")
        
        results['temporal'] = {
            'temporal': {
                'probs': temporal_array,
                'central': central,
                'uncertainty': uncertainty
            }
        }
    
    if not skip_ensemble and 'ensembles' in models:
        logger.info("Running deep ensemble inference...")
        ensemble_results = {}
        
        ensemble_probs = compute_ensemble_probs(
            models['ensembles'], test_loader, device, predict_without_MCD
        )
        
        for k, probs_array in ensemble_probs.items():
            logger.info(f"  Processing {k}-fold ensemble")
            method_name = f'ensemble_{k}'
            central, uncertainty = compute_uncertainty_metrics(
                probs_array, uncertainty_metric, central_tendency, method_name
            )
            actual_central = 'median' if central_tendency in ['auto', 'median'] else central_tendency
            logger.info(f"    Using {actual_central} for central tendency")
            ensemble_results[method_name] = {
                'probs': probs_array,
                'central': central,
                'uncertainty': uncertainty
            }
        
        results['ensemble'] = ensemble_results
    
    if 'best_model' in models:
        logger.info("Computing deterministic baseline...")
        det_probs = predict_without_MCD(models['best_model'], test_loader, device)
        results['deterministic'] = {
            'central': det_probs,
            'uncertainty': np.zeros_like(det_probs)
        }
    
    return results


def analyze_performance(uncertainty_results: Dict, test_labels: np.ndarray, 
                       test_GO_list: List, mlb: MultiLabelBinarizer,
                       thresholds: np.ndarray, logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    """Compute metrics, effective scores, and generate analysis."""
    logger.info("Analyzing performance across methods...")
    
    performance_dfs = {}
    
    for method_type, methods in uncertainty_results.items():
        if method_type == 'deterministic':
            central = methods['central']
            uncertainty = methods['uncertainty']
            
            predicted_terms = store_predicted_terms(central, uncertainty, thresholds, mlb)
            metrics = calculate_sharpness_coverage_and_fdr(predicted_terms, test_GO_list, thresholds)
            
            performance_dfs['deterministic'] = pd.DataFrame({
                'sharpness': metrics[0],
                'coverage': metrics[1], 
                'fdr': metrics[2],
                'precision_micro': metrics[3],
                'precision_macro': metrics[4],
                'recall_micro': metrics[5],
                'recall_macro': metrics[6]
            }, index=thresholds)
            
        else:
            for method_name, method_data in methods.items():
                logger.info(f"  Analyzing {method_name}")
                
                central = method_data['central']
                uncertainty = method_data['uncertainty']
                
                predicted_terms = store_predicted_terms(central, uncertainty, thresholds, mlb)
                metrics = calculate_sharpness_coverage_and_fdr(predicted_terms, test_GO_list, thresholds)
                
                performance_dfs[method_name] = pd.DataFrame({
                    'sharpness': metrics[0],
                    'coverage': metrics[1],
                    'fdr': metrics[2], 
                    'precision_micro': metrics[3],
                    'precision_macro': metrics[4],
                    'recall_micro': metrics[5],
                    'recall_macro': metrics[6]
                }, index=thresholds)
    
    return performance_dfs


def test_normality(uncertainty_results: Dict, test_labels: np.ndarray, 
                  logger: logging.Logger, n_tests: int = 1000) -> Dict[str, Tuple[int, int]]:
    """Test normality of uncertainty distributions."""
    logger.info("Testing normality of uncertainty distributions...")
    
    normality_results = {}
    alpha = 0.05
    
    for method_type, methods in uncertainty_results.items():
        if method_type == 'deterministic':
            continue
            
        for method_name, method_data in methods.items():
            if 'probs' not in method_data:
                continue
                
            arr = method_data['probs']  # shape (runs, N, C)
            runs, N, C = arr.shape
            total_pairs = N * C
            
            k = min(n_tests, total_pairs)
            flat_idx = np.random.choice(total_pairs, size=k, replace=False)
            sample_idxs = flat_idx // C
            class_idxs = flat_idx % C
            
            normal_count = 0
            for si, cj in zip(sample_idxs, class_idxs):
                dist = arr[:, si, cj]
                if runs >= 3:
                    try:
                        W, p_value = shapiro(dist)
                        if p_value > alpha:
                            normal_count += 1
                    except:
                        continue
            
            non_normal_count = k - normal_count
            normality_results[method_name] = (normal_count, non_normal_count)
    
    return normality_results


def robustness_analysis(models: Dict, test_embeddings: np.ndarray, test_labels: np.ndarray,
                       noise_levels: int, device: torch.device, logger: logging.Logger) -> Tuple[np.ndarray, List, List]:
    """Gaussian noise injection testing."""
    logger.info("Running robustness analysis with noise injection...")
    
    # Use 20-fold ensemble for robustness testing 
    if 'ensembles' not in models or 20 not in models['ensembles']['k_folds']:
        logger.warning("20-fold ensemble not available for robustness testing")
        return np.array([]), [], []
    
    models_k20 = [m for (_, m) in models['ensembles']['k_folds'][20]]
    noise_levels_array = np.linspace(0.1, 3.0, noise_levels)
    all_tp_mads = []
    avg_mads = []
    
    def ensemble_mad_on_embeddings(embeddings):
        """Compute MAD for ensemble on given embeddings."""
        with torch.no_grad():
            probs = []
            X = torch.tensor(embeddings, dtype=torch.float32).to(device)
            for m in models_k20:
                logits = m(X)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)
        
        arr = np.stack(probs, axis=0)
        med = np.median(arr, axis=0)
        mad = np.median(np.abs(arr - med[None]), axis=0)
        return mad
    
    for sigma in tqdm(noise_levels_array, desc="Computing robustness"):
        noise = np.random.normal(0, sigma, size=test_embeddings.shape)
        noisy_emb = test_embeddings + noise
        
        mad_matrix = ensemble_mad_on_embeddings(noisy_emb)
        tp_mads = mad_matrix[test_labels.astype(bool)]
        
        all_tp_mads.append(tp_mads)
        avg_mads.append(tp_mads.mean())
    
    return noise_levels_array, all_tp_mads, avg_mads


def create_plots(performance_dfs: Dict, uncertainty_results: Dict, test_labels: np.ndarray,
                normality_results: Dict, robustness_data: Tuple, mlb: MultiLabelBinarizer,
                output_dir: str, save_plot_data: bool, logger: logging.Logger):
    logger.info("Creating plots...")
    
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_data = {}  # Dictionary to store all plot data
    
    performance_plot_data = {
        'methods': {},
        'plot_config': {
            'subplots': ['fdr', 'sharpness', 'precision_macro', 'recall_macro'],
            'subplot_labels': {
                'fdr': {'xlabel': 'Threshold', 'ylabel': 'FDR (micro)'},
                'sharpness': {'xlabel': 'Threshold', 'ylabel': 'Sharpness'},
                'precision_macro': {'xlabel': 'Threshold', 'ylabel': 'Precision (Macro)'},
                'recall_macro': {'xlabel': 'Threshold', 'ylabel': 'Recall (Macro)'}
            },
            'title': 'Performance Comparison Across Uncertainty Methods',
            'layout': {'rows': 2, 'cols': 2, 'figsize': [12, 10]}
        }
    }
    
    for method_name, df in performance_dfs.items():
        performance_plot_data['methods'][method_name] = {
            'thresholds': df.index.tolist(),
            'fdr': df['fdr'].tolist(),
            'sharpness': df['sharpness'].tolist(),
            'precision_macro': df['precision_macro'].tolist(),
            'recall_macro': df['recall_macro'].tolist()
        }
    
    plot_data['performance_comparison'] = performance_plot_data
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=200)
    
    for method_name, df in performance_dfs.items():
        axes[0, 0].plot(df.index, df['fdr'], label=method_name)
        axes[0, 1].plot(df.index, df['sharpness'], label=method_name)
        axes[1, 0].plot(df.index, df['precision_macro'], label=method_name)
        axes[1, 1].plot(df.index, df['recall_macro'], label=method_name)
    
    axes[0, 0].set(xlabel='Threshold', ylabel='FDR (micro)')
    axes[0, 1].set(xlabel='Threshold', ylabel='Sharpness')
    axes[1, 0].set(xlabel='Threshold', ylabel='Precision (Macro)')
    axes[1, 1].set(xlabel='Threshold', ylabel='Recall (Macro)')
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3, fontsize='small')
    fig.suptitle('Performance Comparison Across Uncertainty Methods', fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(os.path.join(plots_dir, 'performance_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    if normality_results:
        labels = list(normality_results.keys())
        normals = [normality_results[n][0] for n in labels]
        non_normals = [normality_results[n][1] for n in labels]
        
        normality_plot_data = {
            'methods': labels,
            'normal_counts': normals,
            'non_normal_counts': non_normals,
            'plot_config': {
                'bar_width': 0.35,
                'figsize': [10, 6],
                'title': 'Normality Test Results',
                'ylabel': '# of (protein, term) distributions',
                'xlabel': 'Methods',
                'rotation': 45,
                'legend_labels': ['Normal', 'Non-Normal']
            }
        }
        plot_data['normality_test'] = normality_plot_data
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, normals, width, label='Normal')
        ax.bar(x + width/2, non_normals, width, label='Non-Normal')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('# of (protein, term) distributions')
        ax.set_title('Normality Test Results')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'normality_test_results.png'), dpi=200, bbox_inches='tight')
        plt.close()
    
    if robustness_data[0].size > 0:
        noise_levels, all_tp_mads, avg_mads = robustness_data
        
        robustness_plot_data = {
            'noise_levels': noise_levels.tolist(),
            'avg_mads': avg_mads,
            'plot_config': {
                'figsize': [8, 6],
                'title': 'Ensemble Epistemic Uncertainty with Input Noise',
                'xlabel': 'Gaussian Noise σ added to embeddings',
                'ylabel': 'Average MAD (over TP calls)',
                'marker': 'o',
                'linewidth': 2
            }
        }
        plot_data['robustness'] = robustness_plot_data
        
        plt.figure(figsize=(8, 6), dpi=200)
        plt.plot(noise_levels, avg_mads, marker='o', linewidth=2)
        plt.xlabel('Gaussian Noise σ added to embeddings')
        plt.ylabel('Average MAD (over TP calls)')
        plt.title('Ensemble Epistemic Uncertainty with Input Noise')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'robustness_analysis.png'), dpi=200, bbox_inches='tight')
        plt.close()
    
    ensemble_data = None
    ensemble_name = None
    if 'ensemble' in uncertainty_results:
        if 'ensemble_20' in uncertainty_results['ensemble']:
            ensemble_data = uncertainty_results['ensemble']['ensemble_20']
            ensemble_name = 'ensemble_20'
        elif 'ensemble_10' in uncertainty_results['ensemble']:
            ensemble_data = uncertainty_results['ensemble']['ensemble_10']
            ensemble_name = 'ensemble_10'
        elif 'ensemble_5' in uncertainty_results['ensemble']:
            ensemble_data = uncertainty_results['ensemble']['ensemble_5']
            ensemble_name = 'ensemble_5'
    
    if ensemble_data is not None:
        logger.info(f"Creating additional uncertainty plots using {ensemble_name}")
        
        central_probs = ensemble_data['central']  # shape (N, C)
        uncertainty_vals = ensemble_data['uncertainty']  # shape (N, C)
        
        # 1. Median vs MAD scatter plot for true positives
        logger.info("Creating median vs MAD scatter plot...")
        med_flat = central_probs.ravel()
        mad_flat = uncertainty_vals.ravel()
        labels_flat = test_labels.ravel().astype(bool)
        
        tp_med = med_flat[labels_flat]
        tp_mad = mad_flat[labels_flat]
        
        # Subsample for plotting if too many points
        max_points = len(tp_med)
        if len(tp_med) > max_points:
            indices = np.random.choice(len(tp_med), max_points, replace=False)
            tp_med_plot = tp_med[indices]
            tp_mad_plot = tp_mad[indices]
        else:
            tp_med_plot = tp_med
            tp_mad_plot = tp_mad
        
        median_mad_scatter_data = {
            'median_values': tp_med_plot.tolist(),
            'mad_values': tp_mad_plot.tolist(),
            'total_points': len(tp_med),
            'plotted_points': len(tp_med_plot),
            'ensemble_method': ensemble_name,
            'plot_config': {
                'figsize': [8, 6],
                'title': f'True Positive Uncertainty Patterns ({ensemble_name})',
                'xlabel': f'Median Probability ({ensemble_name})',
                'ylabel': f'MAD ({ensemble_name})',
                'alpha': 0.1,
                'marker_size': 1
            }
        }
        plot_data['median_mad_scatter'] = median_mad_scatter_data
        
        plt.figure(figsize=(8, 6), dpi=200)
        plt.scatter(tp_med_plot, tp_mad_plot, alpha=0.1, s=1, edgecolors='none')
        plt.xlabel(f'Median Probability ({ensemble_name})')
        plt.ylabel(f'MAD ({ensemble_name})')
        plt.title(f'True Positive Uncertainty Patterns ({ensemble_name})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'median_mad_scatter.png'), dpi=200, bbox_inches='tight')
        plt.close()
        
        # 2. Probability range box plot
        logger.info("Creating probability range box plot...")
        import pandas as pd
        
        df_uncertainty = pd.DataFrame({
            'median': tp_med,
            'mad': tp_mad
        })
        
        prob_bins = np.arange(0.0, 1.1, 0.1)  # [0.0, 0.1, 0.2, ..., 1.0]
        df_uncertainty['prob_bin'] = pd.cut(df_uncertainty['median'], bins=prob_bins, labels=False, include_lowest=True)
        
        prob_range_data = {}
        prob_range_labels = []
        for i in range(len(prob_bins) - 1):
            range_mask = df_uncertainty['prob_bin'] == i
            if range_mask.sum() > 0:
                prob_range_data[i] = df_uncertainty.loc[range_mask, 'mad'].tolist()
                prob_range_labels.append(f"{prob_bins[i]:.1f}-{prob_bins[i+1]:.1f}")
            else:
                prob_range_data[i] = []
                prob_range_labels.append(f"{prob_bins[i]:.1f}-{prob_bins[i+1]:.1f}")
        
        prob_range_boxplot_data = {
            'prob_range_data': prob_range_data,
            'prob_range_labels': prob_range_labels,
            'ensemble_method': ensemble_name,
            'plot_config': {
                'figsize': [8, 6],
                'title': f'Epistemic Uncertainty by Probability Range ({ensemble_name})',
                'xlabel': 'Probability Range',
                'ylabel': 'MAD',
                'color': 'lightblue'
            }
        }
        plot_data['prob_range_boxplot'] = prob_range_boxplot_data
        
        plt.figure(figsize=(8, 6), dpi=200)
        box_data = [prob_range_data[i] for i in range(len(prob_bins) - 1) if len(prob_range_data[i]) > 0]
        valid_labels = [prob_range_labels[i] for i in range(len(prob_bins) - 1) if len(prob_range_data[i]) > 0]
        
        if box_data:
            plt.boxplot(box_data, labels=valid_labels, patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
            plt.xlabel('Probability Range')
            plt.ylabel('MAD')
            plt.title(f'Epistemic Uncertainty by Probability Range ({ensemble_name})')
            plt.xticks(rotation=45)  # Rotate labels for better readability
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'prob_range_boxplot.png'), dpi=200, bbox_inches='tight')
        plt.close()
        
        # 3. Term uncertainty analysis
        logger.info("Creating term uncertainty analysis...")
        
        rows = []
        N, C = uncertainty_vals.shape
        for i in range(N):
            true_classes = np.where(test_labels[i])[0]
            for j in true_classes:
                rows.append({
                    'GO_term': mlb.classes_[j],
                    'MAD': uncertainty_vals[i, j],
                    'median_prob': central_probs[i, j]
                })
        
        df_terms = pd.DataFrame(rows)
        
        # Summarize by GO term (only terms with >5 occurrences)
        df_summary = (
            df_terms
            .groupby('GO_term')['MAD']
            .agg(
                average_MAD='mean',
                count='size',
                std_MAD='std'
            )
            .reset_index()
        )
        df_summary = df_summary[df_summary['count'] > 5].sort_values('average_MAD', ascending=False)
        
        # Get top 10 most and least uncertain terms
        top_uncertain = df_summary.head(10)
        least_uncertain = df_summary.tail(10)
        
        # Save term analysis data
        term_analysis_data = {
            'top_uncertain_terms': {
                'go_terms': top_uncertain['GO_term'].tolist(),
                'avg_mad': top_uncertain['average_MAD'].tolist(),
                'counts': top_uncertain['count'].tolist(),
                'std_mad': top_uncertain['std_MAD'].tolist()
            },
            'least_uncertain_terms': {
                'go_terms': least_uncertain['GO_term'].tolist(),
                'avg_mad': least_uncertain['average_MAD'].tolist(),
                'counts': least_uncertain['count'].tolist(),
                'std_mad': least_uncertain['std_MAD'].tolist()
            },
            'all_terms_summary': {
                'go_terms': df_summary['GO_term'].tolist(),
                'avg_mad': df_summary['average_MAD'].tolist(),
                'counts': df_summary['count'].tolist()
            },
            'ensemble_method': ensemble_name,
            'plot_config': {
                'figsize': [10, 6],
                'title': f'Top 10 Most Uncertain GO Terms ({ensemble_name})',
                'xlabel': 'GO Term',
                'ylabel': 'Average MAD',
                'rotation': 45,
                'color': 'coral'
            }
        }
        plot_data['term_analysis'] = term_analysis_data
        
        if len(top_uncertain) > 0:
            plt.figure(figsize=(12, 6), dpi=200)
            plt.bar(range(len(top_uncertain)), top_uncertain['average_MAD'], color='coral', alpha=0.7)
            plt.xticks(range(len(top_uncertain)), top_uncertain['GO_term'], rotation=45, ha='right')
            plt.xlabel('GO Term')
            plt.ylabel('Average MAD')
            plt.title(f'Top 10 Most Uncertain GO Terms ({ensemble_name})')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'term_uncertainty_ranking.png'), dpi=200, bbox_inches='tight')
            plt.close()
        
        # 4. Term frequency vs uncertainty scatter plot
        logger.info("Creating term frequency vs uncertainty scatter plot...")
        
        frequency_uncertainty_data = {
            'term_counts': df_summary['count'].tolist(),
            'avg_uncertainties': df_summary['average_MAD'].tolist(),
            'go_terms': df_summary['GO_term'].tolist(),
            'ensemble_method': ensemble_name,
            'plot_config': {
                'figsize': [8, 6],
                'title': f'Term Frequency vs Average Uncertainty ({ensemble_name})',
                'xlabel': 'Count of True Positives per GO Term',
                'ylabel': 'Average MAD for GO Term',
                'alpha': 0.6,
                'color': 'darkblue',
                'xscale': 'log'
            }
        }
        plot_data['frequency_uncertainty_scatter'] = frequency_uncertainty_data
        
        plt.figure(figsize=(8, 6), dpi=200)
        plt.scatter(df_summary['count'], df_summary['average_MAD'], alpha=0.6, color='darkblue')
        plt.xlabel('Count of True Positives per GO Term')
        plt.ylabel('Average MAD for GO Term')
        plt.title(f'Term Frequency vs Average Uncertainty ({ensemble_name})')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'frequency_uncertainty_scatter.png'), dpi=200, bbox_inches='tight')
        plt.close()
        
    else:
        logger.warning("No ensemble data available for additional uncertainty plots")
    
    if save_plot_data:
        logger.info("Saving plot data for custom replotting...")
        plot_data_dir = os.path.join(plots_dir, 'data')
        os.makedirs(plot_data_dir, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_for_json(obj):
            """Recursively convert numpy types to Python types for JSON serialization."""
            if isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj
        
        # Save as JSON for easy reading
        json_safe_plot_data = convert_for_json(plot_data)
        with open(os.path.join(plot_data_dir, 'plot_data.json'), 'w') as f:
            json.dump(json_safe_plot_data, f, indent=2)
        
        # Save as pickle for Python users (preserves numpy arrays)
        with open(os.path.join(plot_data_dir, 'plot_data.pkl'), 'wb') as f:
            pickle.dump(plot_data, f)
        
        logger.info(f"Plot data saved to {plot_data_dir}")
        


def save_results(uncertainty_results: Dict, performance_dfs: Dict, 
                predicted_terms_dicts: Dict, config: Dict, thresholds: np.ndarray,
                normality_results: Dict, robustness_data: Tuple,
                output_dir: str, save_extended: bool, logger: logging.Logger):
    logger.info("Saving results...")
    
    arrays_dir = os.path.join(output_dir, 'arrays')
    metrics_dir = os.path.join(output_dir, 'metrics')
    predictions_dir = os.path.join(output_dir, 'predictions')
    
    for dir_path in [arrays_dir, metrics_dir, predictions_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    if save_extended:
        extended_arrays_dir = os.path.join(arrays_dir, 'extended')
        os.makedirs(extended_arrays_dir, exist_ok=True)
    
    for method_type, methods in uncertainty_results.items():
        if method_type == 'deterministic':
            np.save(os.path.join(arrays_dir, 'deterministic_probs.npy'), methods['central'])
        else:
            for method_name, method_data in methods.items():
                if 'probs' in method_data:
                    np.save(os.path.join(arrays_dir, f'{method_name}_probs.npy'), method_data['probs'])
                np.save(os.path.join(arrays_dir, f'{method_name}_central.npy'), method_data['central'])
                np.save(os.path.join(arrays_dir, f'{method_name}_uncertainty.npy'), method_data['uncertainty'])
    
    for method_name, df in performance_dfs.items():
        df.to_csv(os.path.join(metrics_dir, f'{method_name}_metrics.csv'), index=False)
    
    for method_name, pred_dict in predicted_terms_dicts.items():
        with open(os.path.join(predictions_dir, f'predicted_terms_{method_name}.pkl'), 'wb') as f:
            pickle.dump(pred_dict, f)
    
    config_data = {
        'config': config,
        'thresholds': thresholds.tolist(),
        'normality_results': normality_results,
        'save_extended': save_extended
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_data, f, indent=2)
    
    if robustness_data[0].size > 0:
        robustness_dir = os.path.join(output_dir, 'robustness')
        os.makedirs(robustness_dir, exist_ok=True)
        
        noise_levels, all_tp_mads, avg_mads = robustness_data
        
        with open(os.path.join(robustness_dir, 'all_tp_mads.pkl'), 'wb') as f:
            pickle.dump(all_tp_mads, f)
        
        robustness_df = pd.DataFrame({
            'noise_level': noise_levels,
            'avg_mad': avg_mads
        })
        robustness_df.to_csv(os.path.join(robustness_dir, 'robustness_metrics.csv'), index=False)


def main():
    parser = argparse.ArgumentParser(
        description='Uncertainty Analysis for Protein Function Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument('--ontology', required=True, choices=['BPO', 'CCO', 'MFO'],
                       help='Ontology type to analyze')
    parser.add_argument('--models_dir', required=True,
                       help='Directory containing model weights (e.g., bp_publish)')
    
    # Optional arguments
    parser.add_argument('--data_dir', default='processed_data_90_30',
                       help='Directory containing test data')
    parser.add_argument('--config_file', default='TUNED_MODEL_ARCHS.json',
                       help='JSON file containing model architectures')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory (default: uncertainty_results_{ontology_name})')
    
    # Uncertainty method configuration
    parser.add_argument('--uncertainty_metric', choices=['mad', 'std', 'var'], default='mad',
                       help='Uncertainty aggregation method')
    parser.add_argument('--central_tendency', choices=['median', 'mean', 'auto'], default='auto',
                       help='Central tendency measure (auto=median for most methods, mean for temporal)')
    
    # Method selection
    parser.add_argument('--mcd_passes', nargs='+', type=int, default=[10, 20, 30, 40, 50],
                       help='Number of MC Dropout passes to evaluate')
    parser.add_argument('--ensemble_folds', nargs='+', type=int, default=[5, 10, 20],
                       help='K-fold ensemble sizes to use')
    parser.add_argument('--skip_mcd', action='store_true',
                       help='Skip Monte Carlo Dropout analysis')
    parser.add_argument('--skip_temporal', action='store_true',
                       help='Skip temporal ensemble analysis')
    parser.add_argument('--skip_ensemble', action='store_true', 
                       help='Skip deep ensemble analysis')
    parser.add_argument('--skip_robustness', action='store_true',
                       help='Skip robustness testing with noise injection')
    
    # Analysis configuration
    parser.add_argument('--thresholds', nargs='+', type=float, default=None,
                       help='Custom threshold values (default: 0.01 to 0.99 by 0.01)')
    parser.add_argument('--noise_levels', type=int, default=30,
                       help='Number of noise levels for robustness testing')
    
    # Output configuration
    parser.add_argument('--save_extended', action='store_true',
                       help='Save extended analysis with detailed outputs')
    parser.add_argument('--save_plot_data', action='store_true',
                       help='Save raw data used to create plots for custom replotting')
    parser.add_argument('--generate_report', action='store_true',
                       help='Generate comprehensive HTML report')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup configuration
    ontology_config = get_ontology_config(args.ontology)
    ontology_name = ontology_config['name']
    
    if args.output_dir is None:
        args.output_dir = f'uncertainty_results_{ontology_name}'
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir, args.verbose)
    logger.info(f"Starting uncertainty analysis for {args.ontology}")
    logger.info(f"Output directory: {args.output_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        test_embeddings, test_labels, test_GO_df, test_GO_list, mlb = load_test_data(
            ontology_name, args.data_dir, logger
        )
        
        test_dataset = TensorDataset(
            torch.tensor(test_embeddings, dtype=torch.float).to(device),
            torch.tensor(test_labels, dtype=torch.float).to(device)
        )
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
        
        models = load_models(args.models_dir, args.ontology, args.config_file, device, logger)
        
        if args.thresholds is None:
            thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)  # Round to 2 decimal places
        else:
            thresholds = np.round(np.array(args.thresholds), 2)
        
        uncertainty_results = compute_uncertainties(
            models, test_loader, args.mcd_passes, args.uncertainty_metric,
            args.central_tendency, device, logger, args.skip_mcd, 
            args.skip_temporal, args.skip_ensemble
        )
        
        predicted_terms_dicts = {}
        for method_type, methods in uncertainty_results.items():
            if method_type == 'deterministic':
                predicted_terms_dicts['deterministic'] = store_predicted_terms(
                    methods['central'], methods['uncertainty'], thresholds, mlb
                )
            else:
                for method_name, method_data in methods.items():
                    predicted_terms_dicts[method_name] = store_predicted_terms(
                        method_data['central'], method_data['uncertainty'], thresholds, mlb
                    )
        
        performance_dfs = analyze_performance(
            uncertainty_results, test_labels, test_GO_list, mlb, thresholds, logger
        )
        
        normality_results = test_normality(uncertainty_results, test_labels, logger, 1000)
        
        robustness_data = (np.array([]), [], [])
        if not args.skip_robustness:
            robustness_data = robustness_analysis(
                models, test_embeddings, test_labels, args.noise_levels, device, logger
            )
        
        create_plots(performance_dfs, uncertainty_results, test_labels, 
                    normality_results, robustness_data, mlb, args.output_dir, args.save_plot_data, logger)
        
        save_results(
            uncertainty_results, performance_dfs, predicted_terms_dicts,
            vars(args), thresholds, normality_results, robustness_data,
            args.output_dir, args.save_extended, logger
        )
        
        logger.info("Analysis completed successfully!")
        
        print(f"\n{'='*60}")
        print(f"UNCERTAINTY ANALYSIS SUMMARY - {args.ontology}")
        print(f"{'='*60}")
        print(f"Output directory: {args.output_dir}")
        print(f"Test samples: {len(test_GO_df)}")
        print(f"GO terms: {test_labels.shape[1]}")
        print(f"Methods analyzed: {list(performance_dfs.keys())}")
        print(f"Uncertainty metric: {args.uncertainty_metric}")
        if args.central_tendency == 'auto':
            print(f"Central tendency: auto (median for MC Dropout/Ensembles, mean for temporal)")
        else:
            print(f"Central tendency: {args.central_tendency}")
        if robustness_data[0].size > 0:
            print(f"Robustness noise levels: {len(robustness_data[0])}")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 