#!/usr/bin/env python3
"""
Evaluate BLAST Baseline Results using CAFA metrics

This script evaluates BLAST baseline predictions against ground truth annotations
using the evaluate_annotations function from utils_corrected.py.

Usage:
    python evaluate_blast_baseline.py --predictions baseline.pkl --ground_truth test_dict.pkl --ic_dict ic.pkl --ontology MF
"""

import argparse
import pickle
import sys
from pathlib import Path

# Import evaluation functions
from utils_corrected import evaluate_annotations, convert_predictions_to_set, threshold_performance_metrics
import pandas as pd
import numpy as np


def load_pickle_file(file_path, description):
    """
    Load a pickle file and return its contents.
    
    Args:
        file_path (str): Path to pickle file
        description (str): Description for error messages
        
    Returns:
        The loaded object
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Loaded {description}: {len(data)} entries")
        return data
    except Exception as e:
        print(f"Error loading {description} from {file_path}: {e}")
        sys.exit(1)


def load_ic_dict(file_path):
    """
    Load information content dictionary from either pickle file or TSV file.
    
    Args:
        file_path (str): Path to IC file (either .pkl or .tsv)
        
    Returns:
        dict: Mapping from GO term to IC/IA value
    """
    try:
        if file_path.endswith('.tsv'):
            # Load from TSV file (GO_term \t IA_value)
            ic_dict = {}
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        go_term = parts[0]
                        try:
                            ic_value = float(parts[1])
                            ic_dict[go_term] = ic_value
                        except ValueError:
                            continue
            print(f"✓ Loaded IC/IA values from TSV: {len(ic_dict)} entries")
            return ic_dict
        else:
            # Load from pickle file
            with open(file_path, 'rb') as f:
                ic_dict = pickle.load(f)
            print(f"✓ Loaded IC/IA values from pickle: {len(ic_dict)} entries")
            return ic_dict
    except Exception as e:
        print(f"Error loading IC/IA values from {file_path}: {e}")
        sys.exit(1)


def convert_ground_truth_to_sets(ground_truth):
    """
    Convert ground truth annotations to sets regardless of input format.
    
    Args:
        ground_truth (dict): Ground truth annotations in various formats
        
    Returns:
        dict: Ground truth annotations converted to sets
    """
    if not ground_truth:
        return ground_truth
    
    sample_value = next(iter(ground_truth.values()))
    
    if isinstance(sample_value, str):
        print("Converting ground truth strings to sets...")
        import ast
        ground_truth_sets = {}
        for k, v in ground_truth.items():
            try:
                # Try to parse as literal (e.g., "['GO:0001', 'GO:0002']")
                parsed = ast.literal_eval(v)
                if isinstance(parsed, (list, tuple)):
                    ground_truth_sets[k] = set(parsed)
                else:
                    ground_truth_sets[k] = {str(parsed)}
            except (ValueError, SyntaxError):
                # If parsing fails, treat as single term or split by common delimiters
                if ',' in v:
                    ground_truth_sets[k] = set(term.strip() for term in v.split(','))
                elif ';' in v:
                    ground_truth_sets[k] = set(term.strip() for term in v.split(';'))
                else:
                    ground_truth_sets[k] = {v.strip()}
    elif isinstance(sample_value, (list, tuple)):
        print("Converting ground truth lists to sets...")
        ground_truth_sets = {k: set(v) for k, v in ground_truth.items()}
    elif isinstance(sample_value, set):
        print("Ground truth already in set format...")
        ground_truth_sets = ground_truth
    else:
        print(f"Warning: Unknown ground truth format: {type(sample_value)}")
        ground_truth_sets = ground_truth
    
    return ground_truth_sets


def validate_data_format(predictions, ground_truth, ontology):
    """
    Validate that the data is in the expected format and show sample entries.
    
    Args:
        predictions (dict): Baseline predictions
        ground_truth (dict): Ground truth annotations  
        ontology (str): Ontology being evaluated
    """
    print(f"\n=== Data Format Validation for {ontology} ===")
    
    # Check predictions format
    if predictions:
        sample_pred_key = next(iter(predictions.keys()))
        sample_pred_value = predictions[sample_pred_key]
        print(f"Predictions sample - {sample_pred_key}: {type(sample_pred_value)} with {len(sample_pred_value)} terms")
        if isinstance(sample_pred_value, dict):
            sample_terms = list(sample_pred_value.keys())[:3]
            print(f"  Sample GO terms: {sample_terms}")
    
    # Check ground truth format  
    if ground_truth:
        sample_gt_key = next(iter(ground_truth.keys()))
        sample_gt_value = ground_truth[sample_gt_key]
        print(f"Ground truth sample - {sample_gt_key}: {type(sample_gt_value)} with {len(sample_gt_value)} terms")
        if hasattr(sample_gt_value, '__iter__') and not isinstance(sample_gt_value, str):
            sample_terms = list(sample_gt_value)[:3]
            print(f"  Sample GO terms: {sample_terms}")
    
    # Check overlap
    common_keys = set(predictions.keys()).intersection(set(ground_truth.keys()))
    print(f"Common entries between predictions and ground truth: {len(common_keys)}")
    
    if len(common_keys) == 0:
        print("WARNING: No common entries found between predictions and ground truth!")
        print("This could indicate a sequence ID mismatch.")
        return False
    
    return True


def evaluate_baseline_predictions(predictions_file, ground_truth_file, ic_dict_file, ontology):
    """
    Evaluate BLAST baseline predictions against ground truth.
    
    Args:
        predictions_file (str): Path to baseline predictions pickle file
        ground_truth_file (str): Path to ground truth annotations pickle file  
        ic_dict_file (str): Path to information content dictionary (.pkl) or TSV file (.tsv)
        ontology (str): Ontology being evaluated (for reporting)
        
    Returns:
        tuple: Evaluation metrics
    """
    print(f"Evaluating {ontology} BLAST Baseline Predictions")
    print("=" * 60)
    
    # Load data
    predictions = load_pickle_file(predictions_file, "baseline predictions")
    ground_truth = load_pickle_file(ground_truth_file, "ground truth annotations")
    ic_dict = load_ic_dict(ic_dict_file)
    
    # Validate data format
    if not validate_data_format(predictions, ground_truth, ontology):
        return None
    
    # Convert predictions to sets (remove scores)
    print("\nConverting predictions to sets...")
    pred_sets = convert_predictions_to_set(predictions)
    
    # Ensure ground truth is in set format
    ground_truth_sets = convert_ground_truth_to_sets(ground_truth)
    
    # Evaluate
    print(f"\nEvaluating {ontology} predictions...")
    try:
        results = evaluate_annotations(ic_dict, ground_truth_sets, pred_sets)
        f, p, r, s, ru, mi, f_micro, p_micro, r_micro, tp_global, fp_global, fn_global = results
        
        return {
            'ontology': ontology,
            'f1_macro': f,
            'precision_macro': p, 
            'recall_macro': r,
            'semantic_distance': s,
            'remaining_uncertainty': ru,
            'misinformation': mi,
            'f1_micro': f_micro,
            'precision_micro': p_micro,
            'recall_micro': r_micro,
            'true_positives': tp_global,
            'false_positives': fp_global,
            'false_negatives': fn_global,
            'num_test_proteins': len(set(ground_truth_sets.keys()).intersection(set(pred_sets.keys()))),
            'num_predictions': len(pred_sets),
            'total_predicted_terms': sum(len(terms) for terms in pred_sets.values())
        }
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None


def evaluate_with_thresholds(predictions_file, ground_truth_file, ic_dict_file, ontology, threshold_range, output_dir):
    """
    Evaluate BLAST baseline predictions across multiple thresholds.
    
    Args:
        predictions_file (str): Path to baseline predictions pickle file
        ground_truth_file (str): Path to ground truth annotations pickle file
        ic_dict_file (str): Path to information content dictionary (.pkl) or TSV file (.tsv)
        ontology (str): Ontology being evaluated
        threshold_range (np.array): Array of thresholds to evaluate
        output_dir (str): Directory to save results
        
    Returns:
        dict: Threshold evaluation results
    """
    print(f"Evaluating {ontology} BLAST Baseline Predictions Across Thresholds")
    print("=" * 70)
    
    # Load data
    predictions = load_pickle_file(predictions_file, "baseline predictions")
    ground_truth = load_pickle_file(ground_truth_file, "ground truth annotations")
    ic_dict = load_ic_dict(ic_dict_file)
    
    # Validate data format
    if not validate_data_format(predictions, ground_truth, ontology):
        return None
    
    # Ensure ground truth is in set format
    ground_truth_sets = convert_ground_truth_to_sets(ground_truth)
    
    # Run threshold evaluation (keeping scores for thresholding)
    print(f"\nRunning threshold evaluation for {ontology} across {len(threshold_range)} thresholds...")
    try:
        smin, fmax, best_threshold_s, best_threshold_f, s_at_fmax, results_df = threshold_performance_metrics(
            ic_dict, ground_truth_sets, predictions, threshold_range=threshold_range
        )
        
        # Save detailed results
        results_file = f"{output_dir}/{ontology}_blast_baseline_threshold_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"✓ Detailed threshold results saved to: {results_file}")
        
        # Save summary
        summary = {
            'ontology': ontology,
            'smin': smin,
            'fmax': fmax,
            'best_threshold_s': best_threshold_s,
            'best_threshold_f': best_threshold_f,
            's_at_fmax': s_at_fmax,
            'num_thresholds': len(threshold_range),
            'threshold_range': f"{threshold_range[0]:.3f}-{threshold_range[-1]:.3f}",
            'results_df': results_df
        }
        
        summary_file = f"{output_dir}/{ontology}_blast_baseline_summary.pkl"
        with open(summary_file, 'wb') as f:
            pickle.dump(summary, f)
        print(f"✓ Summary results saved to: {summary_file}")
        
        return summary
        
    except Exception as e:
        print(f"Error during threshold evaluation: {e}")
        return None


def print_results(results):
    """Print evaluation results in a formatted table."""
    if not results:
        print("No results to display.")
        return
        
    print(f"\n=== {results['ontology']} BLAST Baseline Evaluation Results ===")
    print(f"{'Metric':<25} {'Value':<15}")
    print("=" * 40)
    print(f"{'F1-score (macro)':<25} {results['f1_macro']:<15.4f}")
    print(f"{'Precision (macro)':<25} {results['precision_macro']:<15.4f}")
    print(f"{'Recall (macro)':<25} {results['recall_macro']:<15.4f}")
    print(f"{'F1-score (micro)':<25} {results['f1_micro']:<15.4f}")
    print(f"{'Precision (micro)':<25} {results['precision_micro']:<15.4f}")
    print(f"{'Recall (micro)':<25} {results['recall_micro']:<15.4f}")
    print(f"{'Semantic distance':<25} {results['semantic_distance']:<15.4f}")
    print(f"{'Remaining uncertainty':<25} {results['remaining_uncertainty']:<15.4f}")
    print(f"{'Misinformation':<25} {results['misinformation']:<15.4f}")
    print()
    print("=== Counts ===")
    print(f"{'True positives':<25} {results['true_positives']:<15}")
    print(f"{'False positives':<25} {results['false_positives']:<15}")
    print(f"{'False negatives':<25} {results['false_negatives']:<15}")
    print(f"{'Test proteins':<25} {results['num_test_proteins']:<15}")
    print(f"{'Proteins with predictions':<25} {results['num_predictions']:<15}")
    print(f"{'Total predicted terms':<25} {results['total_predicted_terms']:<15}")


def print_threshold_results(results):
    """Print threshold evaluation results."""
    if not results:
        print("No results to display.")
        return
        
    print(f"\n=== {results['ontology']} BLAST Baseline Threshold Evaluation ===")
    print(f"{'Metric':<25} {'Value':<15}")
    print("=" * 40)
    print(f"{'S-min':<25} {results['smin']:<15.4f}")
    print(f"{'F-max':<25} {results['fmax']:<15.4f}")
    print(f"{'Best threshold (S-min)':<25} {results['best_threshold_s']:<15.4f}")
    print(f"{'Best threshold (F-max)':<25} {results['best_threshold_f']:<15.4f}")
    print(f"{'S at F-max threshold':<25} {results['s_at_fmax']:<15.4f}")
    print(f"{'Number of thresholds':<25} {results['num_thresholds']:<15}")
    print(f"{'Threshold range':<25} {results['threshold_range']:<15}")
    print()
    print(f"Detailed results saved in CSV format for plotting and analysis.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BLAST baseline predictions using CAFA metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate MF baseline
  python evaluate_blast_baseline.py \\
    --predictions blast_foldseek_results/swissprot_MF_baseline_filtered.pkl \\
    --ground_truth blast_foldseek_results/function_test_dict.pkl \\
    --ic_dict blast_foldseek_results/swissprot_MF_progated_filtered_mlb.pkl \\
    --ontology MF

  # Evaluate BP baseline  
  python evaluate_blast_baseline.py \\
    --predictions blast_foldseek_results/swissprot_BP_baseline_filtered.pkl \\
    --ground_truth blast_foldseek_results/process_test_dict.pkl \\
    --ic_dict blast_foldseek_results/swissprot_BP_progated_filtered_mlb.pkl \\
    --ontology BP
        """
    )
    
    parser.add_argument('--predictions', required=True,
                       help='Path to baseline predictions pickle file')
    parser.add_argument('--ground_truth', required=True,
                       help='Path to ground truth annotations pickle file')
    parser.add_argument('--ic_dict', required=True,
                       help='Path to information content dictionary (.pkl file) or IA values (.tsv file)')
    parser.add_argument('--ontology', required=True, choices=['BP', 'MF', 'CC'],
                       help='Ontology being evaluated')
    parser.add_argument('--output_file', 
                       help='Optional: Save results to pickle file')
    parser.add_argument('--threshold_evaluation', action='store_true',
                       help='Perform threshold evaluation (S-min, F-max) across multiple thresholds')
    parser.add_argument('--threshold_range', type=str, default='0.01,1.0,0.01',
                       help='Threshold range as start,stop,step (default: 0.01,1.0,0.01)')
    parser.add_argument('--output_dir', default='evaluation_results',
                       help='Directory to save detailed results (default: evaluation_results)')
    
    args = parser.parse_args()
    
    # Validate input files
    for file_path in [args.predictions, args.ground_truth, args.ic_dict]:
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.threshold_evaluation:
        # Parse threshold range
        try:
            start, stop, step = map(float, args.threshold_range.split(','))
            threshold_range = np.arange(start, stop + step, step)
            print(f"Evaluating across {len(threshold_range)} thresholds: {start} to {stop} (step: {step})")
        except ValueError:
            print(f"Error: Invalid threshold range format: {args.threshold_range}")
            print("Expected format: start,stop,step (e.g., 0.01,1.0,0.01)")
            sys.exit(1)
        
        # Run threshold evaluation
        results = evaluate_with_thresholds(
            args.predictions, 
            args.ground_truth, 
            args.ic_dict, 
            args.ontology,
            threshold_range,
            args.output_dir
        )
    else:
        # Run single evaluation
        results = evaluate_baseline_predictions(
            args.predictions, 
            args.ground_truth, 
            args.ic_dict, 
            args.ontology
        )
    
    if results:
        # Print results
        if args.threshold_evaluation:
            print_threshold_results(results)
        else:
            print_results(results)
        
        # Save results if requested
        if args.output_file:
            with open(args.output_file, 'wb') as f:
                pickle.dump(results, f)
            print(f"\n✓ Results saved to: {args.output_file}")
    else:
        print("Evaluation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
