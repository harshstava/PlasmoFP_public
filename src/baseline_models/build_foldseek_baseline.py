#!/usr/bin/env python3
"""
FoldSeek Baseline Builder for CAFA Evaluation

This script builds FoldSeek baseline predictions by transferring GO terms from training/SwissProt sequences
to test sequences based on FoldSeek search results. It processes one ontology at a time and uses bit scores
with sigmoid normalization for confidence scoring.

The script expects pre-propagated GO term mappings for both training and SwissProt data.

Usage:
    python build_foldseek_baseline.py --foldseek_results results.tsv --mapping_file annotations.tsv --output_file baseline.pkl --ontology BP
"""

import argparse
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


def validate_ontology(ontology):
    """
    Validate that the provided ontology is one of the supported types.
    
    Args:
        ontology (str): The ontology to validate (BP, MF, or CC)
        
    Returns:
        bool: True if valid, False otherwise
    """
    valid_ontologies = {'BP', 'MF', 'CC'}
    return ontology in valid_ontologies


def load_sequence_annotations(mapping_file):
    """
    Load sequence ID to GO terms mapping from TSV file.
    
    Args:
        mapping_file (str): Path to TSV file with format: sequence_id \t ['GO:term1', 'GO:term2', ...]
        
    Returns:
        dict: Mapping from sequence_id to set of GO terms
    """
    print(f"Loading sequence annotations from {mapping_file}...")
    annotations = {}
    
    with open(mapping_file, 'r') as f:
        for line_num, line in enumerate(tqdm(f, desc="Reading annotations"), 1):
            try:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                    
                seq_id = parts[0]
                go_terms_str = parts[1]
                
                # Parse GO terms list
                import ast
                go_terms = ast.literal_eval(go_terms_str)
                if isinstance(go_terms, list):
                    annotations[seq_id] = set(go_terms)
                    
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Error parsing line {line_num}: {e}")
                continue
    
    print(f"✓ Loaded annotations for {len(annotations)} sequences")
    return annotations


def parse_foldseek_results(foldseek_file, evalue_threshold=1e-5):
    """
    Parse FoldSeek results and extract query-subject mappings with bit scores.
    
    Args:
        foldseek_file (str): Path to FoldSeek results TSV file
        evalue_threshold (float): E-value threshold for filtering hits
        
    Returns:
        dict: Mapping from query_id to list of (subject_id, bit_score, evalue)
    """
    print(f"Parsing FoldSeek results from {foldseek_file}...")
    foldseek_hits = defaultdict(list)
    
    total_lines = 0
    filtered_lines = 0
    
    with open(foldseek_file, 'r') as f:
        for line in tqdm(f, desc="Reading FoldSeek results"):
            try:
                cols = line.strip().split('\t')
                if len(cols) < 12:
                    continue
                    
                query_id = cols[0]
                subject_id = cols[1]
                # Skip sequence identity (cols[2]) - we use bit score instead
                evalue = float(cols[10])  # E-value is column 11 (0-indexed: 10)
                bit_score = float(cols[11])  # Bit score is column 12 (0-indexed: 11)
                
                total_lines += 1
                
                if evalue <= evalue_threshold:
                    foldseek_hits[query_id].append((subject_id, bit_score, evalue))
                    filtered_lines += 1
                    
            except (ValueError, IndexError) as e:
                continue
    
    print(f"✓ Parsed {total_lines} FoldSeek hits, {filtered_lines} passed E-value filter")
    return dict(foldseek_hits)


def normalize_bit_scores_sigmoid(bit_scores):
    """
    Normalize bit scores using adaptive sigmoid transformation.
    
    Args:
        bit_scores (list): List of bit scores
        
    Returns:
        tuple: (scaling_factor, normalization_function)
    """
    if not bit_scores:
        return 1.0, lambda x: 0.5
    
    bit_scores_array = np.array(bit_scores)
    scaling_factor = np.std(bit_scores_array)
    
    # Ensure minimum scaling factor to avoid over-steep curves
    scaling_factor = max(scaling_factor, 10.0)
    
    def normalize_score(bit_score):
        return 1.0 / (1.0 + np.exp(-bit_score / scaling_factor))
    
    print(f"✓ Adaptive scaling factor: {scaling_factor:.2f}")
    return scaling_factor, normalize_score


def build_foldseek_baseline_predictions(foldseek_hits, annotations, ontology):
    """
    Build baseline predictions by transferring GO terms from FoldSeek hits using bit scores.
    
    Args:
        foldseek_hits (dict): Query to hits mapping from parse_foldseek_results
        annotations (dict): Sequence to GO terms mapping (pre-filtered by ontology)
        ontology (str): The ontology being processed (BP, MF, or CC)
        
    Returns:
        dict: Dictionary mapping query_id to {go_term: score, ...}
    """
    predictions = {}
    
    print(f"Building {ontology} predictions using bit scores...")
    
    # Collect all bit scores for adaptive normalization
    all_bit_scores = []
    for hits in foldseek_hits.values():
        for _, bit_score, _ in hits:
            all_bit_scores.append(bit_score)
    
    if not all_bit_scores:
        print("Warning: No bit scores found!")
        return predictions
    
    # Get normalization function
    scaling_factor, normalize_score = normalize_bit_scores_sigmoid(all_bit_scores)
    
    print(f"Bit score range: {min(all_bit_scores):.1f} - {max(all_bit_scores):.1f}")
    
    for query_id in tqdm(foldseek_hits, desc="Processing queries"):
        hits = foldseek_hits[query_id]
        
        # Collect GO terms and scores from all hits
        go_term_scores = defaultdict(list)  # go_term -> [score1, score2, ...]
        
        for subject_id, bit_score, evalue in hits:
            # Extract UniProt accession from AlphaFold ID or use directly
            if subject_id.startswith('AF-') and '-F1-model_v4' in subject_id:
                # Extract from AF-Q9P6S0-F1-model_v4 -> Q9P6S0
                accession = subject_id.split('-')[1]
            else:
                accession = subject_id  # Use as-is if not in AF- format
            
            if accession in annotations:
                # Normalize bit score using sigmoid
                normalized_score = normalize_score(bit_score)
                
                # Add all GO terms from this hit (already filtered by ontology in mapping file)
                for go_term in annotations[accession]:
                    go_term_scores[go_term].append(normalized_score)
        
        # Take maximum score for duplicate terms
        if go_term_scores:
            predictions[query_id] = {}
            for go_term, scores in go_term_scores.items():
                predictions[query_id][go_term] = max(scores)
    
    # Print statistics
    num_queries = len(predictions)
    total_predictions = sum(len(terms) for terms in predictions.values())
    print(f"  {ontology}: {num_queries} queries, {total_predictions} total predictions")
    
    # Print normalization examples
    if all_bit_scores:
        example_scores = [min(all_bit_scores), np.median(all_bit_scores), max(all_bit_scores)]
        print("  Normalization examples:")
        for score in example_scores:
            normalized = normalize_score(score)
            print(f"    Bit score {score:.1f} → {normalized:.3f}")
    
    return predictions


def save_predictions(predictions, output_file):
    """
    Save prediction dictionary to pickle file.
    
    Args:
        predictions (dict): Dictionary with predictions for one ontology
        output_file (str): Output file path
    """
    with open(output_file, 'wb') as f:
        pickle.dump(predictions, f)
    
    num_queries = len(predictions)
    total_predictions = sum(len(terms) for terms in predictions.values())
    print(f"Saved {output_file}: {num_queries} queries, {total_predictions} predictions")


def main():
    parser = argparse.ArgumentParser(
        description="Build FoldSeek baseline predictions for CAFA evaluation (one ontology at a time)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process BP search results
  python build_foldseek_baseline.py \\
    --foldseek_results foldseek_search_results/training_searches/process_vs_training.tsv \\
    --mapping_file training_BP_annotations.tsv \\
    --output_file training_BP_foldseek_baseline.pkl \\
    --ontology BP

  # Process MF SwissProt search results  
  python build_foldseek_baseline.py \\
    --foldseek_results foldseek_search_results/swissprot_searches/function_vs_swissprot.tsv \\
    --mapping_file swissprot_MF_annotations.tsv \\
    --output_file swissprot_MF_foldseek_baseline.pkl \\
    --ontology MF
        """
    )
    
    parser.add_argument('--foldseek_results', required=True,
                       help='Path to FoldSeek results TSV file')
    parser.add_argument('--mapping_file', required=True,
                       help='Path to sequence annotations TSV file (id \\t [GO:terms]) - pre-filtered by ontology')
    parser.add_argument('--output_file', required=True,
                       help='Output pickle file path')
    parser.add_argument('--ontology', required=True, choices=['BP', 'MF', 'CC'],
                       help='Ontology being processed (BP, MF, or CC)')
    parser.add_argument('--evalue_threshold', type=float, default=1e-5,
                       help='E-value threshold for filtering FoldSeek hits (default: 1e-5)')
    
    args = parser.parse_args()
    
    # Validate input files
    for file_path in [args.foldseek_results, args.mapping_file]:
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
    
    # Validate ontology
    if not validate_ontology(args.ontology):
        print(f"Error: Invalid ontology '{args.ontology}'. Must be BP, MF, or CC.")
        sys.exit(1)
    
    print("FoldSeek Baseline Builder")
    print("=" * 50)
    print(f"FoldSeek results: {args.foldseek_results}")
    print(f"Mapping file: {args.mapping_file}")
    print(f"Output file: {args.output_file}")
    print(f"Ontology: {args.ontology}")
    print(f"E-value threshold: {args.evalue_threshold}")
    print()
    
    start_time = time.time()
    
    try:
        # Load sequence annotations (pre-filtered by ontology)
        annotations = load_sequence_annotations(args.mapping_file)
        
        # Parse FoldSeek results
        foldseek_hits = parse_foldseek_results(args.foldseek_results, args.evalue_threshold)
        
        if not foldseek_hits:
            print("Warning: No FoldSeek hits found after filtering!")
            sys.exit(1)
        
        # Build predictions
        print("\n" + "="*50)
        predictions = build_foldseek_baseline_predictions(foldseek_hits, annotations, args.ontology)
        
        if not predictions:
            print(f"Warning: No predictions generated for {args.ontology}!")
            sys.exit(1)
        
        # Save predictions
        print("\n" + "="*50)
        print("Saving prediction dictionary...")
        save_predictions(predictions, args.output_file)
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Completed successfully in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

