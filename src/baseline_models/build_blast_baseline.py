#!/usr/bin/env python3
"""
BLAST Baseline Builder for CAFA Evaluation

This script builds BLAST baseline predictions by transferring GO terms from training/SwissProt sequences
to test sequences based on BLAST search results. It processes one ontology at a time.

The script expects pre-propagated GO term mappings for both training and SwissProt data.

Usage:
    python build_blast_baseline.py --blast_results results.tsv --mapping_file annotations.tsv --output_file baseline.pkl --ontology BP
"""

import ast
import argparse
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

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
                go_terms = ast.literal_eval(go_terms_str)
                if isinstance(go_terms, list):
                    annotations[seq_id] = set(go_terms)
                    
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Error parsing line {line_num}: {e}")
                continue
    
    print(f"✓ Loaded annotations for {len(annotations)} sequences")
    return annotations





def parse_blast_results(blast_file, evalue_threshold=1e-5):
    """
    Parse BLAST results and extract query-subject mappings with scores.
    
    Args:
        blast_file (str): Path to BLAST results TSV file
        evalue_threshold (float): E-value threshold for filtering hits
        
    Returns:
        dict: Mapping from query_id to list of (subject_id, percent_identity, evalue)
    """
    print(f"Parsing BLAST results from {blast_file}...")
    blast_hits = defaultdict(list)
    
    total_lines = 0
    filtered_lines = 0
    
    with open(blast_file, 'r') as f:
        for line in tqdm(f, desc="Reading BLAST results"):
            try:
                cols = line.strip().split('\t')
                if len(cols) < 11:
                    continue
                    
                query_id = cols[0]
                subject_id = cols[1]
                pident = float(cols[2])
                evalue = float(cols[10])
                
                total_lines += 1
                
                if evalue <= evalue_threshold:
                    blast_hits[query_id].append((subject_id, pident, evalue))
                    filtered_lines += 1
                    
            except (ValueError, IndexError) as e:
                continue
    
    print(f"✓ Parsed {total_lines} BLAST hits, {filtered_lines} passed E-value filter")
    return dict(blast_hits)


def build_baseline_predictions(blast_hits, annotations, ontology):
    """
    Build baseline predictions by transferring GO terms from BLAST hits.
    
    Args:
        blast_hits (dict): Query to hits mapping from parse_blast_results
        annotations (dict): Sequence to GO terms mapping (pre-filtered by ontology)
        ontology (str): The ontology being processed (BP, MF, or CC)
        
    Returns:
        dict: Dictionary mapping query_id to {go_term: score, ...}
    """
    predictions = {}
    
    print(f"Building {ontology} predictions...")
    
    for query_id in tqdm(blast_hits, desc="Processing queries"):
        hits = blast_hits[query_id]
        
        # Collect GO terms and scores from all hits
        go_term_scores = defaultdict(list)  # go_term -> [score1, score2, ...]
        
        for subject_id, pident, evalue in hits:
            # Extract UniProt accession from full format (e.g., sp|P12345|PROTEIN_HUMAN -> P12345)
            if subject_id.startswith('sp|') and '|' in subject_id:
                accession = subject_id.split('|')[1]
            else:
                accession = subject_id  # Use as-is if not in sp| format
            
            if accession in annotations:
                # Convert percent identity to 0-1 scale
                score = pident / 100.0
                
                # Add all GO terms from this hit (already filtered by ontology in mapping file)
                for go_term in annotations[accession]:
                    go_term_scores[go_term].append(score)
        
        # Take maximum score for duplicate terms
        if go_term_scores:
            predictions[query_id] = {}
            for go_term, scores in go_term_scores.items():
                predictions[query_id][go_term] = max(scores)
    
    # Print statistics
    num_queries = len(predictions)
    total_predictions = sum(len(terms) for terms in predictions.values())
    print(f"  {ontology}: {num_queries} queries, {total_predictions} total predictions")
    
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
        description="Build BLAST baseline predictions for CAFA evaluation (one ontology at a time)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process BP search results
  python build_blast_baseline.py \\
    --blast_results blast_search_results/training_searches/process_vs_training.tsv \\
    --mapping_file training_BP_annotations.tsv \\
    --output_file training_BP_baseline.pkl \\
    --ontology BP

  # Process MF SwissProt search results  
  python build_blast_baseline.py \\
    --blast_results blast_search_results/swissprot_searches/function_vs_swissprot.tsv \\
    --mapping_file swissprot_MF_annotations.tsv \\
    --output_file swissprot_MF_baseline.pkl \\
    --ontology MF
        """
    )
    
    parser.add_argument('--blast_results', required=True,
                       help='Path to BLAST results TSV file')
    parser.add_argument('--mapping_file', required=True,
                       help='Path to sequence annotations TSV file (id \\t [GO:terms]) - pre-filtered by ontology')
    parser.add_argument('--output_file', required=True,
                       help='Output pickle file path')
    parser.add_argument('--ontology', required=True, choices=['BP', 'MF', 'CC'],
                       help='Ontology being processed (BP, MF, or CC)')
    parser.add_argument('--evalue_threshold', type=float, default=1e-5,
                       help='E-value threshold for filtering BLAST hits (default: 1e-5)')
    
    args = parser.parse_args()
    
    # Validate input files
    for file_path in [args.blast_results, args.mapping_file]:
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
    
    # Validate ontology
    if not validate_ontology(args.ontology):
        print(f"Error: Invalid ontology '{args.ontology}'. Must be BP, MF, or CC.")
        sys.exit(1)
    
    print("BLAST Baseline Builder")
    print("=" * 50)
    print(f"BLAST results: {args.blast_results}")
    print(f"Mapping file: {args.mapping_file}")
    print(f"Output file: {args.output_file}")
    print(f"Ontology: {args.ontology}")
    print(f"E-value threshold: {args.evalue_threshold}")
    print()
    
    start_time = time.time()
    
    try:
        # Load sequence annotations (pre-filtered by ontology)
        annotations = load_sequence_annotations(args.mapping_file)
        
        # Parse BLAST results
        blast_hits = parse_blast_results(args.blast_results, args.evalue_threshold)
        
        if not blast_hits:
            print("Warning: No BLAST hits found after filtering!")
            sys.exit(1)
        
        # Build predictions
        print("\n" + "="*50)
        predictions = build_baseline_predictions(blast_hits, annotations, args.ontology)
        
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
