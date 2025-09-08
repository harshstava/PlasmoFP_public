#!/usr/bin/env python3
"""
Build GO Term Co-occurrence Matrices from SwissProt Data 

This script processes SwissProt GO annotations to create co-occurrence matrices
that count how often pairs of GO terms appear together on the same protein.

Generates separate matrices for:
- All GO terms combined
- Molecular Function (MF) only  
- Biological Process (BP) only
- Cellular Component (CC) only

USAGE:
    python build_cooccurrence_matrices_fixed.py --input uniprotkb_reviewed_true_2025_08_07.tsv --output_dir matrices_fixed/
"""

import argparse
import os
import pickle
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm


def setup_logging(output_dir: str, verbose: bool = False) -> logging.Logger:
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logger = logging.getLogger('cooccurrence')
    logger.setLevel(log_level)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'cooccurrence_build_log.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def extract_go_ids_from_text(text: str) -> Set[str]:
    """Extract GO IDs from text descriptions using regex."""
    if pd.isna(text) or not text.strip():
        return set()
    
    go_pattern = r'(?:\[)?GO:\d{7}(?:\])?'
    matches = re.findall(go_pattern, text)
    
    go_terms = set()
    for match in matches:
        clean_go = match.replace('[', '').replace(']', '')
        if clean_go.startswith('GO:'):
            go_terms.add(clean_go)
    
    return go_terms


def parse_go_terms_from_columns(row: pd.Series) -> Dict[str, Set[str]]:
    """Parse GO terms from specific ontology columns."""
    categories = {
        'MF': set(),  # Molecular Function
        'BP': set(),  # Biological Process
        'CC': set(),  # Cellular Component
        'ALL': set()  # All terms combined
    }
    
    mf_terms = extract_go_ids_from_text(row.get('Gene Ontology (molecular function)', ''))
    bp_terms = extract_go_ids_from_text(row.get('Gene Ontology (biological process)', ''))
    cc_terms = extract_go_ids_from_text(row.get('Gene Ontology (cellular component)', ''))
    
    categories['MF'] = mf_terms
    categories['BP'] = bp_terms  
    categories['CC'] = cc_terms
    
    categories['ALL'] = mf_terms | bp_terms | cc_terms
    
    all_from_combined = extract_go_ids_from_text(row.get('Gene Ontology IDs', ''))
    
    if len(categories['ALL']) > 0 and len(all_from_combined) > 0:
        overlap = len(categories['ALL'] & all_from_combined)
        if overlap / max(len(categories['ALL']), len(all_from_combined)) < 0.8:
            categories['ALL'] = all_from_combined
    elif len(all_from_combined) > len(categories['ALL']):
        categories['ALL'] = all_from_combined
    
    return categories


def build_term_index(all_terms: Set[str]) -> Dict[str, int]:
    """Build index mapping GO terms to matrix indices."""
    sorted_terms = sorted(all_terms)
    return {term: idx for idx, term in enumerate(sorted_terms)}


def count_cooccurrences(protein_annotations: List[Set[str]], 
                       term_index: Dict[str, int], 
                       logger: logging.Logger) -> csr_matrix:
    n_terms = len(term_index)
    logger.info(f"Building co-occurrence matrix for {n_terms} terms")
    
    cooccurrence_counts = defaultdict(int)
    
    for protein_terms in tqdm(protein_annotations, desc="Processing proteins"):
        term_list = list(protein_terms)
        
        for i in range(len(term_list)):
            for j in range(i + 1, len(term_list)):
                term1, term2 = term_list[i], term_list[j]
                
                if term1 in term_index and term2 in term_index:
                    idx1, idx2 = term_index[term1], term_index[term2]
                    
                    if idx1 > idx2:
                        idx1, idx2 = idx2, idx1
                    
                    cooccurrence_counts[(idx1, idx2)] += 1
    
    logger.info(f"Found {len(cooccurrence_counts)} unique term pairs with co-occurrences")
    
    rows, cols, data = [], [], []
    
    for (i, j), count in cooccurrence_counts.items():
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([count, count])
    
    matrix = csr_matrix((data, (rows, cols)), shape=(n_terms, n_terms))
    
    logger.info(f"Created sparse matrix with shape {matrix.shape} and {matrix.nnz} non-zero entries")
    
    return matrix


def save_matrix_and_metadata(matrix: csr_matrix, term_index: Dict[str, int], 
                            ontology: str, output_dir: str, logger: logging.Logger):
    """Save sparse matrix and associated metadata."""
    prefix = f"cooccurrence_{ontology.lower()}"
    
    matrix_file = os.path.join(output_dir, f"{prefix}_matrix.npz")
    save_npz(matrix_file, matrix)
    logger.info(f"Saved {ontology} co-occurrence matrix to {matrix_file}")
    
    index_file = os.path.join(output_dir, f"{prefix}_term_index.pkl")
    with open(index_file, 'wb') as f:
        pickle.dump(term_index, f)
    logger.info(f"Saved {ontology} term index to {index_file}")
    
    reverse_index = {idx: term for term, idx in term_index.items()}
    reverse_file = os.path.join(output_dir, f"{prefix}_index_to_term.pkl")
    with open(reverse_file, 'wb') as f:
        pickle.dump(reverse_index, f)
    logger.info(f"Saved {ontology} reverse index to {reverse_file}")
    
    metadata = {
        'ontology': ontology,
        'n_terms': len(term_index),
        'matrix_shape': matrix.shape,
        'n_nonzero': int(matrix.nnz),
        'density': float(matrix.nnz) / (matrix.shape[0] * matrix.shape[1]) if matrix.shape[0] > 0 else 0.0,
        'max_cooccurrence': int(matrix.max()) if matrix.nnz > 0 else 0,
        'mean_cooccurrence': float(matrix.mean()) if matrix.nnz > 0 else 0.0
    }
    
    metadata_file = os.path.join(output_dir, f"{prefix}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved {ontology} metadata to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Build GO Term Co-occurrence Matrices from SwissProt Data (FIXED)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--input', required=True,
                       help='Input SwissProt TSV file')
    parser.add_argument('--output_dir', default='matrices_fixed',
                       help='Output directory for matrices (default: matrices_fixed)')
    parser.add_argument('--ontologies', nargs='+', 
                       choices=['ALL', 'MF', 'BP', 'CC'], 
                       default=['ALL', 'MF', 'BP', 'CC'],
                       help='Ontologies to process (default: all)')
    parser.add_argument('--min_terms_per_protein', type=int, default=1,
                       help='Minimum GO terms per protein to include (default: 1)')
    parser.add_argument('--chunk_size', type=int, default=10000,
                       help='Chunk size for processing large files (default: 10000)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    logger = setup_logging(args.output_dir, args.verbose)
    
    logger.info("Starting GO term co-occurrence matrix construction (FIXED VERSION)")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Ontologies to process: {args.ontologies}")
    logger.info(f"Minimum terms per protein: {args.min_terms_per_protein}")
    
    try:
        logger.info("Loading SwissProt annotations...")
        
        chunks = []
        for chunk in tqdm(pd.read_csv(args.input, sep='\t', chunksize=args.chunk_size), 
                         desc="Loading data chunks"):
            chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(df)} protein entries")
        
        logger.info("Processing GO annotations using column-based categorization...")
        protein_annotations = {
            'ALL': [],
            'MF': [],
            'BP': [],
            'CC': []
        }
        
        all_terms = {
            'ALL': set(),
            'MF': set(),
            'BP': set(),
            'CC': set()
        }
        
        proteins_processed = 0
        proteins_with_annotations = 0
        ontology_stats = {'MF': 0, 'BP': 0, 'CC': 0}
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing proteins"):
            categorized = parse_go_terms_from_columns(row)
            
            total_terms = len(categorized['ALL'])
            if total_terms < args.min_terms_per_protein:
                continue
                
            proteins_processed += 1
            if total_terms > 0:
                proteins_with_annotations += 1
            
            for ontology in ['ALL', 'MF', 'BP', 'CC']:
                ont_terms = categorized[ontology]
                if ont_terms:
                    protein_annotations[ontology].append(ont_terms)
                    all_terms[ontology].update(ont_terms)
                    
                    if ontology in ontology_stats and ont_terms:
                        ontology_stats[ontology] += 1
        
        logger.info(f"Processed {proteins_processed} proteins with sufficient annotations")
        logger.info(f"Proteins with any GO annotations: {proteins_with_annotations}")
        
        for ontology in ['ALL', 'MF', 'BP', 'CC']:
            logger.info(f"{ontology}: {len(all_terms[ontology])} unique terms, "
                       f"{len(protein_annotations[ontology])} proteins")
        
        for ontology in ['MF', 'BP', 'CC']:
            if all_terms[ontology]:
                sample_terms = list(all_terms[ontology])[:5]
                logger.info(f"Sample {ontology} terms: {sample_terms}")
        
        for ontology in args.ontologies:
            if ontology not in all_terms:
                logger.warning(f"Unknown ontology: {ontology}")
                continue
                
            if not all_terms[ontology]:
                logger.warning(f"No terms found for ontology: {ontology}")
                continue
                
            logger.info(f"\nBuilding co-occurrence matrix for {ontology}")
            
            term_index = build_term_index(all_terms[ontology])
            
            matrix = count_cooccurrences(protein_annotations[ontology], term_index, logger)
            
            save_matrix_and_metadata(matrix, term_index, ontology, args.output_dir, logger)
        
        stats = {
            'total_proteins_in_file': len(df),
            'proteins_processed': proteins_processed,
            'proteins_with_annotations': proteins_with_annotations,
            'min_terms_per_protein': args.min_terms_per_protein,
            'ontology_stats': {
                ont: {
                    'unique_terms': len(all_terms[ont]),
                    'proteins_with_terms': len(protein_annotations[ont])
                }
                for ont in ['ALL', 'MF', 'BP', 'CC']
            },
            'categorization_method': 'column_based'
        }
        
        stats_file = os.path.join(args.output_dir, 'build_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved build statistics to {stats_file}")
        
        logger.info(f"\nCo-occurrence matrix construction completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error during matrix construction: {str(e)}")
        raise


if __name__ == "__main__":
    main() 