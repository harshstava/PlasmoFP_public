#!/usr/bin/env python3
"""
Calculate Resnik similarity matrices for GO terms using GOATools.

This script loads GO terms from a text file and calculates separate pairwise
Resnik semantic similarity matrices for each GO ontology (BP, CC, MF).
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from goatools.obo_parser import GODag
from goatools.semantic import resnik_sim, TermCounts


def load_go_terms(filename):
    """Load GO terms from a text file (one per line)."""
    go_terms = []
    with open(filename, 'r') as f:
        for line in f:
            term = line.strip()
            if term and term.startswith('GO:'):
                go_terms.append(term)
    return go_terms


def separate_terms_by_ontology(go_dag, go_terms):
    """Separate GO terms by their ontology namespace (BP, CC, MF)."""
    bp_terms = []  # Biological Process
    cc_terms = []  # Cellular Component
    mf_terms = []  # Molecular Function
    unknown_terms = []
    
    for term in go_terms:
        if term in go_dag:
            namespace = go_dag[term].namespace
            if namespace == 'biological_process':
                bp_terms.append(term)
            elif namespace == 'cellular_component':
                cc_terms.append(term)
            elif namespace == 'molecular_function':
                mf_terms.append(term)
            else:
                unknown_terms.append(term)
        else:
            unknown_terms.append(term)
    
    return {
        'BP': bp_terms,
        'CC': cc_terms,
        'MF': mf_terms,
        'unknown': unknown_terms
    }


def create_term_counts(go_dag, go_terms):
    """Create TermCounts object needed for Resnik similarity calculation."""
    # Create a simple annotation dictionary where each GO term 
    # annotates itself (required for TermCounts)
    associations = defaultdict(set)
    for term in go_terms:
        if term in go_dag:
            associations[f"protein_{term}"] = {term}
    
    # Create TermCounts object
    termcounts = TermCounts(go_dag, associations)
    return termcounts


def calculate_resnik_matrix(go_dag, go_terms, termcounts):
    """Calculate pairwise Resnik similarity matrix."""
    n_terms = len(go_terms)
    similarity_matrix = np.zeros((n_terms, n_terms))
    
    print(f"Calculating Resnik similarity for {n_terms} GO terms...")
    
    # Calculate pairwise similarities
    for i in tqdm(range(n_terms), desc="Processing terms"):
        term1 = go_terms[i]
        
        # Self-similarity is 1.0
        similarity_matrix[i, i] = 1.0
        
        for j in range(i + 1, n_terms):
            term2 = go_terms[j]
            
            # Check if both terms exist in the ontology
            if term1 in go_dag and term2 in go_dag:
                try:
                    # Calculate Resnik similarity
                    sim_score = resnik_sim(term1, term2, go_dag, termcounts)
                    
                    # Handle potential None values
                    if sim_score is None:
                        sim_score = 0.0
                    
                    similarity_matrix[i, j] = sim_score
                    similarity_matrix[j, i] = sim_score  # Matrix is symmetric
                    
                except Exception as e:
                    print(f"Warning: Could not calculate similarity between {term1} and {term2}: {e}")
                    similarity_matrix[i, j] = 0.0
                    similarity_matrix[j, i] = 0.0
            else:
                # If terms don't exist in ontology, set similarity to 0
                similarity_matrix[i, j] = 0.0
                similarity_matrix[j, i] = 0.0
    
    return similarity_matrix


def save_similarity_matrix(similarity_matrix, go_terms, output_file, ontology_name=""):
    """Save similarity matrix as a TSV file with GO terms as headers."""
    # Create DataFrame with GO terms as both row and column labels
    df = pd.DataFrame(similarity_matrix, index=go_terms, columns=go_terms)
    
    # Save as TSV
    df.to_csv(output_file, sep='\t')
    print(f"Similarity matrix saved to: {output_file}")
    
    # Print some statistics
    prefix = f"{ontology_name} " if ontology_name else ""
    print(f"\n{prefix}Matrix statistics:")
    print(f"Shape: {similarity_matrix.shape}")
    print(f"Min similarity: {np.min(similarity_matrix):.4f}")
    print(f"Max similarity: {np.max(similarity_matrix):.4f}")
    print(f"Mean similarity: {np.mean(similarity_matrix):.4f}")
    print(f"Non-zero similarities: {np.count_nonzero(similarity_matrix)} / {similarity_matrix.size}")


def process_ontology(go_dag, ontology_terms, ontology_name, output_prefix):
    """Process a single ontology and generate its similarity matrix."""
    if len(ontology_terms) == 0:
        print(f"No {ontology_name} terms found, skipping...")
        return
    
    print(f"\nProcessing {ontology_name} ontology with {len(ontology_terms)} terms...")
    
    # Create TermCounts object for this ontology
    termcounts = create_term_counts(go_dag, ontology_terms)
    
    # Calculate similarity matrix
    similarity_matrix = calculate_resnik_matrix(go_dag, ontology_terms, termcounts)
    
    # Save results
    output_file = f"{output_prefix}_{ontology_name}_similarity_matrix_NEW_2.tsv"
    save_similarity_matrix(similarity_matrix, ontology_terms, output_file, ontology_name)


def main():
    # File paths
    go_obo_file = "go-basic.obo"
    go_terms_file = "combined_existing_pfp_NEW_2.txt"
    output_prefix = "resnik"
    
    # Check if input files exist
    if not os.path.exists(go_obo_file):
        print(f"Error: GO ontology file not found: {go_obo_file}")
        sys.exit(1)
    
    if not os.path.exists(go_terms_file):
        print(f"Error: GO terms file not found: {go_terms_file}")
        sys.exit(1)
    
    print("Loading GO ontology...")
    # Load GO ontology
    go_dag = GODag(go_obo_file, optional_attrs=['relationship'])
    print(f"Loaded {len(go_dag)} GO terms from ontology")
    
    print("Loading GO terms from file...")
    # Load GO terms
    go_terms = load_go_terms(go_terms_file)
    print(f"Loaded {len(go_terms)} GO terms from file")
    
    print("Separating terms by ontology...")
    # Separate terms by ontology
    ontology_terms = separate_terms_by_ontology(go_dag, go_terms)
    
    # Print summary
    print(f"\nOntology breakdown:")
    print(f"Biological Process (BP): {len(ontology_terms['BP'])} terms")
    print(f"Cellular Component (CC): {len(ontology_terms['CC'])} terms")
    print(f"Molecular Function (MF): {len(ontology_terms['MF'])} terms")
    print(f"Unknown/Invalid: {len(ontology_terms['unknown'])} terms")
    
    if ontology_terms['unknown']:
        print(f"First few unknown terms: {ontology_terms['unknown'][:5]}")
    
    # Process each ontology separately
    for ontology_name in ['BP', 'CC', 'MF']:
        if len(ontology_terms[ontology_name]) > 0:
            process_ontology(go_dag, ontology_terms[ontology_name], ontology_name, output_prefix)
        else:
            print(f"\nNo {ontology_name} terms found, skipping...")
    
    print("\nAll similarity matrices have been generated!")
    print("Generated files:")
    for ontology_name in ['BP', 'CC', 'MF']:
        if len(ontology_terms[ontology_name]) > 0:
            filename = f"{output_prefix}_{ontology_name}_similarity_matrix.tsv"
            print(f"  - {filename}")
    
    print("Done!")


if __name__ == "__main__":
    main()
