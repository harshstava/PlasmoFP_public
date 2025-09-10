#!/usr/bin/env python3
"""
Mask-and-Compare Imputation Analysis with Random Control
+ Uncertainty (bootstrap CIs, paired permutation p-values)

Tests PlasmoFP's imputation capability by:
1. Masking existing annotations from target ontology
2. Comparing predicted vs real annotations against cross-ontology existing
3. Including random control with shuffled predictions
4. Reporting imputation efficiency = predicted_score / ground_truth_score
5. 95% bootstrap CIs for scores/efficiencies + paired permutation p-values
6. Per-term quality CSV for downstream IC/depth analysis
"""

import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
import logging
from scipy.sparse import csr_matrix
import random
import json

from build_cooccurrence_matrices_fixed import extract_go_ids_from_text

def filter_to_deepest_terms(go_terms: List[str], ancestors_dict: Dict[str, List[str]]) -> List[str]:
    """Filter GO terms to keep only the deepest (most specific) terms."""
    if not go_terms:
        return []
    
    deepest_terms = []
    for term in go_terms:
        is_deepest = True
        term_ancestors = set(ancestors_dict.get(term, []))
        for other_term in go_terms:
            if other_term != term:
                other_ancestors = set(ancestors_dict.get(other_term, []))
                if term in other_ancestors:
                    is_deepest = False
                    break
        if is_deepest:
            deepest_terms.append(term)
    return deepest_terms

def load_ancestors_dictionaries(ancestors_dir: str, logger: logging.Logger) -> Dict[str, Dict[str, List[str]]]:
    """Load ancestor dictionaries for all ontologies."""
    ancestors = {}
    ontology_files = {
        'MF': 'trained_function_ALL_data_ancestors_dict.json',
        'BP': 'trained_process_ALL_data_ancestors_dict.json', 
        'CC': 'trained_component_ALL_data_ancestors_dict.json'
    }
    for ontology, filename in ontology_files.items():
        filepath = os.path.join(ancestors_dir, filename)
        if os.path.exists(filepath):
            logger.info(f"Loading {ontology} ancestors from {filepath}")
            with open(filepath, 'r') as f:
                ancestors[ontology] = json.load(f)
            logger.info(f"Loaded {len(ancestors[ontology])} {ontology} ancestor mappings")
        else:
            logger.warning(f"Ancestor file not found: {filepath}")
            ancestors[ontology] = {}
    return ancestors

def setup_logging(log_file: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class PropagatedCooccurrenceMatrix:
    def __init__(self, matrices_dir: str, ontology: str, logger: logging.Logger):
        self.logger = logger
        self.ontology = ontology
        matrix_file = os.path.join(matrices_dir, f'cooccurrence_{ontology.lower()}_propagated_matrix.npz')
        index_file = os.path.join(matrices_dir, f'cooccurrence_{ontology.lower()}_propagated_index_to_term.pkl')
        if not os.path.exists(matrix_file):
            matrix_file = os.path.join(matrices_dir, f'cooccurrence_{ontology.lower()}_matrix.npz')
            index_file = os.path.join(matrices_dir, f'cooccurrence_{ontology.lower()}_index_to_term.pkl')
        self.logger.info(f"Loading {ontology} propagated co-occurrence matrix from {matrix_file}")
        npz_data = np.load(matrix_file)
        if 'data' in npz_data.files:
            self.matrix = csr_matrix((npz_data['data'], npz_data['indices'], npz_data['indptr']), 
                                     shape=npz_data['shape'])
        else:
            self.matrix = csr_matrix(npz_data['matrix'])
        with open(index_file, 'rb') as f:
            self.index_to_term = pickle.load(f)
        self.term_to_index = {term: idx for idx, term in self.index_to_term.items()}
        self.logger.info(f"Loaded {ontology} matrix: {self.matrix.shape[0]} terms, {self.matrix.nnz} non-zero entries")
    
    def get_cooccurrence(self, term1: str, term2: str) -> int:
        """Get co-occurrence count between two terms."""
        if term1 not in self.term_to_index or term2 not in self.term_to_index:
            return 0
        idx1 = self.term_to_index[term1]
        idx2 = self.term_to_index[term2]
        return self.matrix[idx1, idx2]

def load_test_annotations(test_files: Dict[str, str], logger: logging.Logger) -> Dict[str, Dict[str, List[str]]]:
    annotations = {}
    ontology_columns = {
        'BP': 'Gene Ontology (biological process)',
        'MF': 'Gene Ontology (molecular function)',
        'CC': 'Gene Ontology (cellular component)'
    }
    for ontology, filepath in test_files.items():
        logger.info(f"Loading {ontology} annotations from {filepath}")
        df = pd.read_csv(filepath, sep='\t')
        ontology_annotations = {}
        for _, row in df.iterrows():
            protein_id = row['Entry']
            go_text = row[ontology_columns[ontology]]
            if pd.notna(go_text):
                go_terms = extract_go_ids_from_text(str(go_text))
                if go_terms:
                    ontology_annotations[protein_id] = go_terms
        annotations[ontology] = ontology_annotations
        logger.info(f"Loaded {len(ontology_annotations)} {ontology} proteins")
    return annotations

def load_predictions(pred_files: Dict[str, str], test_files: Dict[str, str], 
                    fdr_threshold: float, ancestors_dicts: Dict[str, Dict[str, List[str]]], 
                    apply_deepest_filtering: bool, logger: logging.Logger) -> Dict[str, Dict[str, List[str]]]:
    predictions = {}
    for ontology in ['BP', 'MF', 'CC']:
        pred_filepath = pred_files[ontology]
        logger.info(f"Loading {ontology} predictions from {pred_filepath}")
        with open(pred_filepath, 'rb') as f:
            pred_data = pickle.load(f)
        test_filepath = test_files[ontology]
        df = pd.read_csv(test_filepath, sep='\t')
        protein_list = df['Entry'].tolist()
        ontology_predictions = {}
        if fdr_threshold in pred_data:
            pred_list = pred_data[fdr_threshold]
            if len(pred_list) == len(protein_list):
                for i, protein_id in enumerate(protein_list):
                    if i < len(pred_list) and pred_list[i]:
                        terms = list(pred_list[i])
                        if apply_deepest_filtering and ontology in ancestors_dicts:
                            terms = filter_to_deepest_terms(terms, ancestors_dicts[ontology])
                        if terms:
                            ontology_predictions[protein_id] = terms
            else:
                logger.warning(f"Mismatch in {ontology}: {len(pred_list)} predictions vs {len(protein_list)} proteins")
        predictions[ontology] = ontology_predictions
        logger.info(f"Loaded {len(ontology_predictions)} {ontology} predictions")
    return predictions

def create_imputation_test_cases(target_ontology: str, 
                                annotations: Dict[str, Dict[str, List[str]]], 
                                predictions: Dict[str, Dict[str, List[str]]],
                                logger: logging.Logger) -> List[Dict]:
    other_ontologies = [ont for ont in ['BP', 'MF', 'CC'] if ont != target_ontology]
    logger.info(f"Creating {target_ontology} imputation test cases...")
    logger.info(f"Target: {target_ontology} predictions vs {'+'.join(other_ontologies)} existing")
    test_cases = []
    target_proteins = set(annotations[target_ontology].keys()) & set(predictions[target_ontology].keys())
    for protein_id in target_proteins:
        cross_ontology_terms = []
        cross_ontology_sources = []
        for other_ont in other_ontologies:
            if protein_id in annotations[other_ont]:
                cross_ontology_terms.extend(annotations[other_ont][protein_id])
                cross_ontology_sources.append(other_ont)
        if cross_ontology_terms:
            test_cases.append({
                'protein_id': protein_id,
                'target_existing': annotations[target_ontology][protein_id],
                'target_predictions': predictions[target_ontology][protein_id],
                'cross_ontology_existing': cross_ontology_terms,
                'cross_ontology_sources': cross_ontology_sources
            })
    logger.info(f"Created {len(test_cases)} {target_ontology} imputation test cases")
    context_counts = {}
    for case in test_cases:
        context = '+'.join(sorted(case['cross_ontology_sources']))
        context_counts[context] = context_counts.get(context, 0) + 1
    logger.info(f"{target_ontology} test case contexts:")
    for context, count in sorted(context_counts.items()):
        logger.info(f"  {context}: {count} proteins")
    return test_cases

def shuffle_predictions_across_proteins(test_cases: List[Dict], logger: logging.Logger) -> List[Dict]:
    logger.info("Creating cross-protein shuffle control by shuffling predictions across proteins...")
    all_predictions = [case['target_predictions'] for case in test_cases]
    shuffled_predictions = all_predictions.copy()
    random.shuffle(shuffled_predictions)
    shuffled_cases = []
    for i, case in enumerate(test_cases):
        shuffled_case = case.copy()
        shuffled_case['target_predictions'] = shuffled_predictions[i]
        shuffled_cases.append(shuffled_case)
    logger.info(f"Created {len(shuffled_cases)} cross-protein shuffle control test cases")
    return shuffled_cases

def create_random_term_control(test_cases: List[Dict], logger: logging.Logger) -> List[Dict]:
    logger.info("Creating pure random control by randomly selecting terms from prediction vocabulary...")
    all_predicted_terms = set()
    prediction_lengths = []
    for case in test_cases:
        terms = case['target_predictions']
        all_predicted_terms.update(terms)
        prediction_lengths.append(len(terms))
    all_predicted_terms = list(all_predicted_terms)
    logger.info(f"Vocabulary size: {len(all_predicted_terms)} unique terms")
    logger.info(f"Average predictions per protein: {np.mean(prediction_lengths):.1f}")
    random_cases = []
    for case in test_cases:
        random_case = case.copy()
        n_terms = len(case['target_predictions'])
        if n_terms > 0 and len(all_predicted_terms) > 0:
            random_terms = random.sample(all_predicted_terms, min(n_terms, len(all_predicted_terms)))
            random_case['target_predictions'] = random_terms
        else:
            random_case['target_predictions'] = []
        random_cases.append(random_case)
    logger.info(f"Created {len(random_cases)} pure random control test cases")
    return random_cases

def calculate_pairwise_quality(test_cases: List[Dict], 
                              cooccurrence_matrix: PropagatedCooccurrenceMatrix,
                              min_cooccurrence: int,
                              use_predictions: bool,
                              logger: logging.Logger) -> Dict:
    case_type = "predictions" if use_predictions else "existing"
    logger.info(f"Calculating pairwise quality using {case_type}...")
    total_pairs = 0
    good_pairs = 0
    defined_pairs = 0
    for case in test_cases:
        if use_predictions:
            target_terms = case['target_predictions']
        else:
            target_terms = case['target_existing']
        cross_ontology_terms = case['cross_ontology_existing']
        for target_term in target_terms:
            for cross_term in cross_ontology_terms:
                total_pairs += 1
                cooccurrence = cooccurrence_matrix.get_cooccurrence(target_term, cross_term)
                if cooccurrence >= min_cooccurrence:
                    defined_pairs += 1
                    good_pairs += 1
                elif cooccurrence > 0:
                    defined_pairs += 1
    score = (good_pairs / defined_pairs * 100) if defined_pairs > 0 else 0
    results = {
        'score': score,
        'total_pairs': total_pairs,
        'good_pairs': good_pairs,
        'defined_pairs': defined_pairs,
        'proteins': len(test_cases)
    }
    logger.info(f"{case_type.capitalize()} score: {score:.2f}% ({good_pairs}/{defined_pairs} pairs, {len(test_cases)} proteins)")
    return results

def per_protein_quality_breakdown(test_cases, cooccurrence_matrix, min_cooccurrence, use_predictions, logger):
    """Return a per-protein dataframe with columns: protein_id, good, defined, total_pairs."""
    rows = []
    for case in test_cases:
        target_terms = case['target_predictions'] if use_predictions else case['target_existing']
        cross_terms = case['cross_ontology_existing']
        good = 0
        defined = 0
        total_pairs = 0
        for t in target_terms:
            for c in cross_terms:
                total_pairs += 1
                co = cooccurrence_matrix.get_cooccurrence(t, c)
                if co >= min_cooccurrence:
                    defined += 1
                    good += 1
                elif co > 0:
                    defined += 1
        rows.append({
            "protein_id": case['protein_id'],
            "good": good,
            "defined": defined,
            "total_pairs": total_pairs
        })
    return pd.DataFrame(rows)

def _score_from_df(df: pd.DataFrame) -> float:
    g = df['good'].sum()
    d = df['defined'].sum()
    return (g / d * 100.0) if d > 0 else 0.0

def bootstrap_ci(df: pd.DataFrame, B: int = 5000, seed: int = 42):
    """Nonparametric bootstrap CI for the aggregated score (ratio of sums)."""
    rng = np.random.default_rng(seed)
    n = len(df)
    if n == 0:
        return np.nan, np.nan, np.array([])
    idx = np.arange(n)
    stats = []
    for _ in range(B):
        samp = df.iloc[rng.integers(0, n, n)]
        stats.append(_score_from_df(samp))
    stats = np.array(stats)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return lo, hi, stats

def paired_bootstrap_ratio_ci(df_num: pd.DataFrame, df_den: pd.DataFrame, B: int = 5000, seed: int = 42):
    """CI for efficiency = score(num) / score(den) * 100 using paired resampling by protein."""
    rng = np.random.default_rng(seed)
    if len(df_num) == 0 or len(df_den) == 0:
        return np.nan, np.nan, np.array([])
    # align by protein_id
    assert set(df_num['protein_id']) == set(df_den['protein_id'])
    df_num = df_num.set_index('protein_id').sort_index()
    df_den = df_den.set_index('protein_id').sort_index()
    n = len(df_num)
    stats = []
    keys = df_den.index.to_list()
    for _ in range(B):
        samp_idx = rng.integers(0, n, n)
        samp_keys = [keys[i] for i in samp_idx]
        m = _score_from_df(df_num.loc[samp_keys].reset_index())
        g = _score_from_df(df_den.loc[samp_keys].reset_index())
        if g > 0:
            stats.append((m / g) * 100.0)
    stats = np.array(stats)
    if stats.size == 0:
        return np.nan, np.nan, np.array([])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return lo, hi, stats

def paired_permutation_pvalue(df_a: pd.DataFrame, df_b: pd.DataFrame, n_perm: int = 10000, seed: int = 42):
    """Two-sided paired permutation p-value for difference in aggregated scores A - B."""
    rng = np.random.default_rng(seed)
    if len(df_a) == 0 or len(df_b) == 0:
        return np.nan, np.nan
    assert set(df_a['protein_id']) == set(df_b['protein_id'])
    dfa = df_a.set_index('protein_id').sort_index()[['good','defined']].values
    dfb = df_b.set_index('protein_id').sort_index()[['good','defined']].values
    # observed
    obs = _score_from_df(pd.DataFrame(dfa, columns=['good','defined'])) - \
          _score_from_df(pd.DataFrame(dfb, columns=['good','defined']))
    N = dfa.shape[0]
    count = 0
    for _ in range(n_perm):
        flips = rng.integers(0, 2, N)  # 0 keep, 1 swap
        permA = np.where(flips[:,None]==0, dfa, dfb)
        permB = np.where(flips[:,None]==0, dfb, dfa)
        diff = _score_from_df(pd.DataFrame(permA, columns=['good','defined'])) - \
               _score_from_df(pd.DataFrame(permB, columns=['good','defined']))
        if abs(diff) >= abs(obs) - 1e-12:
            count += 1
    p = (count + 1) / (n_perm + 1)  # add-one smoothing
    return obs, p

def per_term_quality_summary(test_cases, cooccurrence_matrix, min_cooccurrence, logger):
    """Return a dataframe per predicted term: n_predictions, defined_pairs, good_pairs, pairwise_quality_pct."""
    from collections import defaultdict
    term_good = defaultdict(int)
    term_defined = defaultdict(int)
    term_count = defaultdict(int)
    for case in test_cases:
        preds = case['target_predictions']
        cross = case['cross_ontology_existing']
        for t in preds:
            term_count[t] += 1
            for c in cross:
                co = cooccurrence_matrix.get_cooccurrence(t, c)
                if co >= min_cooccurrence:
                    term_defined[t] += 1
                    term_good[t] += 1
                elif co > 0:
                    term_defined[t] += 1
    rows = []
    for t in term_count:
        d = term_defined[t]
        g = term_good[t]
        rows.append({
            "term": t,
            "n_predictions": term_count[t],
            "defined_pairs": d,
            "good_pairs": g,
            "pairwise_quality_pct": (g / d * 100.0) if d > 0 else np.nan
        })
    df = pd.DataFrame(rows).sort_values("pairwise_quality_pct", na_position='last')
    logger.info(f"Per-term summary: {len(df)} terms")
    return df

def evaluate_ontology_imputation(target_ontology: str,
                                annotations: Dict[str, Dict[str, List[str]]], 
                                predictions: Dict[str, Dict[str, List[str]]],
                                cooccurrence_matrix: PropagatedCooccurrenceMatrix,
                                min_cooccurrence: int,
                                output_dir: str,
                                logger: logging.Logger) -> Dict:
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATING {target_ontology} IMPUTATION")
    logger.info(f"{'='*60}")
    test_cases = create_imputation_test_cases(target_ontology, annotations, predictions, logger)
    if not test_cases:
        logger.warning(f"No test cases for {target_ontology} imputation")
        return {}
    shuffled_cases = shuffle_predictions_across_proteins(test_cases, logger)
    random_cases = create_random_term_control(test_cases, logger)

    logger.info(f"\nCalculating {target_ontology} imputation metrics...")
    prediction_results = calculate_pairwise_quality(test_cases, cooccurrence_matrix, min_cooccurrence, True, logger)
    existing_results   = calculate_pairwise_quality(test_cases, cooccurrence_matrix, min_cooccurrence, False, logger)
    shuffle_results    = calculate_pairwise_quality(shuffled_cases, cooccurrence_matrix, min_cooccurrence, True, logger)
    random_results     = calculate_pairwise_quality(random_cases, cooccurrence_matrix, min_cooccurrence, True, logger)

    imputation_efficiency = (prediction_results['score'] / existing_results['score'] * 100) if existing_results['score'] > 0 else 0
    shuffle_efficiency    = (shuffle_results['score']    / existing_results['score'] * 100) if existing_results['score'] > 0 else 0
    random_efficiency     = (random_results['score']     / existing_results['score'] * 100) if existing_results['score'] > 0 else 0

    pred_df = per_protein_quality_breakdown(test_cases,      cooccurrence_matrix, min_cooccurrence, True,  logger)
    gt_df   = per_protein_quality_breakdown(test_cases,      cooccurrence_matrix, min_cooccurrence, False, logger)
    shuf_df = per_protein_quality_breakdown(shuffled_cases,  cooccurrence_matrix, min_cooccurrence, True,  logger)
    rand_df = per_protein_quality_breakdown(random_cases,    cooccurrence_matrix, min_cooccurrence, True,  logger)

    pred_lo, pred_hi, _ = bootstrap_ci(pred_df)
    gt_lo,   gt_hi,   _ = bootstrap_ci(gt_df)
    shuf_lo, shuf_hi, _ = bootstrap_ci(shuf_df)
    rand_lo, rand_hi, _ = bootstrap_ci(rand_df)

    eff_lo, eff_hi, _       = paired_bootstrap_ratio_ci(pred_df, gt_df)
    eff_sh_lo, eff_sh_hi, _ = paired_bootstrap_ratio_ci(shuf_df, gt_df)
    eff_rd_lo, eff_rd_hi, _ = paired_bootstrap_ratio_ci(rand_df, gt_df)

    diff_m_sh, p_m_sh = paired_permutation_pvalue(pred_df, shuf_df)
    diff_m_rd, p_m_rd = paired_permutation_pvalue(pred_df, rand_df)

    term_df = per_term_quality_summary(test_cases, cooccurrence_matrix, min_cooccurrence, logger)
    term_csv = os.path.join(output_dir, f"{target_ontology}_per_term_quality.csv")
    term_df.to_csv(term_csv, index=False)
    logger.info(f"Saved per-term quality to {term_csv}")

    results = {
        'target_ontology': target_ontology,
        'test_cases': len(test_cases),
        'prediction_score': prediction_results['score'],
        'existing_score': existing_results['score'],
        'shuffle_score': shuffle_results['score'],
        'random_score': random_results['score'],
        'imputation_efficiency': imputation_efficiency,
        'shuffle_efficiency': shuffle_efficiency,
        'random_efficiency': random_efficiency,
        'prediction_score_CI': (pred_lo, pred_hi),
        'existing_score_CI': (gt_lo, gt_hi),
        'shuffle_score_CI': (shuf_lo, shuf_hi),
        'random_score_CI': (rand_lo, rand_hi),
        'imputation_efficiency_CI': (eff_lo, eff_hi),
        'shuffle_efficiency_CI': (eff_sh_lo, eff_sh_hi),
        'random_efficiency_CI': (eff_rd_lo, eff_rd_hi),
        'model_vs_shuffle_diff_pct': diff_m_sh,
        'model_vs_shuffle_pval': p_m_sh,
        'model_vs_random_diff_pct': diff_m_rd,
        'model_vs_random_pval': p_m_rd,
        'prediction_results': prediction_results,
        'existing_results': existing_results,
        'shuffle_results': shuffle_results,
        'random_results': random_results
    }

    logger.info(f"\n{target_ontology} IMPUTATION SUMMARY:")
    logger.info(f"  Prediction score:      {prediction_results['score']:.2f}%  [95% CI {pred_lo:.2f}, {pred_hi:.2f}]")
    logger.info(f"  Ground truth score:    {existing_results['score']:.2f}%  [95% CI {gt_lo:.2f}, {gt_hi:.2f}]")
    logger.info(f"  Shuffle control score: {shuffle_results['score']:.2f}%  [95% CI {shuf_lo:.2f}, {shuf_hi:.2f}]")
    logger.info(f"  Random control score:  {random_results['score']:.2f}%  [95% CI {rand_lo:.2f}, {rand_hi:.2f}]")
    logger.info(f"  Imputation efficiency: {imputation_efficiency:.1f}%        [95% CI {eff_lo:.1f}, {eff_hi:.1f}]")
    logger.info(f"  Shuffle efficiency:    {shuffle_efficiency:.1f}%          [95% CI {eff_sh_lo:.1f}, {eff_sh_hi:.1f}]")
    logger.info(f"  Random efficiency:     {random_efficiency:.1f}%           [95% CI {eff_rd_lo:.1f}, {eff_rd_hi:.1f}]")
    logger.info(f"  Model vs Shuffle:      {diff_m_sh:+.2f} percentage points (paired permutation p={p_m_sh:.4g})")
    logger.info(f"  Model vs Random:       {diff_m_rd:+.2f} percentage points (paired permutation p={p_m_rd:.4g})")

    return results

def main():
    parser = argparse.ArgumentParser(description="Mask-and-Compare Imputation Analysis")
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Directory containing test TSV files')
    parser.add_argument('--predictions_dir', type=str, required=True,
                       help='Directory containing prediction files')
    parser.add_argument('--matrices_dir', type=str, required=True,
                       help='Directory containing propagated co-occurrence matrices')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--min_cooccurrence', type=int, default=3,
                       help='Minimum co-occurrence threshold for "good" pairs')
    parser.add_argument('--fdr_threshold', type=float, default=0.10,
                       help='FDR threshold for predictions (default: 0.10)')
    parser.add_argument('--ontologies', nargs='+', default=['MF', 'BP', 'CC'],
                       help='Ontologies to evaluate (default: MF BP CC)')
    parser.add_argument('--ancestors_dir', type=str, required=True,
                       help='Directory containing ancestor dictionaries for deepest filtering')
    parser.add_argument('--apply_deepest_filtering', action='store_true',
                       help='Apply deepest term filtering to predictions (recommended for fair comparison)')
    args = parser.parse_args()

    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'mask_and_compare_imputation_log.txt')
    logger = setup_logging(log_file)

    logger.info("Starting mask-and-compare imputation analysis")
    logger.info(f"Test directory: {args.test_dir}")
    logger.info(f"Predictions directory: {args.predictions_dir}")
    logger.info(f"Matrices directory: {args.matrices_dir}")
    logger.info(f"Minimum co-occurrence: {args.min_cooccurrence}")
    logger.info(f"FDR threshold: {args.fdr_threshold}")
    logger.info(f"Target ontologies: {args.ontologies}")
    logger.info(f"Apply deepest filtering: {args.apply_deepest_filtering}")
    if args.apply_deepest_filtering:
        logger.info("Apply deepest term filtering to predictions for fair comparison")
    logger.info("Report 95% CIs (bootstrap) and p-values (paired permutation)")
    try:
        test_files = {
            'BP': os.path.join(args.test_dir, 'process_test.tsv'),
            'MF': os.path.join(args.test_dir, 'function_test.tsv'),
            'CC': os.path.join(args.test_dir, 'component_test.tsv')
        }
        from prediction_file_utils import get_ontology_prediction_files
        try:
            pred_files = get_ontology_prediction_files(args.predictions_dir, prefer_fdr=True, logger=logger)
        except FileNotFoundError:
            logger.warning("Using fallback prediction file paths")
            pred_files = {
                'BP': os.path.join(args.predictions_dir, 'process_test_predictions_and_FDR_NEW/bpo/predicted_terms.pkl'),
                'MF': os.path.join(args.predictions_dir, 'function_test_predictions_and_FDR_NEW/mfo/predicted_terms.pkl'),
                'CC': os.path.join(args.predictions_dir, 'component_test_predictions_and_FDR_NEW/cco/predicted_terms.pkl')
            }

        logger.info("Loading co-occurrence matrix...")
        cooccurrence_matrix = PropagatedCooccurrenceMatrix(args.matrices_dir, 'ALL', logger)

        ancestors_dicts = {}
        if args.apply_deepest_filtering:
            logger.info("Loading ancestor dictionaries for deepest filtering...")
            ancestors_dicts = load_ancestors_dictionaries(args.ancestors_dir, logger)

        logger.info("Loading test annotations...")
        annotations = load_test_annotations(test_files, logger)

        logger.info("Loading predictions...")
        predictions = load_predictions(pred_files, test_files, args.fdr_threshold, 
                                       ancestors_dicts, args.apply_deepest_filtering, logger)

        all_results = {}
        for ontology in args.ontologies:
            results = evaluate_ontology_imputation(
                ontology, annotations, predictions, cooccurrence_matrix,
                args.min_cooccurrence, args.output_dir, logger
            )
            if results:
                all_results[ontology] = results
    
        results_file = os.path.join(args.output_dir, 'mask_and_compare_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(all_results, f)

        summary_data = []
        for ontology, r in all_results.items():
            summary_data.append({
                'ontology': ontology,
                'test_cases': r['test_cases'],
                'prediction_score': r['prediction_score'],
                'prediction_score_CI_low': r['prediction_score_CI'][0],
                'prediction_score_CI_high': r['prediction_score_CI'][1],
                'ground_truth_score': r['existing_score'],
                'ground_truth_score_CI_low': r['existing_score_CI'][0],
                'ground_truth_score_CI_high': r['existing_score_CI'][1],
                'shuffle_score': r['shuffle_score'],
                'shuffle_score_CI_low': r['shuffle_score_CI'][0],
                'shuffle_score_CI_high': r['shuffle_score_CI'][1],
                'random_score': r['random_score'],
                'random_score_CI_low': r['random_score_CI'][0],
                'random_score_CI_high': r['random_score_CI'][1],
                'imputation_efficiency_pct': r['imputation_efficiency'],
                'imputation_efficiency_CI_low': r['imputation_efficiency_CI'][0],
                'imputation_efficiency_CI_high': r['imputation_efficiency_CI'][1],
                'shuffle_efficiency_pct': r['shuffle_efficiency'],
                'shuffle_efficiency_CI_low': r['shuffle_efficiency_CI'][0],
                'shuffle_efficiency_CI_high': r['shuffle_efficiency_CI'][1],
                'random_efficiency_pct': r['random_efficiency'],
                'random_efficiency_CI_low': r['random_efficiency_CI'][0],
                'random_efficiency_CI_high': r['random_efficiency_CI'][1],
                'model_vs_shuffle_diff_pct': r['model_vs_shuffle_diff_pct'],
                'model_vs_shuffle_pval': r['model_vs_shuffle_pval'],
                'model_vs_random_diff_pct': r['model_vs_random_diff_pct'],
                'model_vs_random_pval': r['model_vs_random_pval'],
            })
        summary_file = os.path.join(args.output_dir, 'imputation_summary.csv')
        pd.DataFrame(summary_data).to_csv(summary_file, index=False)

        logger.info("="*80)
        logger.info(f"Saved: {results_file}")
        logger.info(f"Saved: {summary_file}")
        logger.info("Per-term files saved per ontology: *_per_term_quality.csv")

    except Exception as e:
        logger.exception(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
