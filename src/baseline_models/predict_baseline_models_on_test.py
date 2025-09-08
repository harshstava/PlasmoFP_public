#!/usr/bin/env python3
"""
Unified Model Inference Script v2 - Ontology-Specific Inputs

This script provides unified inference capabilities for models trained with:
- train_sar_cnn.py (SAR CNN models) 
- train_cnn_cafa.py (CAFA CNN models)
- train_cafa_models.py (CAFA TM-Vec models)

Supports ontology-specific input files for efficient batch processing.

USAGE EXAMPLES:

# Test CNN models with ontology-specific FASTA files
python predict_with_models_v2.py --model_dir sar_cnn_models --model_type sar_cnn \
    --ontology ALL --output_dir results \
    --bpo_fasta bpo_sequences.fasta --bpo_tsv bpo_annotations.tsv \
    --cco_fasta cco_sequences.fasta --cco_tsv cco_annotations.tsv \
    --mfo_fasta mfo_sequences.fasta --mfo_tsv mfo_annotations.tsv

# Test TM-Vec models with ontology-specific embeddings
python predict_with_models_v2.py --model_dir cafa_tmvec_models --model_type cafa_tmvec \
    --ontology ALL --output_dir results \
    --bpo_embeddings bpo_embeddings.npy --bpo_tsv bpo_annotations.tsv \
    --cco_embeddings cco_embeddings.npy --cco_tsv cco_annotations.tsv \
    --mfo_embeddings mfo_embeddings.npy --mfo_tsv mfo_annotations.tsv

# Single ontology (backward compatibility)
python predict_with_models_v2.py --model_dir sar_cnn_models --model_type sar_cnn \
    --ontology BPO --fasta sequences.fasta --bp_tsv annotations.tsv --output_dir results
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from Bio import SeqIO
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from utils_corrected import (
    PFP, process_GO_data, predict_without_MCD,
    threshold_performance_metrics, calculate_aupr_micro
)

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_INDEX['X'] = len(AMINO_ACIDS)

MODEL_TYPES = {
    'sar_cnn': {
        'architecture': 'cnn',
        'input_type': 'sequences',
        'model_file_pattern': '{ontology}_cnn_best_model.pt',
        'params_file_pattern': '{ontology}_cnn_best_params.pkl',
        'mlb_file_pattern': '{ontology_type}_mlb.pkl'
    },
    'cafa_cnn': {
        'architecture': 'cnn', 
        'input_type': 'sequences',
        'model_file_pattern': '{ontology}_cafa_cnn_best_model.pt',
        'params_file_pattern': '{ontology}_cafa_cnn_best_params.pkl',
        'mlb_file_pattern': '{ontology_type}_mlb.pkl'
    },
    'cafa_tmvec': {
        'architecture': 'ffnn',
        'input_type': 'embeddings',
        'model_file_pattern': '{ontology}_best_model.pt',
        'params_file_pattern': '{ontology}_best_params.json',
        'mlb_file_pattern': '{ontology}_mlb.pkl'
    }
}

ONTOLOGY_MAPPING = {
    'BPO': 'process',
    'CCO': 'component', 
    'MFO': 'function'
}


class SimpleCNN(nn.Module):
    """CNN architecture - reused from training scripts."""
    
    def __init__(self, num_classes: int, max_length: int, conv1_out: int, 
                 conv2_out: int, fc_size: int, dropout_rate: float = 0.5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=21, out_channels=conv1_out, kernel_size=8, padding=4)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=conv1_out, out_channels=conv2_out, kernel_size=8, padding=4)
        self.fc1 = nn.Linear(conv2_out * (max_length // 4), fc_size)
        self.fc2 = nn.Linear(fc_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    """Setup logging - reused from predict_with_uncertainty.py."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logger = logging.getLogger('model_prediction')
    logger.setLevel(log_level)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    log_file = output_dir / 'prediction_log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_ic_dict(ic_path: Optional[Path], logger: logging.Logger) -> Optional[Dict[str, float]]:
    """Load Information Content dictionary - reused from predict_with_uncertainty.py."""
    if ic_path is None:
        ic_path = Path('IA_all.tsv')
    
    if not ic_path.exists():
        logger.warning(f"IC file not found: {ic_path}. CAFA metrics will be limited.")
        return None
    
    try:
        ic_df = pd.read_csv(ic_path, sep='\t', header=None, names=['GO_term', 'IC'])
        ic_dict = dict(zip(ic_df['GO_term'], ic_df['IC']))
        logger.info(f"Loaded IC dictionary with {len(ic_dict)} terms from {ic_path}")
        return ic_dict
    except Exception as e:
        logger.warning(f"Failed to load IC dictionary: {e}")
        return None


def one_hot_encode_sequence(sequence: str, max_length: int) -> np.ndarray:
    """One-hot encode protein sequence - reused from CNN training scripts."""
    encoding = np.zeros((21, max_length), dtype=np.float16)
    
    for i, aa in enumerate(sequence):
        if i >= max_length:
            break
        index = AA_TO_INDEX.get(aa, AA_TO_INDEX['X'])
        encoding[index, i] = 1.0
    
    return encoding


def process_sequences_batch(sequences: List[str], max_length: int, batch_size: int = 1000) -> np.ndarray:
    """Process sequences in batches - reused from CNN training scripts."""
    encoded_sequences = []
    
    for i in tqdm(range(0, len(sequences), batch_size), desc="Encoding sequences"):
        batch_end = min(i + batch_size, len(sequences))
        batch_sequences = sequences[i:batch_end]
        
        batch_encoded = []
        for sequence in batch_sequences:
            batch_encoded.append(one_hot_encode_sequence(sequence, max_length))
        
        encoded_sequences.extend(batch_encoded)
    
    return np.array(encoded_sequences, dtype=np.float16)


def load_fasta_sequences(fasta_path: Path) -> List[str]:
    """Load sequences from FASTA file."""
    sequences = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        sequences.append(str(record.seq))
    return sequences


def generate_embeddings_from_fasta(fasta_file: Path, output_dir: Path, logger: logging.Logger) -> Path:
    """Generate embeddings using generate_embeddings.py - reused from predict_with_uncertainty.py."""
    logger.info(f"Generating embeddings from {fasta_file}")
    
    if not Path('generate_embeddings.py').exists():
        raise FileNotFoundError("generate_embeddings.py not found in current directory")
    
    embeddings_file = output_dir / f'generated_embeddings_{fasta_file.stem}.npy'
    
    import subprocess
    cmd = [
        'python', 'generate_embeddings.py', 
        '--fasta', str(fasta_file),
        '--output', str(embeddings_file)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Embedding generation completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Embedding generation failed: {e.stderr}")
        raise
    
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Expected embeddings file not created: {embeddings_file}")
    
    return embeddings_file


def load_model_and_params(model_dir: Path, ontology: str, model_type: str, 
                         device: torch.device, logger: logging.Logger) -> Tuple[nn.Module, MultiLabelBinarizer]:
    """Load trained model and associated MultiLabelBinarizer."""
    config = MODEL_TYPES[model_type]
    ontology_type = ONTOLOGY_MAPPING[ontology]
    
    model_file = model_dir / config['model_file_pattern'].format(ontology=ontology)
    params_file = model_dir / config['params_file_pattern'].format(ontology=ontology)
    
    if model_type in ['sar_cnn']:
        mlb_file = model_dir / config['mlb_file_pattern'].format(ontology_type=ontology_type)
    elif model_type in ['cafa_cnn']:
        mlb_file = model_dir.parent / config['mlb_file_pattern'].format(ontology_type=ontology_type)
    else: 
        mlb_file = model_dir / config['mlb_file_pattern'].format(ontology=ontology)
    
    for file_path in [model_file, params_file, mlb_file]:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    logger.info(f"Loading {model_type} model for {ontology}")
    logger.info(f"Model: {model_file}")
    logger.info(f"Params: {params_file}")
    logger.info(f"MLB: {mlb_file}")
    
    with open(mlb_file, 'rb') as f:
        mlb = pickle.load(f)
    
    num_classes = len(mlb.classes_)
    logger.info(f"Loaded MLB with {num_classes} classes")
    
    if config['architecture'] == 'cnn':
        with open(params_file, 'rb') as f:
            params_data = pickle.load(f)
        
        best_params = params_data['best_params']
        
        model = SimpleCNN(
            num_classes=num_classes,
            max_length=1200,  # Default from training scripts
            conv1_out=best_params['conv1_out'],
            conv2_out=best_params['conv2_out'],
            fc_size=best_params['fc_size'],
            dropout_rate=best_params['dropout_rate']
        ).to(device)
        
    else:  
        with open(params_file, 'r') as f:
            best_params = json.load(f)
        
        model = PFP(
            input_dim=512,  
            hidden_dims=best_params['architecture'],
            output_dim=num_classes,
            dropout_rate=best_params['dropout_rate']
        ).to(device)
    
    
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    
    logger.info(f"Successfully loaded {config['architecture'].upper()} model")
    
    return model, mlb


def predict_with_model(model: nn.Module, data_loader: DataLoader, device: torch.device) -> np.ndarray:
    """Generate predictions - reused from training scripts."""
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    
    return np.concatenate(all_probs, axis=0)


def evaluate_predictions_cafa(predictions: np.ndarray, tsv_file: Path, mlb: MultiLabelBinarizer,
                             ic_dict: Optional[Dict[str, float]], logger: logging.Logger) -> Tuple[Dict, pd.DataFrame]:
    """Evaluate predictions using CAFA-style metrics - adapted from predict_with_uncertainty.py."""
    logger.info(f"Evaluating predictions against {tsv_file}")
    
    
    n_samples = predictions.shape[0]
    dummy_embeddings = np.zeros((n_samples, 512))
    
    try:
        test_GO_df, _, test_GO_list, _ = process_GO_data(str(tsv_file), dummy_embeddings)
    except Exception as e:
        logger.error(f"Failed to process TSV file {tsv_file}: {e}")
        raise
    
    
    if len(test_GO_list) != predictions.shape[0]:
        logger.warning(f"Sample count mismatch: TSV has {len(test_GO_list)}, predictions have {predictions.shape[0]}")
        min_len = min(len(test_GO_list), predictions.shape[0])
        test_GO_list = test_GO_list[:min_len]
        predictions = predictions[:min_len]
    
    
    real_annots_dict = {}
    pred_annots_dict_with_scores = {}
    
    for i, go_terms in enumerate(test_GO_list):
        protein_id = f"protein_{i}"
        real_annots_dict[protein_id] = set(go_terms)
        
        go_scores = {}
        for j, go_term in enumerate(mlb.classes_):
            go_scores[go_term] = predictions[i, j]
        pred_annots_dict_with_scores[protein_id] = go_scores
    
    
    if ic_dict is None:
        logger.warning("No IC dictionary available. Using dummy IC values")
        ic_dict = {go_term: 1.0 for go_term in mlb.classes_}
    
    
    thresholds = np.arange(0.01, 1.00, 0.01)
    
    logger.info("Computing threshold-based metrics...")
    smin, fmax, best_threshold_s, best_threshold_f, s_at_fmax, metrics_df = \
        threshold_performance_metrics(ic_dict, real_annots_dict, pred_annots_dict_with_scores, threshold_range=thresholds)
    
    logger.info("Computing AUPR micro...")
    aupr_micro = calculate_aupr_micro(real_annots_dict, pred_annots_dict_with_scores)
    
    
    fmax_micro = metrics_df['f_micro'].max()
    best_threshold_f_micro_idx = metrics_df['f_micro'].idxmax()
    best_threshold_f_micro = metrics_df.loc[best_threshold_f_micro_idx, 'n']
    
    summary_metrics = {
        'fmax_macro': fmax,
        'fmax_micro': fmax_micro,
        'smin': smin,
        'aupr_micro': aupr_micro,
        'best_threshold_fmax_macro': best_threshold_f,
        'best_threshold_fmax_micro': best_threshold_f_micro,
        'best_threshold_smin': best_threshold_s,
        's_at_fmax': s_at_fmax,
        'total_proteins': len(real_annots_dict),
        'total_go_terms': len(mlb.classes_)
    }
    
    logger.info(f"CAFA Evaluation Results:")
    logger.info(f"  F-max macro: {fmax:.4f} @ threshold {best_threshold_f}")
    logger.info(f"  F-max micro: {fmax_micro:.4f} @ threshold {best_threshold_f_micro}")
    logger.info(f"  S-min: {smin:.4f} @ threshold {best_threshold_s}")
    logger.info(f"  AUPR micro: {aupr_micro:.4f}")
    
    return summary_metrics, metrics_df


def save_results(ontology: str, predictions: np.ndarray, summary_metrics: Optional[Dict],
                metrics_df: Optional[pd.DataFrame], output_dir: Path, logger: logging.Logger):
    """Save prediction results."""
    logger.info(f"Saving results for {ontology}")
    
    ontology_dir = output_dir / ontology.lower()
    ontology_dir.mkdir(parents=True, exist_ok=True)
    
    
    np.save(ontology_dir / 'predictions.npy', predictions)
    
    
    if summary_metrics is not None:
        with open(ontology_dir / 'summary_metrics.json', 'w') as f:
            json.dump(summary_metrics, f, indent=2)
    
    if metrics_df is not None:
        metrics_df.to_csv(ontology_dir / 'detailed_metrics.csv', index=False)
    
    logger.info(f"Results saved to {ontology_dir}")


def prepare_ontology_inputs(args, config, logger: logging.Logger) -> Dict[str, Tuple[Optional[Path], Optional[Path]]]:
    """Prepare ontology-specific input files."""
    ontology_inputs = {}
    
    for ontology in ['BPO', 'CCO', 'MFO']:
        input_file = None
        tsv_file = None
        
        if config['input_type'] == 'sequences':
            
            if ontology == 'BPO':
                input_file = getattr(args, 'bpo_fasta', None)
                tsv_file = getattr(args, 'bpo_tsv', None)
            elif ontology == 'CCO':
                input_file = getattr(args, 'cco_fasta', None)
                tsv_file = getattr(args, 'cco_tsv', None)
            elif ontology == 'MFO':
                input_file = getattr(args, 'mfo_fasta', None)
                tsv_file = getattr(args, 'mfo_tsv', None)
        else:  # embeddings
            
            if ontology == 'BPO':
                input_file = getattr(args, 'bpo_embeddings', None)
                tsv_file = getattr(args, 'bpo_tsv', None)
            elif ontology == 'CCO':
                input_file = getattr(args, 'cco_embeddings', None)
                tsv_file = getattr(args, 'cco_tsv', None)
            elif ontology == 'MFO':
                input_file = getattr(args, 'mfo_embeddings', None)
                tsv_file = getattr(args, 'mfo_tsv', None)
        
        
        if input_file is None:
            if config['input_type'] == 'sequences':
                input_file = args.fasta
            else:
                input_file = args.embeddings
        
        if tsv_file is None:
            if ontology == 'BPO':
                tsv_file = args.bp_tsv
            elif ontology == 'CCO':
                tsv_file = args.cc_tsv
            elif ontology == 'MFO':
                tsv_file = args.mf_tsv
        
        ontology_inputs[ontology] = (input_file, tsv_file)
        
        if input_file:
            logger.info(f"{ontology} input: {input_file}")
        if tsv_file:
            logger.info(f"{ontology} TSV: {tsv_file}")
    
    return ontology_inputs


def main():
    parser = argparse.ArgumentParser(
        description='Unified Model Inference Script v2 - Ontology-Specific Inputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--model_dir', type=Path, required=True,
                       help='Directory containing trained model files')
    parser.add_argument('--model_type', choices=['sar_cnn', 'cafa_cnn', 'cafa_tmvec'], required=True,
                       help='Type of model to load')
    parser.add_argument('--ontology', choices=['BPO', 'CCO', 'MFO', 'ALL'], required=True,
                       help='Ontology to predict (or ALL for all ontologies)')
    parser.add_argument('--output_dir', type=Path, required=True,
                       help='Output directory for results')
    
    parser.add_argument('--bpo_fasta', type=Path, help='BPO-specific FASTA file')
    parser.add_argument('--cco_fasta', type=Path, help='CCO-specific FASTA file')
    parser.add_argument('--mfo_fasta', type=Path, help='MFO-specific FASTA file')
    
    parser.add_argument('--bpo_embeddings', type=Path, help='BPO-specific embeddings file')
    parser.add_argument('--cco_embeddings', type=Path, help='CCO-specific embeddings file')
    parser.add_argument('--mfo_embeddings', type=Path, help='MFO-specific embeddings file')
    
    parser.add_argument('--bpo_tsv', type=Path, help='BPO annotations TSV for evaluation')
    parser.add_argument('--cco_tsv', type=Path, help='CCO annotations TSV for evaluation')
    parser.add_argument('--mfo_tsv', type=Path, help='MFO annotations TSV for evaluation')
    
    parser.add_argument('--fasta', type=Path, help='Input FASTA file (fallback for all ontologies)')
    parser.add_argument('--embeddings', type=Path, help='Pre-computed embeddings (fallback for all ontologies)')
    parser.add_argument('--bp_tsv', type=Path, help='BPO annotations TSV (legacy)')
    parser.add_argument('--cc_tsv', type=Path, help='CCO annotations TSV (legacy)') 
    parser.add_argument('--mf_tsv', type=Path, help='MFO annotations TSV (legacy)')
    
    parser.add_argument('--ic_file', type=Path, help='Information Content TSV file')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for inference')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Device to use for inference')
    parser.add_argument('--max_length', type=int, default=1200,
                       help='Maximum sequence length for CNN models')
    parser.add_argument('--skip_metrics', action='store_true',
                       help='Skip evaluation metrics computation')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir, args.verbose)
    
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    ic_dict = load_ic_dict(args.ic_file, logger)
    
    if args.ontology == 'ALL':
        ontologies = ['BPO', 'CCO', 'MFO']
    else:
        ontologies = [args.ontology]
    
    config = MODEL_TYPES[args.model_type]
    
    ontology_inputs = prepare_ontology_inputs(args, config, logger)
    
    overall_results = {}
    
    try:
        for ontology in ontologies:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {ontology}")
            logger.info(f"{'='*50}")
            
            input_file, tsv_file = ontology_inputs[ontology]
            
            if input_file is None:
                logger.error(f"No input file specified for {ontology}")
                continue
            
            if not input_file.exists():
                logger.error(f"Input file not found for {ontology}: {input_file}")
                continue
            
            if config['input_type'] == 'sequences':
                logger.info(f"Loading sequences from {input_file}")
                sequences = load_fasta_sequences(input_file)
                logger.info(f"Loaded {len(sequences)} sequences")
                
                encoded_data = process_sequences_batch(sequences, args.max_length)
                dataset = TensorDataset(torch.tensor(encoded_data, dtype=torch.float32))
                
            else:  
                logger.info(f"Loading embeddings from {input_file}")
                embeddings = np.load(input_file)
                logger.info(f"Loaded embeddings shape: {embeddings.shape}")
                dataset = TensorDataset(torch.tensor(embeddings, dtype=torch.float32))
            
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            
            model, mlb = load_model_and_params(args.model_dir, ontology, args.model_type, device, logger)
            
            logger.info("Generating predictions...")
            predictions = predict_with_model(model, data_loader, device)
            logger.info(f"Generated predictions shape: {predictions.shape}")
            
            summary_metrics = None
            metrics_df = None
            
            if tsv_file and not args.skip_metrics:
                if tsv_file.exists():
                    try:
                        summary_metrics, metrics_df = evaluate_predictions_cafa(
                            predictions, tsv_file, mlb, ic_dict, logger
                        )
                        overall_results[ontology] = summary_metrics
                    except Exception as e:
                        logger.error(f"Evaluation failed for {ontology}: {e}")
                else:
                    logger.warning(f"TSV file not found for {ontology}: {tsv_file}")
            
            save_results(ontology, predictions, summary_metrics, metrics_df, args.output_dir, logger)
        
        if overall_results:
            with open(args.output_dir / 'overall_summary.json', 'w') as f:
                json.dump(overall_results, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info("PREDICTION COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*60}")
        logger.info(f"Model type: {args.model_type}")
        logger.info(f"Model directory: {args.model_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Processed ontologies: {ontologies}")
        
        if overall_results:
            logger.info("\nPerformance Summary:")
            for ontology, metrics in overall_results.items():
                logger.info(f"  {ontology}:")
                logger.info(f"    F-max macro: {metrics['fmax_macro']:.4f} @ {metrics['best_threshold_fmax_macro']:.2f}")
                logger.info(f"    F-max micro: {metrics['fmax_micro']:.4f} @ {metrics['best_threshold_fmax_micro']:.2f}")
                logger.info(f"    S-min: {metrics['smin']:.4f} @ {metrics['best_threshold_smin']:.2f}")
                logger.info(f"    AUPR micro: {metrics['aupr_micro']:.4f}")
        
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 