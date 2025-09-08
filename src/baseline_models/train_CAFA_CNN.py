#!/usr/bin/env python3
"""
This script trains 1D Convolutional Neural Networks for GO term prediction using the unified CAFA dataset.
Supports all three Gene Ontology aspects: BPO, MFO, and CCO with automated hyperparameter optimization.

USAGE EXAMPLES:

    # Train CNN for a specific ontology with Optuna optimization
    python train_cnn_cafa.py --ontology BPO --cafa_data /path/to/CAFA_5_all_data_proteinData.pkl --mlb_dir /path/to/mlbs --output_dir models

    # Train CNNs for all ontologies sequentially
    python train_cnn_cafa.py --ontology ALL --cafa_data /path/to/CAFA_5_all_data_proteinData.pkl --mlb_dir /path/to/mlbs --output_dir models

    # Train with custom Optuna parameters
    python train_cnn_cafa.py --ontology MFO --cafa_data /path/to/CAFA_5_all_data_proteinData.pkl --mlb_dir /path/to/mlbs --output_dir models \
        --optuna_trials 50 --epochs_per_trial 15

    # Train with predetermined hyperparameters (skip Optuna)
    python train_cnn_cafa.py --ontology CCO --cafa_data /path/to/CAFA_5_all_data_proteinData.pkl --mlb_dir /path/to/mlbs --output_dir models \
        --conv1_out 256 --conv2_out 128 --fc_size 512 --lr 0.001 --epochs 20 --skip_optuna

    # Train with custom validation split and batch size
    python train_cnn_cafa.py --ontology ALL --cafa_data /path/to/CAFA_5_all_data_proteinData.pkl --mlb_dir /path/to/mlbs --output_dir models \
        --val_ratio 0.1 --batch_size 64

OUTPUTS:
    For each ontology, the script generates:
    - {ontology}_cafa_cnn_best_params.pkl        # Optuna optimization results
    - {ontology}_cafa_cnn_best_model.pt          # Best trained CNN model
    - {ontology}_cafa_cnn_optimization_log.txt   # Hyperparameter search logs
    - {ontology}_cafa_cnn_evaluation.json        # Comprehensive metrics
    - {ontology}_cafa_cnn_predictions.pkl        # Test set predictions

EXPECTED DATA STRUCTURE:
    cafa_data: CAFA_5_all_data_proteinData.pkl   # Unified protein dataset
    mlb_dir/
    ├── function_mlb.pkl             # MFO MultiLabelBinarizer
    ├── process_mlb.pkl              # BPO MultiLabelBinarizer
    └── component_mlb.pkl            # CCO MultiLabelBinarizer
"""

import argparse
import json
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from tqdm import tqdm
import optuna

from utils_corrected import threshold_performance_metrics, calculate_aupr_micro, evaluate_annotations

ONTOLOGY_CONFIG = {
    'BPO': {
        'type': 'process',
        'full_name': 'Biological Process',
        'mlb_file': 'process_mlb.pkl'
    },
    'MFO': {
        'type': 'function',
        'full_name': 'Molecular Function',
        'mlb_file': 'function_mlb.pkl'
    },
    'CCO': {
        'type': 'component',
        'full_name': 'Cellular Component',
        'mlb_file': 'component_mlb.pkl'
    }
}

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_INDEX['X'] = len(AMINO_ACIDS)  # Non-standard amino acid encoding


@dataclass
class ProteinData:
    seq_id: str
    tm_vec: Optional[List[float]]
    protT5: Optional[List[float]]
    sequence: str
    mfo_terms: List[str]
    bpo_terms: List[str]
    cco_terms: List[str]


def setup_logging(output_dir: Path, ontology: str) -> logging.Logger:
    """Setup logging for training process."""
    log_file = output_dir / f"cafa_cnn_optimization_log_{ontology}.txt"
    
    logger = logging.getLogger(f"CAFA_CNN_{ontology}")
    logger.setLevel(logging.INFO)
    
    logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def one_hot_encode_sequence(sequence: str, max_length: int) -> np.ndarray:
    """One-hot encode a protein sequence to fixed length."""
    encoding = np.zeros((21, max_length), dtype=np.float16)
    
    for i, aa in enumerate(sequence):
        if i >= max_length:
            break
        index = AA_TO_INDEX.get(aa, AA_TO_INDEX['X'])
        encoding[index, i] = 1.0
    
    return encoding


def process_sequences_batch(sequences: List[str], max_length: int, batch_size: int = 1000) -> np.ndarray:
    """Process sequences in batches for memory efficiency."""
    encoded_sequences = []
    
    for i in tqdm(range(0, len(sequences), batch_size), desc="Encoding sequences"):
        batch_end = min(i + batch_size, len(sequences))
        batch_sequences = sequences[i:batch_end]
        
        batch_encoded = []
        for sequence in batch_sequences:
            batch_encoded.append(one_hot_encode_sequence(sequence, max_length))
        
        encoded_sequences.extend(batch_encoded)
    
    return np.array(encoded_sequences, dtype=np.float16)


class SimpleCNN(nn.Module):
    """1D Convolutional Neural Network for protein sequence classification."""
    
    def __init__(self, num_classes: int, max_length: int, conv1_out: int, conv2_out: int, fc_size: int, dropout_rate: float = 0.5):
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


def load_cafa_data(cafa_file: Path, mlb_dir: Path, ontology: str, val_ratio: float, 
                  max_length: int, logger: logging.Logger) -> Tuple:
    """Load and process CAFA dataset for specified ontology."""
    ontology_config = ONTOLOGY_CONFIG[ontology]
    logger.info(f"Loading CAFA data for {ontology} ({ontology_config['full_name']})")
    
    logger.info(f"Loading protein data from {cafa_file}")
    with open(cafa_file, 'rb') as f:
        protein_data = pickle.load(f)
    logger.info(f"Loaded {len(protein_data)} total proteins from CAFA dataset")
    
    mlb_file = mlb_dir / ontology_config['mlb_file']
    if not mlb_file.exists():
        raise FileNotFoundError(f"MultiLabelBinarizer file not found: {mlb_file}")
    
    with open(mlb_file, 'rb') as f:
        mlb = pickle.load(f)
    logger.info(f"Loaded MultiLabelBinarizer with {len(mlb.classes_)} GO classes")
    
    logger.info(f"Filtering proteins with {ontology} terms...")
    if ontology == 'MFO':
        filtered_proteins = [data for data in protein_data if data.mfo_terms]
        go_terms_attr = 'mfo_terms'
    elif ontology == 'BPO':
        filtered_proteins = [data for data in protein_data if data.bpo_terms]
        go_terms_attr = 'bpo_terms'
    elif ontology == 'CCO':
        filtered_proteins = [data for data in protein_data if data.cco_terms]
        go_terms_attr = 'cco_terms'
    else:
        raise ValueError(f"Unknown ontology: {ontology}")
    
    logger.info(f"Filtered to {len(filtered_proteins)} proteins with {ontology} annotations")
    
    logger.info("Filtering proteins with terms present in MultiLabelBinarizer...")
    final_proteins = []
    for protein in tqdm(filtered_proteins, desc=f"Filtering {ontology} proteins"):
        existing_terms = getattr(protein, go_terms_attr)
        filtered_terms = [term for term in existing_terms if term in mlb.classes_]
        if filtered_terms:
            if ontology == 'MFO':
                final_proteins.append(ProteinData(
                    seq_id=protein.seq_id, sequence=protein.sequence,
                    mfo_terms=filtered_terms, bpo_terms=protein.bpo_terms,
                    cco_terms=protein.cco_terms, tm_vec=protein.tm_vec, protT5=protein.protT5
                ))
            elif ontology == 'BPO':
                final_proteins.append(ProteinData(
                    seq_id=protein.seq_id, sequence=protein.sequence,
                    mfo_terms=protein.mfo_terms, bpo_terms=filtered_terms,
                    cco_terms=protein.cco_terms, tm_vec=protein.tm_vec, protT5=protein.protT5
                ))
            elif ontology == 'CCO':
                final_proteins.append(ProteinData(
                    seq_id=protein.seq_id, sequence=protein.sequence,
                    mfo_terms=protein.mfo_terms, bpo_terms=protein.bpo_terms,
                    cco_terms=filtered_terms, tm_vec=protein.tm_vec, protT5=protein.protT5
                ))
    
    logger.info(f"Final dataset: {len(final_proteins)} proteins after MLB filtering")
    
    sequences = [protein.sequence for protein in final_proteins]
    seq_ids = [protein.seq_id for protein in final_proteins]
    go_terms_list = [getattr(protein, go_terms_attr) for protein in final_proteins]
    
    actual_max_length = max(len(seq) for seq in sequences)
    avg_length = np.mean([len(seq) for seq in sequences])
    logger.info(f"Sequence statistics: Max={actual_max_length}, Avg={avg_length:.1f}, Using max_length={max_length}")
    
    logger.info("One-hot encoding sequences...")
    encoded_sequences = process_sequences_batch(sequences, max_length)
    logger.info(f"Encoded sequences shape: {encoded_sequences.shape}")
    
    logger.info("Transforming GO annotations to binary labels...")
    labels = mlb.transform(go_terms_list)
    logger.info(f"Labels shape: {labels.shape}")
    
    logger.info(f"Splitting data with validation ratio: {val_ratio}")
    total_samples = len(final_proteins)
    val_size = int(total_samples * val_ratio)
    train_size = total_samples - val_size
    
    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_encoded = encoded_sequences[train_indices]
    val_encoded = encoded_sequences[val_indices]
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    train_seq_ids = [seq_ids[i] for i in train_indices]
    val_seq_ids = [seq_ids[i] for i in val_indices]
    train_go_list = [go_terms_list[i] for i in train_indices]
    val_go_list = [go_terms_list[i] for i in val_indices]
    
    logger.info(f"Split complete: Train={len(train_seq_ids)}, Val={len(val_seq_ids)}")
    
    return (train_encoded, val_encoded, encoded_sequences, 
            train_labels, val_labels, labels,
            train_seq_ids, val_seq_ids, seq_ids,
            train_go_list, val_go_list, go_terms_list,
            mlb)


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for sequences, labels in dataloader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def compute_fmax(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    predictions, targets = [], []
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = torch.sigmoid(model(sequences))
            predictions.append(outputs.cpu())
            targets.append(labels.cpu())
    
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    
    thresholds = np.arange(0.1, 1.0, 0.1)
    fmax = 0.0
    
    for threshold in thresholds:
        binarized_preds = (predictions > threshold).int()
        f1 = f1_score(targets, binarized_preds, average='micro', zero_division=0)
        fmax = max(fmax, f1)
    
    return fmax


def train_and_evaluate_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                           criterion: nn.Module, optimizer: torch.optim.Optimizer, 
                           device: torch.device, num_epochs: int) -> float:
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    
    fmax = compute_fmax(model, val_loader, device)
    return fmax


def optuna_objective(trial: optuna.Trial, train_loader: DataLoader, val_loader: DataLoader,
                    max_length: int, num_classes: int, device: torch.device, 
                    epochs_per_trial: int) -> float:

    conv1_out = trial.suggest_int('conv1_out', 32, 512)
    conv2_out = trial.suggest_int('conv2_out', 32, 512)
    fc_size = trial.suggest_int('fc_size', 128, 1024)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    
    model = SimpleCNN(num_classes, max_length, conv1_out, conv2_out, fc_size, dropout_rate).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    fmax = train_and_evaluate_model(model, train_loader, val_loader, criterion, optimizer, 
                                   device, epochs_per_trial)
    
    return fmax


def run_optuna_optimization(train_loader: DataLoader, val_loader: DataLoader, max_length: int,
                           num_classes: int, device: torch.device, n_trials: int, 
                           epochs_per_trial: int, logger: logging.Logger) -> Dict:
    
    logger.info(f"Starting Optuna optimization with {n_trials} trials")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: optuna_objective(trial, train_loader, val_loader, max_length, 
                                     num_classes, device, epochs_per_trial),
        n_trials=n_trials
    )
    
    best_params = study.best_params
    best_fmax = study.best_value
    
    logger.info(f"Optuna optimization complete. Best F-max: {best_fmax:.4f}")
    logger.info(f"Best parameters: {best_params}")
    
    return {
        'best_params': best_params,
        'best_fmax': best_fmax,
        'study': study
    }


def train_final_model(best_params: Dict, full_encoded: np.ndarray, full_labels: np.ndarray,
                     max_length: int, num_classes: int, device: torch.device, 
                     epochs: int, batch_size: int, logger: logging.Logger) -> nn.Module:
    """Train final model on full dataset with best hyperparameters."""
    
    logger.info("Training final model on full dataset with best hyperparameters")
    
    model = SimpleCNN(
        num_classes=num_classes,
        max_length=max_length,
        conv1_out=best_params['conv1_out'],
        conv2_out=best_params['conv2_out'],
        fc_size=best_params['fc_size'],
        dropout_rate=best_params['dropout_rate']
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    
    full_dataset = TensorDataset(
        torch.tensor(full_encoded, dtype=torch.float32),
        torch.tensor(full_labels, dtype=torch.float32)
    )
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    
    logger.info(f"Training final model on {len(full_dataset)} samples for {epochs} epochs")
    
    for epoch in tqdm(range(epochs), desc="Training final model on full data"):
        train_loss = train_epoch(model, full_loader, criterion, optimizer, device)
        
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch}: Train Loss {train_loss:.4f}")
    
    logger.info("Final model training on full dataset completed!")
    
    return model


def predict_with_model(model: nn.Module, data_loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for sequences in data_loader:
            if isinstance(sequences, (list, tuple)):
                sequences = sequences[0]
            sequences = sequences.to(device)
            logits = model(sequences)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    
    return np.concatenate(all_probs, axis=0)





def train_ontology_cafa_cnn(ontology: str, cafa_file: Path, mlb_dir: Path, output_dir: Path, 
                           val_ratio: float = 0.05, max_length: int = 1200, batch_size: int = 32,
                           optuna_trials: int = 20, epochs_per_trial: int = 10,
                           final_epochs: int = 20, skip_optuna: bool = False,
                           manual_params: Optional[Dict] = None,
                           ic_file: Optional[Path] = None) -> bool:
    """Train CNN for specific ontology using CAFA dataset."""
    
    try:
        logger = setup_logging(output_dir, ontology)
        logger.info(f"Starting CAFA CNN training for {ontology} ({ONTOLOGY_CONFIG[ontology]['full_name']})")
        
        (train_encoded, val_encoded, full_encoded,
         train_labels, val_labels, full_labels,
         train_seq_ids, val_seq_ids, full_seq_ids,
         train_go_list, val_go_list, full_go_list,
         mlb) = load_cafa_data(cafa_file, mlb_dir, ontology, val_ratio, max_length, logger)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        train_dataset = TensorDataset(
            torch.tensor(train_encoded, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(val_encoded, dtype=torch.float32),
            torch.tensor(val_labels, dtype=torch.float32)
        )
        val_eval_dataset = TensorDataset(torch.tensor(val_encoded, dtype=torch.float32))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        val_eval_loader = DataLoader(val_eval_dataset, batch_size=batch_size, shuffle=False)
        
        num_classes = len(mlb.classes_)
        
        if skip_optuna and manual_params:
            logger.info("Using manual hyperparameters, skipping Optuna optimization")
            best_params = manual_params
            optuna_results = {'best_params': manual_params, 'best_fmax': None}
        else:
            optuna_results = run_optuna_optimization(
                train_loader, val_loader, max_length, num_classes, device,
                optuna_trials, epochs_per_trial, logger
            )
            best_params = optuna_results['best_params']
        
        final_model = train_final_model(
            best_params, full_encoded, full_labels, max_length, 
            num_classes, device, final_epochs, batch_size, logger
        )
        
        model_path = output_dir / f"{ontology}_cafa_cnn_best_model.pt"
        params_path = output_dir / f"{ontology}_cafa_cnn_best_params.pkl"
        
        torch.save(final_model.state_dict(), model_path)
        with open(params_path, 'wb') as f:
            pickle.dump(optuna_results, f)
        
        logger.info(f"Saved model to {model_path}")
        logger.info(f"Saved parameters to {params_path}")
        
        logger.info("=" * 50)
        logger.info(f"CAFA CNN TRAINING COMPLETED FOR {ontology}")
        logger.info("=" * 50)
        
        return True
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error training CAFA CNN for {ontology}: {str(e)}")
        else:
            print(f"Error training CAFA CNN for {ontology}: {str(e)}")
        return False


def main():
    """Main function to orchestrate CAFA CNN training."""
    parser = argparse.ArgumentParser(
        description="Train CNN models for GO term prediction using CAFA dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--ontology",
        choices=['BPO', 'MFO', 'CCO', 'ALL'],
        required=True,
        help="GO ontology to train (BPO, MFO, CCO, or ALL)"
    )
    
    parser.add_argument(
        "--cafa_data",
        type=Path,
        required=True,
        help="Path to CAFA_5_all_data_proteinData.pkl file"
    )
    
    parser.add_argument(
        "--mlb_dir",
        type=Path,
        required=True,
        help="Directory containing MultiLabelBinarizer pickle files"
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save trained models and results"
    )
    
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Validation set ratio for train/val split"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=1200,
        help="Maximum sequence length for padding/truncation"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--optuna_trials",
        type=int,
        default=20,
        help="Number of Optuna optimization trials"
    )
    
    parser.add_argument(
        "--epochs_per_trial",
        type=int,
        default=10,
        help="Number of epochs per Optuna trial"
    )
    
    parser.add_argument(
        "--final_epochs",
        type=int,
        default=20,
        help="Number of epochs for final model training"
    )
    
    parser.add_argument(
        "--skip_optuna",
        action="store_true",
        help="Skip Optuna optimization and use manual parameters"
    )
    
    parser.add_argument("--conv1_out", type=int, help="First conv layer output channels")
    parser.add_argument("--conv2_out", type=int, help="Second conv layer output channels")
    parser.add_argument("--fc_size", type=int, help="Fully connected layer size")
    parser.add_argument("--dropout_rate", type=float, help="Dropout rate")
    parser.add_argument("--lr", type=float, help="Learning rate")
    
    parser.add_argument(
        "--ic_file",
        type=Path,
        help="Path to information content pickle file for comprehensive evaluation"
    )
    
    args = parser.parse_args()
    
    if not args.cafa_data.exists():
        print(f"Error: CAFA data file does not exist: {args.cafa_data}")
        sys.exit(1)
    
    if not args.mlb_dir.exists():
        print(f"Error: MLB directory does not exist: {args.mlb_dir}")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    manual_params = None
    if args.skip_optuna:
        required_params = ['conv1_out', 'conv2_out', 'fc_size', 'dropout_rate', 'lr']
        if not all(getattr(args, param) is not None for param in required_params):
            print("Error: When using --skip_optuna, must provide all manual parameters:")
            print("--conv1_out, --conv2_out, --fc_size, --dropout_rate, --lr")
            sys.exit(1)
        
        manual_params = {
            'conv1_out': args.conv1_out,
            'conv2_out': args.conv2_out,
            'fc_size': args.fc_size,
            'dropout_rate': args.dropout_rate,
            'lr': args.lr
        }
    
    if args.ontology == 'ALL':
        ontologies = ['BPO', 'MFO', 'CCO']
    else:
        ontologies = [args.ontology]
    
    print(f"Training CAFA CNN models for ontologies: {ontologies}")
    print(f"CAFA data: {args.cafa_data}")
    print(f"MLB directory: {args.mlb_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Validation ratio: {args.val_ratio}")
    print(f"Max sequence length: {args.max_length}")
    print(f"Batch size: {args.batch_size}")
    if not args.skip_optuna:
        print(f"Optuna trials: {args.optuna_trials}")
        print(f"Epochs per trial: {args.epochs_per_trial}")
    print(f"Final training epochs: {args.final_epochs}")
    print("=" * 60)
    
    success_count = 0
    total_count = len(ontologies)
    
    for ontology in ontologies:
        print(f"\nStarting CAFA CNN training for {ontology}...")
        success = train_ontology_cafa_cnn(
            ontology=ontology,
            cafa_file=args.cafa_data,
            mlb_dir=args.mlb_dir,
            output_dir=args.output_dir,
            val_ratio=args.val_ratio,
            max_length=args.max_length,
            batch_size=args.batch_size,
            optuna_trials=args.optuna_trials,
            epochs_per_trial=args.epochs_per_trial,
            final_epochs=args.final_epochs,
            skip_optuna=args.skip_optuna,
            manual_params=manual_params,
            ic_file=args.ic_file
        )
        
        if success:
            success_count += 1
            print(f"✓ Successfully completed CAFA CNN training for {ontology}")
        else:
            print(f"✗ Failed to train CAFA CNN for {ontology}")
    
    print("\n" + "=" * 60)
    print("CAFA CNN TRAINING SUMMARY")
    print("=" * 60)
    print(f"Successfully trained: {success_count}/{total_count} ontologies")
    
    if success_count < total_count:
        print("Some training runs failed. Check log files for details.")
        sys.exit(1)
    else:
        print("All CAFA CNN training completed successfully!")


if __name__ == "__main__":
    main() 