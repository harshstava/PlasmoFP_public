#!/usr/bin/env python3
"""
This script trains 1D Convolutional Neural Networks for GO term prediction using raw protein sequences.
Supports all three Gene Ontology aspects: BPO, MFO, and CCO with automated hyperparameter optimization.

USAGE EXAMPLES:

    # Train CNN for a specific ontology with Optuna optimization
    python train_sar_cnn.py --ontology BPO --data_dir /path/to/fasta_and_tsv --output_dir models

    # Train CNNs for all ontologies sequentially
    python train_sar_cnn.py --ontology ALL --data_dir /path/to/fasta_and_tsv --output_dir models

    # Train with custom Optuna parameters
    python train_sar_cnn.py --ontology MFO --data_dir /path/to/fasta_and_tsv --output_dir models \
        --optuna_trials 50 --epochs_per_trial 15

    # Train with predetermined hyperparameters (skip Optuna)
    python train_sar_cnn.py --ontology CCO --data_dir /path/to/fasta_and_tsv --output_dir models \
        --conv1_out 256 --conv2_out 128 --fc_size 512 --lr 0.001 --epochs 20 --skip_optuna

    # Train with custom sequence length and batch size
    python train_sar_cnn.py --ontology ALL --data_dir /path/to/fasta_and_tsv --output_dir models \
        --max_length 1500 --batch_size 64

OUTPUTS:
    For each ontology, the script generates:
    - {ontology}_cnn_best_params.pkl        # Optuna optimization results
    - {ontology}_cnn_best_model.pt          # Best trained CNN model
    - {ontology}_cnn_optimization_log.txt   # Hyperparameter search logs
    - {ontology}_cnn_evaluation.json        # Comprehensive metrics
    - {ontology}_cnn_predictions.pkl        # Test set predictions

EXPECTED DATA STRUCTURE:
    data_dir/
    ├── process_train.fasta          # BPO training sequences
    ├── process_val.fasta            # BPO validation sequences  
    ├── process_test.fasta           # BPO test sequences
    ├── process_train.tsv            # BPO training GO annotations
    ├── process_val.tsv              # BPO validation GO annotations
    ├── process_test.tsv             # BPO test GO annotations
    ├── process_mlb.pkl              # BPO MultiLabelBinarizer
    ├── function_*.fasta/tsv         # MFO files (similar structure)
    ├── function_mlb.pkl             # MFO MultiLabelBinarizer
    ├── component_*.fasta/tsv        # CCO files (similar structure)
    └── component_mlb.pkl            # CCO MultiLabelBinarizer
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Bio import SeqIO
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from tqdm import tqdm
import optuna
import ast

from utils_corrected import threshold_performance_metrics, calculate_aupr_micro, evaluate_annotations

ONTOLOGY_CONFIG = {
    'BPO': {
        'type': 'process',
        'full_name': 'Biological Process'
    },
    'MFO': {
        'type': 'function',
        'full_name': 'Molecular Function'
    },
    'CCO': {
        'type': 'component',
        'full_name': 'Cellular Component'
    }
}

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_INDEX['X'] = len(AMINO_ACIDS)  # Non-standard amino acid encoding


def setup_logging(output_dir: Path, ontology: str) -> logging.Logger:
    """Setup logging for training process."""
    log_file = output_dir / f"cnn_optimization_log_{ontology}.txt"
    
    logger = logging.getLogger(f"SAR_CNN_{ontology}")
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


def read_fasta(file_path: Path) -> List[Tuple[str, str]]:
    """Read FASTA file and return list of (sequence_id, sequence) tuples."""
    sequences = []
    for record in SeqIO.parse(str(file_path), "fasta"):
        sequences.append((record.id, str(record.seq)))
    return sequences


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
    """1D CNN for protein sequence classification."""
    
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


def load_ontology_data(data_dir: Path, ontology: str, max_length: int, logger: logging.Logger) -> Tuple:
    """Load FASTA sequences and GO annotations for specified ontology."""
    ontology_type = ONTOLOGY_CONFIG[ontology]['type']
    logger.info(f"Loading data for {ontology} ({ontology_type})")
    
    train_fasta = data_dir / f"{ontology_type}_train.fasta"
    val_fasta = data_dir / f"{ontology_type}_val.fasta"
    test_fasta = data_dir / f"{ontology_type}_test.fasta"
    
    train_tsv = data_dir / f"{ontology_type}_train.tsv"
    val_tsv = data_dir / f"{ontology_type}_val.tsv"
    test_tsv = data_dir / f"{ontology_type}_test.tsv"
    
    mlb_file = data_dir / f"{ontology_type}_mlb.pkl"
    
    required_files = [train_fasta, val_fasta, test_fasta, train_tsv, val_tsv, test_tsv, mlb_file]
    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    logger.info("Loading FASTA sequences...")
    train_sequences = read_fasta(train_fasta)
    val_sequences = read_fasta(val_fasta)
    test_sequences = read_fasta(test_fasta)
    
    train_seq_ids = [seq[0] for seq in train_sequences]
    train_seqs = [seq[1] for seq in train_sequences]
    
    val_seq_ids = [seq[0] for seq in val_sequences]
    val_seqs = [seq[1] for seq in val_sequences]
    
    test_seq_ids = [seq[0] for seq in test_sequences]
    test_seqs = [seq[1] for seq in test_sequences]
    
    logger.info(f"Loaded {len(train_seqs)} train, {len(val_seqs)} val, {len(test_seqs)} test sequences")
    
    logger.info("One-hot encoding sequences...")
    train_encoded = process_sequences_batch(train_seqs, max_length)
    val_encoded = process_sequences_batch(val_seqs, max_length)
    test_encoded = process_sequences_batch(test_seqs, max_length)
    
    logger.info(f"Encoded shapes: Train {train_encoded.shape}, Val {val_encoded.shape}, Test {test_encoded.shape}")
    
    logger.info("Loading GO annotations...")
    
    def load_go_annotations(tsv_path):
        go_df = pd.read_csv(tsv_path, sep='\t')
        go_df['Raw propagated GO terms'] = go_df['Raw propagated GO terms'].apply(ast.literal_eval)
        return go_df['Raw propagated GO terms'].tolist()
    
    train_go_list = load_go_annotations(train_tsv)
    val_go_list = load_go_annotations(val_tsv)
    test_go_list = load_go_annotations(test_tsv)
    
    with open(mlb_file, 'rb') as f:
        mlb = pickle.load(f)
    
    train_labels = mlb.transform(train_go_list)
    val_labels = mlb.transform(val_go_list)
    test_labels = mlb.transform(test_go_list)
    
    logger.info(f"Label shapes: Train {train_labels.shape}, Val {val_labels.shape}, Test {test_labels.shape}")
    logger.info(f"Number of GO classes: {len(mlb.classes_)}")
    
    return (train_encoded, val_encoded, test_encoded,
            train_labels, val_labels, test_labels,
            train_seq_ids, val_seq_ids, test_seq_ids,
            train_go_list, val_go_list, test_go_list,
            mlb)


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: torch.optim.Optimizer, device: torch.device) -> float:
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


def train_final_model(best_params: Dict, train_loader: DataLoader, val_loader: DataLoader,
                     max_length: int, num_classes: int, device: torch.device, 
                     epochs: int, logger: logging.Logger) -> nn.Module:
    logger.info("Training final model with best hyperparameters")
    
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
    
    for epoch in tqdm(range(epochs), desc="Training final model"):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        if epoch % 5 == 0:
            val_fmax = compute_fmax(model, val_loader, device)
            logger.info(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val F-max {val_fmax:.4f}")
    
    final_fmax = compute_fmax(model, val_loader, device)
    logger.info(f"Final validation F-max: {final_fmax:.4f}")
    
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





def train_ontology_cnn(ontology: str, data_dir: Path, output_dir: Path, 
                      max_length: int = 1200, batch_size: int = 32,
                      optuna_trials: int = 20, epochs_per_trial: int = 10,
                      final_epochs: int = 20, skip_optuna: bool = False,
                      manual_params: Optional[Dict] = None,
                      ic_file: Optional[Path] = None) -> bool:
    """Train CNN for specific ontology."""
    
    try:
        logger = setup_logging(output_dir, ontology)
        logger.info(f"Starting CNN training for {ontology} ({ONTOLOGY_CONFIG[ontology]['full_name']})")
        
        (train_encoded, val_encoded, test_encoded,
         train_labels, val_labels, test_labels,
         train_seq_ids, val_seq_ids, test_seq_ids,
         train_go_list, val_go_list, test_go_list,
         mlb) = load_ontology_data(data_dir, ontology, max_length, logger)
        
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
        test_dataset = TensorDataset(torch.tensor(test_encoded, dtype=torch.float32))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
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
            best_params, train_loader, val_loader, max_length, 
            num_classes, device, final_epochs, logger
        )
        
        model_path = output_dir / f"{ontology}_cnn_best_model.pt"
        params_path = output_dir / f"{ontology}_cnn_best_params.pkl"
        
        torch.save(final_model.state_dict(), model_path)
        with open(params_path, 'wb') as f:
            pickle.dump(optuna_results, f)
        
        logger.info(f"Saved model to {model_path}")
        logger.info(f"Saved parameters to {params_path}")
        
        logger.info("=" * 50)
        logger.info(f"CNN TRAINING COMPLETED FOR {ontology}")
        logger.info("=" * 50)
        
        return True
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error training CNN for {ontology}: {str(e)}")
        else:
            print(f"Error training CNN for {ontology}: {str(e)}")
        return False


def main():
    """Main function to orchestrate CNN training."""
    parser = argparse.ArgumentParser(
        description="Train SAR-CNN models for GO term prediction",
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
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing FASTA files and TSV annotations"
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save trained models and results"
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
    
    if not args.data_dir.exists():
        print(f"Error: Data directory does not exist: {args.data_dir}")
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
    
    print(f"Training CNN models for ontologies: {ontologies}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
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
        print(f"\nStarting CNN training for {ontology}...")
        success = train_ontology_cnn(
            ontology=ontology,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
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
            print(f"✓ Successfully completed CNN training for {ontology}")
        else:
            print(f"✗ Failed to train CNN for {ontology}")
    
    print("\n" + "=" * 60)
    print("CNN TRAINING SUMMARY")
    print("=" * 60)
    print(f"Successfully trained: {success_count}/{total_count} ontologies")
    
    if success_count < total_count:
        print("Some training runs failed. Check log files for details.")
        sys.exit(1)
    else:
        print("All CNN training completed successfully!")


if __name__ == "__main__":
    main() 