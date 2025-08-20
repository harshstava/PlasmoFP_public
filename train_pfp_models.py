#!/usr/bin/env python3
"""
This script trains neural network models for GO term prediction across all three Gene Ontology aspects:
- BPO (Biological Process)
- MFO (Molecular Function) 
- CCO (Cellular Component)

The script produces both deterministic models and k-fold ensemble models for uncertainty quantification.

USAGE EXAMPLES:

    # Train models for a specific ontology
    python train_pfp_models.py --ontology BPO --data_dir processed_data_90_30 --output_dir models

    # Train models for all ontologies sequentially
    python train_pfp_models.py --ontology ALL --data_dir processed_data_90_30 --output_dir models

OUTPUTS:
    For each ontology, the script generates:
    - {ontology}_best_model.pt           # Deterministic model
    - {ontology}_epoch_states.pt         # Epoch states for temporal ensemble
    - {ontology}_k{k}_fold{i}.pt         # K-fold ensemble models  
    - {ontology}_mlb.pkl                 # MultiLabelBinarizer
    - training_log_{ontology}.txt        # Training logs

DEPENDENCIES:
    - utils.py (must be in same directory)
    - TUNED_MODEL_ARCHS.json (model configurations)
    - Processed training data in specified data directory
"""

import argparse
import copy
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tqdm import tqdm

from utils_corrected import process_GO_data, PFP

# Ontology configuration mapping
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


def setup_logging(output_dir: Path, ontology: str) -> logging.Logger:
    log_file = output_dir / f"training_log_{ontology}.txt"
    
    logger = logging.getLogger(f"PFP_{ontology}")
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


def load_tuned_configs(config_path: Path) -> Dict:
    with open(config_path, 'r') as f:
        return json.load(f)


def load_data(data_dir: Path, ontology_type: str, logger: logging.Logger) -> Tuple:
    train_embeddings_path = data_dir / f"{ontology_type}_train.npy"
    train_tsv_path = data_dir / f"{ontology_type}_train.tsv"
    val_embeddings_path = data_dir / f"{ontology_type}_val.npy"
    val_tsv_path = data_dir / f"{ontology_type}_val.tsv"
    
    for path in [train_embeddings_path, train_tsv_path, val_embeddings_path, val_tsv_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")
    
    logger.info("Loading embeddings and annotations...")
    
    train_embeddings = np.load(train_embeddings_path)
    val_embeddings = np.load(val_embeddings_path)
    
    train_tsv, train_embeddings, train_GO_list, train_GO_annotated = process_GO_data(
        str(train_tsv_path), train_embeddings
    )
    val_tsv, val_embeddings, val_GO_list, val_GO_annotated = process_GO_data(
        str(val_tsv_path), val_embeddings
    )
    
    logger.info(f"Training data shape: {train_tsv.shape}, Validation data shape: {val_tsv.shape}")
    
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform(train_GO_list)
    val_labels = mlb.transform(val_GO_list)

    #save the mlb
    with open(data_dir / f"{ontology_type}_mlb.pkl", 'wb') as f:
        pickle.dump(mlb, f)
    
    logger.info(f"Label shapes - Train: {train_labels.shape}, Val: {val_labels.shape}")
    
    return (train_embeddings, val_embeddings, train_labels, val_labels, mlb)


def get_criterion(loss_type: str) -> nn.Module:
    """Get loss criterion based on configuration."""
    if loss_type in ['Balanced', 'Balanced BCE']:
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'BCEWithLogits':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            running_loss += criterion(outputs, y_batch).item()
    return running_loss / len(loader)


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    patience: int = 3,
    min_delta: float = 1e-4,
    logger: Optional[logging.Logger] = None
) -> Tuple[nn.Module, Dict]:
    """Train model with early stopping."""
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    train_losses, val_losses = [], []
    model_states = {}
    
    if logger:
        logger.info(f"Starting training with early stopping (patience={patience})")
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        running_train = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_train += loss.item()
        
        avg_train = running_train / len(train_loader)
        train_losses.append(avg_train)
        
        # Validation phase
        avg_val = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(avg_val)
        
        # Save epoch state
        model_states[epoch] = copy.deepcopy(model.state_dict())
        
        # Log progress
        if logger:
            logger.info(f"Epoch {epoch:>2}/{num_epochs}  Train Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f}")
        
        # Check for improvement
        if best_val_loss - avg_val > min_delta:
            best_val_loss = avg_val
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if logger:
                logger.info("  ↳ Val loss improved; saving best weights.")
        else:
            epochs_no_improve += 1
            if logger:
                logger.info(f"  ↳ No improvement for {epochs_no_improve} epoch(s).")
        
        # Early stopping check
        if epochs_no_improve >= patience:
            if logger:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
            break
    
    # Restore best weights
    model.load_state_dict(best_model_wts)
    if logger:
        logger.info(f"Training complete. Best Val Loss: {best_val_loss:.4f}")
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_states': model_states,
        'best_val_loss': best_val_loss
    }


def train_deterministic_model(
    ontology: str,
    config: Dict,
    train_embeddings: np.ndarray,
    val_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
    batch_size: int,
    epochs: int,
    patience: int,
    logger: logging.Logger
) -> Tuple[nn.Module, Dict]:
    """Train a single deterministic model."""
    
    logger.info(f"Training deterministic model for {ontology}")
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(train_embeddings, dtype=torch.float).to(device),
        torch.tensor(train_labels, dtype=torch.float).to(device)
    )
    val_dataset = TensorDataset(
        torch.tensor(val_embeddings, dtype=torch.float).to(device),
        torch.tensor(val_labels, dtype=torch.float).to(device)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = PFP(
        input_dim=train_embeddings.shape[1],
        hidden_dims=config['CHOSEN_CONFIG_ARCH'],
        output_dim=config['NUM_CLASSES'],
        dropout_rate=0.3  # Fixed gamma value
    ).to(device)
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=config['CHOSEN_CONFIG_LR'])
    criterion = get_criterion('Balanced')
    
    # Train model
    model, training_info = train_with_early_stopping(
        model, train_loader, val_loader, criterion, optimizer,
        epochs, device, patience, 1e-4, logger
    )
    
    return model, training_info


def train_kfold_ensemble(
    ontology: str,
    config: Dict,
    train_embeddings: np.ndarray,
    val_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
    batch_size: int,
    epochs: int,
    patience: int,
    k_folds: List[int],
    output_dir: Path,
    logger: logging.Logger
) -> Dict:
    """Train k-fold ensemble models."""
    
    logger.info(f"Training k-fold ensemble models for {ontology} with k={k_folds}")
    
    # Create validation loader (shared across all folds)
    val_dataset = TensorDataset(
        torch.tensor(val_embeddings, dtype=torch.float32),
        torch.tensor(val_labels, dtype=torch.float32)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    all_ensembles = {}
    
    for k in k_folds:
        logger.info(f"Training {k}-fold ensemble")
        
        # Initialize stratified k-fold
        kf = MultilabelStratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        fold_states = []
        
        for fold_idx, (train_idx, _) in enumerate(kf.split(train_embeddings, train_labels), start=1):
            logger.info(f"Training fold {fold_idx}/{k}")
            
            # Create fold-specific training data
            fold_train_X = torch.tensor(train_embeddings[train_idx], dtype=torch.float32)
            fold_train_Y = torch.tensor(train_labels[train_idx], dtype=torch.float32)
            fold_train_loader = DataLoader(
                TensorDataset(fold_train_X, fold_train_Y),
                batch_size=batch_size,
                shuffle=True
            )
            
            # Initialize fresh model for this fold
            model = PFP(
                input_dim=train_embeddings.shape[1],
                hidden_dims=config['CHOSEN_CONFIG_ARCH'],
                output_dim=config['NUM_CLASSES'],
                dropout_rate=0.3
            ).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=config['CHOSEN_CONFIG_LR'])
            criterion = get_criterion('Balanced')
            
            # Train fold model
            fold_model, _ = train_with_early_stopping(
                model, fold_train_loader, val_loader, criterion, optimizer,
                epochs, device, patience, 1e-4, None  # Skip detailed logging for folds
            )
            
            # Save fold model in proper directory structure
            k_fold_dir = output_dir / "k_folds" / str(k)
            k_fold_dir.mkdir(parents=True, exist_ok=True)
            fold_filename = f"{ontology}_k{k}_fold{fold_idx}.pt"
            fold_path = k_fold_dir / fold_filename
            torch.save(fold_model.state_dict(), fold_path)
            fold_states.append((fold_idx, fold_model.state_dict()))
            
            logger.info(f"Saved fold {fold_idx} to {fold_path}")
        
        all_ensembles[k] = fold_states
    
    return all_ensembles


def train_ontology(
    ontology: str,
    data_dir: Path,
    output_dir: Path,
    config_path: Path,
    batch_size: int = 1024,
    epochs: int = 30,
    patience: int = 3,
    k_folds: Optional[List[int]] = None,
    skip_ensemble: bool = False
) -> bool:
    """Train models for a specific ontology."""
    
    try:
        # Setup logging
        logger = setup_logging(output_dir, ontology)
        logger.info(f"Starting training for {ontology} ({ONTOLOGY_CONFIG[ontology]['full_name']})")
        
        # Load configurations
        tuned_configs = load_tuned_configs(config_path)
        ontology_config = tuned_configs[ontology]
        ontology_type = ONTOLOGY_CONFIG[ontology]['type']
        
        logger.info(f"Configuration: {ontology_config['CHOSEN_CONFIG']}")
        
        # Load data
        train_embeddings, val_embeddings, train_labels, val_labels, mlb = load_data(
            data_dir, ontology_type, logger
        )
        
        # Save MultiLabelBinarizer
        mlb_path = output_dir / f"{ontology}_mlb.pkl"
        with open(mlb_path, 'wb') as f:
            pickle.dump(mlb, f)
        logger.info(f"Saved MultiLabelBinarizer to {mlb_path}")
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Train deterministic model
        logger.info("=" * 50)
        logger.info("TRAINING DETERMINISTIC MODEL")
        logger.info("=" * 50)
        
        det_model, det_info = train_deterministic_model(
            ontology, ontology_config, train_embeddings, val_embeddings,
            train_labels, val_labels, device, batch_size, epochs, patience, logger
        )
        
        # Save deterministic model
        det_model_path = output_dir / f"{ontology}_best_model.pt"
        torch.save(det_model.state_dict(), det_model_path)
        logger.info(f"Saved deterministic model to {det_model_path}")
        
        # Save epoch states for temporal ensemble
        epoch_states_path = output_dir / f"{ontology}_epoch_states.pt"
        torch.save(det_info['model_states'], epoch_states_path)
        logger.info(f"Saved {len(det_info['model_states'])} epoch states to {epoch_states_path}")
        
        # Train k-fold ensembles (if not skipped)
        if not skip_ensemble:
            logger.info("=" * 50)
            logger.info("TRAINING K-FOLD ENSEMBLES")
            logger.info("=" * 50)
            
            if k_folds is None:
                k_folds = [5, 10, 20]  # Default k-fold configurations
            
            ensemble_info = train_kfold_ensemble(
                ontology, ontology_config, train_embeddings, val_embeddings,
                train_labels, val_labels, device, batch_size, epochs, patience,
                k_folds, output_dir, logger
            )
            
            logger.info(f"Completed k-fold ensemble training for k={k_folds}")
        
        logger.info("=" * 50)
        logger.info(f"TRAINING COMPLETED FOR {ontology}")
        logger.info("=" * 50)
        
        return True
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error training {ontology}: {str(e)}")
        else:
            print(f"Error training {ontology}: {str(e)}")
        return False


def main():
    """Main function to orchestrate training."""
    parser = argparse.ArgumentParser(
        description="Train PFP models for GO term prediction",
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
        help="Directory containing training data (embeddings and TSV files)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save trained models"
    )
    
    parser.add_argument(
        "--config_path",
        type=Path,
        default=Path("TUNED_MODEL_ARCHS.json"),
        help="Path to model configuration JSON file"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum number of training epochs"
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience"
    )
    
    parser.add_argument(
        "--k_folds",
        type=int,
        nargs='+',
        default=[5, 10, 20],
        help="K-fold configurations for ensemble training"
    )
    
    parser.add_argument(
        "--skip_ensemble",
        action="store_true",
        help="Skip k-fold ensemble training (train only deterministic models)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.data_dir.exists():
        print(f"Error: Data directory does not exist: {args.data_dir}")
        sys.exit(1)
    
    if not args.config_path.exists():
        print(f"Error: Configuration file does not exist: {args.config_path}")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine ontologies to train
    if args.ontology == 'ALL':
        ontologies = ['BPO', 'MFO', 'CCO']
    else:
        ontologies = [args.ontology]
    
    print(f"Training ontologies: {ontologies}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"K-fold configurations: {args.k_folds}")
    print("=" * 60)
    
    # Train each ontology
    success_count = 0
    total_count = len(ontologies)
    
    for ontology in ontologies:
        print(f"\nStarting training for {ontology}...")
        success = train_ontology(
            ontology=ontology,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config_path=args.config_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            k_folds=args.k_folds,
            skip_ensemble=args.skip_ensemble
        )
        
        if success:
            success_count += 1
            print(f"✓ Successfully completed training for {ontology}")
        else:
            print(f"✗ Failed to train {ontology}")
    
    # Final summary
    print("\n" + "=" * 60)
    print(f"Successfully trained: {success_count}/{total_count} ontologies")
    print("All training completed successfully!")


if __name__ == "__main__":
    main() 