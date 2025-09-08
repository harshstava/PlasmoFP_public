#!/usr/bin/env python3
"""
CAFA-TM-Vec Model

This script trains neural network models for GO term prediction using CAFA sequences
and the same architecture as PlasmoFP models. 

USAGE:
    python train_cafa_models.py --data_dir cafa_preprocessing --output_dir cafa_publish

OUTPUTS:
    For each ontology (MFO, BPO, CCO):
    - {ontology}_best_model.pt           # Best trained model
    - {ontology}_best_params.json        # Best hyperparameters from Optuna
    - {ontology}_optuna_study.pkl        # Complete Optuna study object
    - training_log_{ontology}.txt        # Training logs

"""

import argparse
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
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import optuna
from tqdm import tqdm

from utils_corrected import PFP


def setup_logging(output_dir: Path, ontology: str) -> logging.Logger:
    """Setup logging for training process."""
    log_file = output_dir / f"training_log_{ontology}.txt"
    
    logger = logging.getLogger(f"CAFA_{ontology}")
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


def load_cafa_data(data_dir: Path, ontology: str, logger: logging.Logger) -> Tuple:
    """Load CAFA preprocessed data for specified ontology."""
    
    ontology_mapping = {
        'MFO': ('MFO', 'function'),
        'BPO': ('BPO', 'process'), 
        'CCO': ('CCO', 'component')
    }
    
    prefix, mlb_type = ontology_mapping[ontology]
    
    tm_vec_go_terms_path = data_dir / f"CAFA_5_{prefix}_tm_vec_go_terms.pkl"
    mlb_path = data_dir.parent / f"{mlb_type}_mlb.pkl"
    
    for path in [tm_vec_go_terms_path, mlb_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")
    
    logger.info(f"Loading {ontology} data...")
    
    with open(tm_vec_go_terms_path, 'rb') as f:
        tm_vec_go_terms = pickle.load(f)
    
    with open(mlb_path, 'rb') as f:
        mlb = pickle.load(f)
    
    embeddings = [item[0] for item in tm_vec_go_terms]
    go_terms = [item[1] for item in tm_vec_go_terms]
    
    embeddings = np.array(embeddings)
    
    labels = mlb.transform(go_terms)
    
    logger.info(f"{ontology} data loaded - Embeddings: {embeddings.shape}, Labels: {labels.shape}")
    logger.info(f"Number of GO terms for {ontology}: {len(mlb.classes_)}")
    
    return embeddings, labels, mlb


class ProteinDataset(torch.utils.data.Dataset):
    """Dataset for protein embeddings and GO term labels."""
    
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def compute_fmax(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Compute F-max score for model evaluation."""
    model.eval()
    predictions, targets = [], []
    
    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = torch.sigmoid(model(embeddings))
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


def objective(trial, train_dataset, val_dataset, input_dim: int, output_dim: int, device: torch.device, logger: logging.Logger):
    """Optuna objective function for hyperparameter optimization."""
    
    architectures = [
        [256],
        [256, 128],
        [256, 128, 64],
        [256, 128, 62, 32]
    ]
    
    learning_rates = [0.01, 0.001, 0.0001, 0.00001]
    epoch_counts = [10, 15, 20, 30, 40]
    
    architecture = trial.suggest_categorical("architecture", architectures)
    learning_rate = trial.suggest_categorical("learning_rate", learning_rates)
    epochs = trial.suggest_categorical("epochs", epoch_counts)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    batch_size = 256
    
    logger.info(f"Trial {trial.number}: arch={architecture}, lr={learning_rate}, epochs={epochs}, dropout={dropout_rate:.3f}")
    
    model = PFP(input_dim, architecture, output_dim, dropout_rate).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Trial {trial.number}, Epoch {epoch + 1}/{epochs}: Loss = {running_loss:.4f}")
    
    fmax = compute_fmax(model, val_loader, device)
    logger.info(f"Trial {trial.number} completed: F-max = {fmax:.4f}")
    
    return fmax


def optimize_hyperparameters(embeddings: np.ndarray, labels: np.ndarray, input_dim: int, output_dim: int, 
                           device: torch.device, logger: logging.Logger, n_trials: int = 50) -> Tuple[Dict, optuna.Study]:
    """Perform hyperparameter optimization using Optuna."""
    
    dataset = ProteinDataset(embeddings, labels)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Optimization split - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    study = optuna.create_study(direction="maximize")
    
    def objective_wrapper(trial):
        return objective(trial, train_dataset, val_dataset, input_dim, output_dim, device, logger)
    
    logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
    study.optimize(objective_wrapper, n_trials=n_trials)
    
    best_params = study.best_params
    best_fmax = study.best_value
    
    logger.info(f"Optimization completed!")
    logger.info(f"Best F-max: {best_fmax:.4f}")
    logger.info(f"Best parameters: {best_params}")
    
    return best_params, study


def train_final_model(embeddings: np.ndarray, labels: np.ndarray, best_params: Dict, 
                     input_dim: int, output_dim: int, device: torch.device, logger: logging.Logger) -> nn.Module:
    """Train final model on full dataset using best hyperparameters."""
    
    logger.info("Training final model on full dataset...")
    
    architecture = best_params["architecture"]
    learning_rate = best_params["learning_rate"]
    epochs = best_params["epochs"]
    dropout_rate = best_params["dropout_rate"]
    batch_size = 256
    
    dataset = ProteinDataset(embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = PFP(input_dim, architecture, output_dim, dropout_rate).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    logger.info(f"Training for {epochs} epochs with architecture {architecture}")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for embeddings_batch, labels_batch in dataloader:
            embeddings_batch, labels_batch = embeddings_batch.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        logger.info(f"Epoch {epoch + 1}/{epochs}: Loss = {running_loss:.4f}")
    
    logger.info("Final model training completed!")
    return model


def train_ontology(ontology: str, data_dir: Path, output_dir: Path, device: torch.device, n_trials: int = 50):
    """Train model for a specific ontology."""
    
    ontology_output_dir = output_dir / f"cafa_{ontology.lower()}_publish"
    ontology_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(ontology_output_dir, ontology)
    
    try:
        embeddings, labels, mlb = load_cafa_data(data_dir, ontology, logger)
        input_dim = embeddings.shape[1]  # Should be 512 for TM-Vec
        output_dim = labels.shape[1]     # Number of GO terms
        
        best_params, study = optimize_hyperparameters(
            embeddings, labels, input_dim, output_dim, device, logger, n_trials
        )
        
        final_model = train_final_model(
            embeddings, labels, best_params, input_dim, output_dim, device, logger
        )
        
        model_path = ontology_output_dir / f"{ontology}_best_model.pt"
        params_path = ontology_output_dir / f"{ontology}_best_params.json"
        study_path = ontology_output_dir / f"{ontology}_optuna_study.pkl"
        mlb_path = ontology_output_dir / f"{ontology}_mlb.pkl"
        
        torch.save(final_model.state_dict(), model_path)
        
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
            
        with open(mlb_path, 'wb') as f:
            pickle.dump(mlb, f)
        
        logger.info(f"Model and results saved to {ontology_output_dir}")
        logger.info(f"Final model saved: {model_path}")
        logger.info(f"Best parameters saved: {params_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed for {ontology}: {str(e)}")
        return False


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train CAFA models using preprocessed data")
    parser.add_argument("--data_dir", type=str, required=True, 
                       help="Directory containing CAFA preprocessing outputs")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save trained models")
    parser.add_argument("--ontology", type=str, choices=["MFO", "BPO", "CCO", "ALL"], 
                       default="ALL", help="Ontology to train (default: ALL)")
    parser.add_argument("--n_trials", type=int, default=50,
                       help="Number of Optuna trials for hyperparameter optimization")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    ontologies = ["MFO", "BPO", "CCO"] if args.ontology == "ALL" else [args.ontology]
    
    results = {}
    for ontology in ontologies:
        print(f"\n{'='*60}")
        print(f"Training {ontology} ontology")
        print(f"{'='*60}")
        
        success = train_ontology(ontology, data_dir, output_dir, device, args.n_trials)
        results[ontology] = success
        
        if success:
            print(f"✓ {ontology} training completed successfully")
        else:
            print(f"✗ {ontology} training failed")
    
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for ontology, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"{ontology}: {status}")
    
    successful_count = sum(results.values())
    total_count = len(results)
    print(f"\nCompleted {successful_count}/{total_count} ontologies successfully")


if __name__ == "__main__":
    main() 