#!/usr/bin/env python3
"""
This script generates protein embeddings using TM-Vec models.
Supports processing single FASTA files or directories containing multiple FASTA files.
Outputs embeddings in both pickle (dictionary format) and numpy array formats.

Dependencies:
    - tm-vec
    - transformers
    - torch
    - biopython
    - numpy
    - faiss-cpu

## Required Model Files

Download the TM-Vec model files:
- `tm_vec_cath_model.ckpt` - TM-Vec model checkpoint
- `tm_vec_cath_model_params.json` - TM-Vec configuration file

## Usage Examples

### Process a single FASTA file
python generate_embeddings.py \
    --input protein_sequences.fasta \
    --output ./embeddings/ \
    --tm_vec_model tm_vec_cath_model.ckpt \
    --tm_vec_config tm_vec_cath_model_params.json

### Process a directory of FASTA files
python generate_embeddings.py \
    --input ./fasta_files/ \
    --output ./embeddings/ \
    --tm_vec_model tm_vec_cath_model.ckpt \
    --tm_vec_config tm_vec_cath_model_params.json

### Use default model paths (if files are in current directory)
python generate_embeddings.py \
    --input ./fasta_files/ \
    --output ./embeddings/

### Force CPU usage
python generate_embeddings.py \
    --input ./fasta_files/ \
    --output ./embeddings/ \
    --device cpu

"""

import argparse
import logging
import gc
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Union, Tuple

import numpy as np
import torch
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm

from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
from tm_vec.tm_vec_utils import encode


def setup_logging(output_dir: Path) -> logging.Logger:
    """
    Configure logging for the embedding generation process.
    
    Args:
        output_dir: Directory where log file will be saved
        
    Returns:
        Configured logger instance
    """
    log_file = output_dir / "embedding_generation.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def validate_inputs(input_path: Path, tm_vec_model: Path, tm_vec_config: Path) -> bool:
    """
    Validate input paths and model files exist.
    
    Args:
        input_path: Path to input FASTA file or directory
        tm_vec_model: Path to TM-Vec model checkpoint
        tm_vec_config: Path to TM-Vec configuration file
        
    Returns:
        True if all inputs are valid, False otherwise
    """
    if not input_path.exists():
        logging.error(f"Input path does not exist: {input_path}")
        return False
    
    if not tm_vec_model.exists():
        logging.error(f"TM-Vec model checkpoint not found: {tm_vec_model}")
        return False
    
    if not tm_vec_config.exists():
        logging.error(f"TM-Vec config file not found: {tm_vec_config}")
        return False
    
    return True


def get_fasta_files(input_path: Path) -> List[Path]:
    """
    Get list of FASTA files to process.
    
    Args:
        input_path: Path to single FASTA file or directory containing FASTA files
        
    Returns:
        List of FASTA file paths
    """
    if input_path.is_file():
        if input_path.suffix.lower() in ['.fasta', '.fa', '.fas']:
            return [input_path]
        else:
            logging.warning(f"File {input_path} does not have a recognized FASTA extension")
            return [input_path]  # Still try to process it
    
    elif input_path.is_dir():
        fasta_files = []
        for ext in ['*.fasta', '*.fa', '*.fas']:
            fasta_files.extend(input_path.glob(ext))
        
        if not fasta_files:
            logging.error(f"No FASTA files found in directory: {input_path}")
            return []
        
        return sorted(fasta_files)
    
    else:
        logging.error(f"Input path is neither file nor directory: {input_path}")
        return []


def load_models(tm_vec_model_path: Path, tm_vec_config_path: Path, device: torch.device) -> Tuple[object, object, object]:
    """
    Load and initialize ProtT5 and TM-Vec models.
    
    Args:
        tm_vec_model_path: Path to TM-Vec model checkpoint
        tm_vec_config_path: Path to TM-Vec configuration file
        device: PyTorch device for model placement
        
    Returns:
        Tuple of (model_deep, model, tokenizer)
    """
    logging.info("Loading ProtT5 tokenizer and model...")
    
    # Load ProtT5 model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    
    # Move to device and set to evaluation mode
    model = model.to(device)
    model = model.eval()
    
    logging.info("Loading TM-Vec model...")
    
    # Load TM-Vec model
    tm_vec_config = trans_basic_block_Config.from_json(str(tm_vec_config_path))
    model_deep = trans_basic_block.load_from_checkpoint(
        str(tm_vec_model_path), 
        config=tm_vec_config
    )
    model_deep = model_deep.to(device)
    model_deep = model_deep.eval()
    
    # Clean up memory
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    logging.info(f"Models loaded successfully on device: {device}")
    
    return model_deep, model, tokenizer


def process_fasta_file(
    fasta_path: Path,
    output_dir: Path,
    model_deep: object,
    model: object,
    tokenizer: object,
    device: torch.device
) -> bool:
    """
    Process a single FASTA file and generate embeddings.
    
    Args:
        fasta_path: Path to input FASTA file
        output_dir: Directory for output files
        model_deep: TM-Vec model instance
        model: ProtT5 model instance
        tokenizer: ProtT5 tokenizer instance
        device: PyTorch device
        
    Returns:
        True if processing successful, False otherwise
    """
    try:
        logging.info(f"Processing file: {fasta_path.name}")
        
        # Read sequences from FASTA file
        sequences = list(SeqIO.parse(str(fasta_path), "fasta"))
        
        if not sequences:
            logging.warning(f"No sequences found in {fasta_path}")
            return False
        
        # Extract sequence IDs
        seq_ids = [rec.id for rec in sequences]
        
        logging.info(f"Found {len(sequences)} sequences in {fasta_path.name}")
        
        # Generate embeddings
        logging.info("Generating embeddings...")
        embeddings = encode(sequences, model_deep, model, tokenizer, device)
        
        # Define output file paths
        pkl_output = output_dir / f"{fasta_path.stem}_embeddings.pkl"
        npy_output = output_dir / f"{fasta_path.stem}_embeddings.npy"
        
        # Save pickle file (ID to embedding mapping)
        with open(pkl_output, "wb") as f:
            pickle.dump(dict(zip(seq_ids, embeddings)), f)
        
        # Save numpy array
        np.save(npy_output, embeddings)
        
        # Log results
        logging.info(
            f"Successfully processed {fasta_path.name}: "
            f"{len(sequences)} sequences → {pkl_output.name}, {npy_output.name}"
        )
        logging.info(f"Embedding shape: {embeddings.shape}")
        
        # Clean up memory
        del embeddings
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logging.error(f"Error processing {fasta_path}: {str(e)}")
        return False


def main():
    """
    Main function to orchestrate the embedding generation process.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate protein embeddings using TM-Vec and ProtT5 models"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input FASTA file or directory containing FASTA files"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for embedding files"
    )
    
    parser.add_argument(
        "--tm_vec_model",
        type=Path,
        help="Path to TM-Vec model checkpoint file (.ckpt)"
    )
    
    parser.add_argument(
        "--tm_vec_config",
        type=Path,
        help="Path to TM-Vec configuration file (.json)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for computation (default: auto-detect)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(args.output)
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    if not args.tm_vec_model:
        args.tm_vec_model = Path("tm_vec_cath_model.ckpt")
        
    if not args.tm_vec_config:
        args.tm_vec_config = Path("tm_vec_cath_model_params.json")
    
    if not validate_inputs(args.input, args.tm_vec_model, args.tm_vec_config):
        sys.exit(1)
    
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logging.info(f"Using device: {device}")
    
    fasta_files = get_fasta_files(args.input)
    if not fasta_files:
        sys.exit(1)
    
    logging.info(f"Found {len(fasta_files)} FASTA file(s) to process")
    
    try:
        model_deep, model, tokenizer = load_models(
            args.tm_vec_model, 
            args.tm_vec_config, 
            device
        )
    except Exception as e:
        logging.error(f"Failed to load models: {str(e)}")
        sys.exit(1)
    
    successful_files = 0
    failed_files = 0
    
    for fasta_file in tqdm(fasta_files, desc="Processing FASTA files"):
        success = process_fasta_file(
            fasta_file,
            args.output,
            model_deep,
            model,
            tokenizer,
            device
        )
        
        if success:
            successful_files += 1
        else:
            failed_files += 1
    
    logging.info("=" * 50)
    logging.info(f"Successfully processed: {successful_files} files")
    logging.info(f"Failed to process: {failed_files} files")
    logging.info(f"Output directory: {args.output}")
    logging.info("=" * 50)
    
    if failed_files > 0:
        sys.exit(1)


if __name__ == "__main__":
    main() 