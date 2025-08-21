"""
This script processes CAFA-5 training data by:
1. Loading protein sequences, TM-Vec embeddings, and GO term annotations
2. Creating structured protein data objects
3. Filtering data based on existing vocabularies
4. Exporting processed data in various formats

Input Requirements:
- CAFA_5_train_sequences.fasta: FASTA file with protein sequences
- CAFA_train_TMVec.pkl: TM-Vec embeddings for the sequences
- CAFA_5_train_terms.tsv: TSV file with GO term annotations
- function_mlb.pkl: MultiLabelBinarizer for MFO terms (for filtering)
- process_mlb.pkl: MultiLabelBinarizer for BPO terms (for filtering)
- component_mlb.pkl: MultiLabelBinarizer for CCO terms (for filtering)

Output Products:
- CAFA_5_all_data_proteinData.pkl: All protein data objects (compatible with train_cnn_cafa.py)
- CAFA_5_MFO_tm_vec_go_terms.pkl: MFO data tuples
- CAFA_5_BPO_tm_vec_go_terms.pkl: BPO data tuples
- CAFA_5_CCO_tm_vec_go_terms.pkl: CCO data tuples
- CAFA_5_MFO_filtered_sequences.fasta: Filtered MFO sequences
- CAFA_5_BPO_filtered_sequences.fasta: Filtered BPO sequences
- CAFA_5_CCO_filtered_sequences.fasta: Filtered CCO sequences
"""

from dataclasses import dataclass
from typing import List, Optional
import os
import pickle
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
import warnings


@dataclass
class ProteinData:
    """Data structure to hold protein information including embeddings and GO terms."""
    seq_id: str
    tm_vec: Optional[List[float]]
    protT5: Optional[List[float]]
    sequence: str
    mfo_terms: List[str]
    bpo_terms: List[str]
    cco_terms: List[str]


def load_sequences(fasta_path: str, max_length: int = 1200) -> List[SeqRecord]:
    """
    Load protein sequences from FASTA file and filter by length.
    
    Args:
        fasta_path: Path to FASTA file
        max_length: Maximum sequence length to keep
        
    Returns:
        List of filtered SeqRecord objects
    """
    print(f"Loading sequences from {fasta_path}")
    sequences = list(SeqIO.parse(fasta_path, 'fasta'))
    print(f"Total sequences before filtering: {len(sequences)}")
    
    filtered_sequences = [seq for seq in sequences if len(seq.seq) <= max_length]
    print(f"Total sequences after filtering (length <= {max_length}): {len(filtered_sequences)}")
    
    return filtered_sequences


def load_embeddings(embeddings_path: str):
    """Load TM-Vec embeddings from pickle file."""
    print(f"Loading TM-Vec embeddings from {embeddings_path}")
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"TMVec shape: {embeddings.shape}")
    return embeddings


def load_go_terms(terms_path: str):
    """
    Load GO terms from TSV file and create mappings by aspect.
    
    Args:
        terms_path: Path to TSV file with GO terms
        
    Returns:
        Tuple of (mfo_mapping, bpo_mapping, cco_mapping)
    """
    print(f"Loading GO terms from {terms_path}")
    terms_df = pd.read_csv(terms_path, sep='\t')
    
    mfo_terms = terms_df[terms_df['aspect'] == 'MFO']
    bpo_terms = terms_df[terms_df['aspect'] == 'BPO']
    cco_terms = terms_df[terms_df['aspect'] == 'CCO']
    
    def create_terms_mapping(terms_df):
        """Create mapping from EntryID to list of terms."""
        return terms_df.groupby('EntryID')['term'].apply(list).to_dict()
    
    mfo_mapping = create_terms_mapping(mfo_terms)
    bpo_mapping = create_terms_mapping(bpo_terms)
    cco_mapping = create_terms_mapping(cco_terms)
    
    print(f"MFO terms mapping: {len(mfo_mapping)} proteins")
    print(f"BPO terms mapping: {len(bpo_mapping)} proteins")
    print(f"CCO terms mapping: {len(cco_mapping)} proteins")
    
    return mfo_mapping, bpo_mapping, cco_mapping


def create_protein_data_list(sequences, embeddings, mfo_mapping, bpo_mapping, cco_mapping):
    """
    Create list of ProteinData objects from sequences, embeddings, and GO term mappings.
    
    Args:
        sequences: List of SeqRecord objects
        embeddings: NumPy array of TM-Vec embeddings
        mfo_mapping: Dict mapping protein IDs to MFO terms
        bpo_mapping: Dict mapping protein IDs to BPO terms
        cco_mapping: Dict mapping protein IDs to CCO terms
        
    Returns:
        List of ProteinData objects
    """
    print("Creating protein data objects...")
    protein_data_list = []
    
    for i, seq in enumerate(tqdm(sequences, desc="Processing sequences")):
        seq_id = seq.id
        tm_vec = embeddings[i] if i < len(embeddings) else None
        mfo_terms = mfo_mapping.get(seq_id, [])
        bpo_terms = bpo_mapping.get(seq_id, [])
        cco_terms = cco_mapping.get(seq_id, [])
        
        protein_data = ProteinData(
            seq_id=seq_id,
            tm_vec=tm_vec,
            protT5=None,
            sequence=str(seq.seq),
            mfo_terms=mfo_terms,
            bpo_terms=bpo_terms,
            cco_terms=cco_terms
        )
        protein_data_list.append(protein_data)
    
    return protein_data_list


def count_terms_by_aspect(protein_data_list):
    """Count and print statistics about proteins with different GO term aspects."""
    mfo_count = len([data for data in protein_data_list if data.mfo_terms])
    bpo_count = len([data for data in protein_data_list if data.bpo_terms])
    cco_count = len([data for data in protein_data_list if data.cco_terms])
    
    print(f"Number of proteins with MFO terms: {mfo_count}")
    print(f"Number of proteins with BPO terms: {bpo_count}")
    print(f"Number of proteins with CCO terms: {cco_count}")
    
    return mfo_count, bpo_count, cco_count


def filter_by_aspect(protein_data_list):
    """Filter protein data by GO term aspect."""
    print("Filtering protein data by aspect...")
    
    mfo_protein_data = [data for data in protein_data_list if data.mfo_terms]
    bpo_protein_data = [data for data in protein_data_list if data.bpo_terms]
    cco_protein_data = [data for data in protein_data_list if data.cco_terms]
    
    print(f"MFO protein data: {len(mfo_protein_data)} proteins")
    print(f"BPO protein data: {len(bpo_protein_data)} proteins")
    print(f"CCO protein data: {len(cco_protein_data)} proteins")
    
    return mfo_protein_data, bpo_protein_data, cco_protein_data


def save_tm_vec_go_terms(mfo_data, bpo_data, cco_data):
    """Save TM-Vec and GO terms tuples for each aspect."""
    print("Saving TM-Vec and GO terms tuples...")
    
    mfo_tm_vec_go_terms = [(data.tm_vec, data.mfo_terms) for data in mfo_data]
    bpo_tm_vec_go_terms = [(data.tm_vec, data.bpo_terms) for data in bpo_data]
    cco_tm_vec_go_terms = [(data.tm_vec, data.cco_terms) for data in cco_data]
    
    with open('CAFA_5_MFO_tm_vec_go_terms.pkl', 'wb') as f:
        pickle.dump(mfo_tm_vec_go_terms, f)
    
    with open('CAFA_5_BPO_tm_vec_go_terms.pkl', 'wb') as f:
        pickle.dump(bpo_tm_vec_go_terms, f)
    
    with open('CAFA_5_CCO_tm_vec_go_terms.pkl', 'wb') as f:
        pickle.dump(cco_tm_vec_go_terms, f)
    
    print("Saved TM-Vec and GO terms tuples")


def load_multilabel_binarizers():
    """Load MultiLabelBinarizer objects for filtering terms."""
    print("Loading MultiLabelBinarizer objects...")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        with open('function_mlb.pkl', 'rb') as f:
            function_mlb = pickle.load(f)
        
        with open('process_mlb.pkl', 'rb') as f:
            process_mlb = pickle.load(f)
        
        with open('component_mlb.pkl', 'rb') as f:
            component_mlb = pickle.load(f)
    
    print(f"Function MLB classes: {len(function_mlb.classes_)}")
    print(f"Process MLB classes: {len(process_mlb.classes_)}")
    print(f"Component MLB classes: {len(component_mlb.classes_)}")
    
    return function_mlb, process_mlb, component_mlb


def filter_protein_data_by_vocabulary(mfo_data, bpo_data, cco_data, function_mlb, process_mlb, component_mlb):
    """Filter protein data to only include terms present in the vocabulary."""
    print("Filtering protein data by vocabulary...")
    
    print(f"Original MFO protein data size: {len(mfo_data)}")
    mfo_filtered = []
    for data in tqdm(mfo_data, desc="Filtering MFO"):
        filtered_terms = [term for term in data.mfo_terms if term in function_mlb.classes_]
        if filtered_terms:
            mfo_filtered.append(ProteinData(
                seq_id=data.seq_id,
                tm_vec=data.tm_vec,
                protT5=data.protT5,
                sequence=data.sequence,
                mfo_terms=filtered_terms,
                bpo_terms=data.bpo_terms,
                cco_terms=data.cco_terms
            ))
    print(f"Filtered MFO protein data size: {len(mfo_filtered)}")
    
    print(f"Original BPO protein data size: {len(bpo_data)}")
    bpo_filtered = []
    for data in tqdm(bpo_data, desc="Filtering BPO"):
        filtered_terms = [term for term in data.bpo_terms if term in process_mlb.classes_]
        if filtered_terms:
            bpo_filtered.append(ProteinData(
                seq_id=data.seq_id,
                tm_vec=data.tm_vec,
                protT5=data.protT5,
                sequence=data.sequence,
                mfo_terms=data.mfo_terms,
                bpo_terms=filtered_terms,
                cco_terms=data.cco_terms
            ))
    print(f"Filtered BPO protein data size: {len(bpo_filtered)}")
    
    print(f"Original CCO protein data size: {len(cco_data)}")
    cco_filtered = []
    for data in tqdm(cco_data, desc="Filtering CCO"):
        filtered_terms = [term for term in data.cco_terms if term in component_mlb.classes_]
        if filtered_terms:
            cco_filtered.append(ProteinData(
                seq_id=data.seq_id,
                tm_vec=data.tm_vec,
                protT5=data.protT5,
                sequence=data.sequence,
                mfo_terms=data.mfo_terms,
                bpo_terms=data.bpo_terms,
                cco_terms=filtered_terms
            ))
    print(f"Filtered CCO protein data size: {len(cco_filtered)}")
    
    return mfo_filtered, bpo_filtered, cco_filtered


def export_filtered_sequences(mfo_filtered, bpo_filtered, cco_filtered, sequences):
    """Export filtered sequences to FASTA files."""
    print("Exporting filtered sequences...")
    
    sequences_dict = {seq.id: str(seq.seq) for seq in sequences}
    
    mfo_ids = [data.seq_id for data in mfo_filtered]
    bpo_ids = [data.seq_id for data in bpo_filtered]
    cco_ids = [data.seq_id for data in cco_filtered]
    
    mfo_seq_records = [SeqRecord(Seq(sequences_dict[seq_id]), id=seq_id) for seq_id in mfo_ids]
    bpo_seq_records = [SeqRecord(Seq(sequences_dict[seq_id]), id=seq_id) for seq_id in bpo_ids]
    cco_seq_records = [SeqRecord(Seq(sequences_dict[seq_id]), id=seq_id) for seq_id in cco_ids]
    
    assert len(mfo_seq_records) == len(mfo_filtered), "MFO sequences and data length mismatch"
    assert len(bpo_seq_records) == len(bpo_filtered), "BPO sequences and data length mismatch"
    assert len(cco_seq_records) == len(cco_filtered), "CCO sequences and data length mismatch"
    

    SeqIO.write(mfo_seq_records, 'CAFA_5_MFO_filtered_sequences.fasta', 'fasta')
    SeqIO.write(bpo_seq_records, 'CAFA_5_BPO_filtered_sequences.fasta', 'fasta')
    SeqIO.write(cco_seq_records, 'CAFA_5_CCO_filtered_sequences.fasta', 'fasta')
    
    print("Exported filtered sequences to FASTA files")


def main():
    """Main preprocessing pipeline."""
    print("=== CAFA Data Preprocessing Pipeline ===")
    
    # Check input files exist
    required_files = [
        'CAFA_5_train_sequences.fasta',
        'CAFA_train_TMVec.pkl',
        'CAFA_5_train_terms.tsv',
        'function_mlb.pkl',
        'process_mlb.pkl',
        'component_mlb.pkl'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"ERROR: Required file not found: {file_path}")
            return
    
    sequences = load_sequences('CAFA_5_train_sequences.fasta')
    
    embeddings = load_embeddings('CAFA_train_TMVec.pkl')
    
    mfo_mapping, bpo_mapping, cco_mapping = load_go_terms('CAFA_5_train_terms.tsv')
    
    protein_data_list = create_protein_data_list(sequences, embeddings, mfo_mapping, bpo_mapping, cco_mapping)
    
    print("Saving protein data list...")
    with open('CAFA_5_all_data_proteinData.pkl', 'wb') as f:
        pickle.dump(protein_data_list, f)
    print("Saved CAFA_5_all_data_proteinData.pkl")
    
    count_terms_by_aspect(protein_data_list)
    
    mfo_data, bpo_data, cco_data = filter_by_aspect(protein_data_list)
    
    save_tm_vec_go_terms(mfo_data, bpo_data, cco_data)
    
    function_mlb, process_mlb, component_mlb = load_multilabel_binarizers()
    
    mfo_filtered, bpo_filtered, cco_filtered = filter_protein_data_by_vocabulary(
        mfo_data, bpo_data, cco_data, function_mlb, process_mlb, component_mlb
    )
    
    export_filtered_sequences(mfo_filtered, bpo_filtered, cco_filtered, sequences)
    
    print("\n=== Preprocessing Complete ===")
    print("Output files generated:")
    print("- CAFA_5_all_data_proteinData.pkl (compatible with train_cnn_cafa.py)")
    print("- CAFA_5_MFO_tm_vec_go_terms.pkl")
    print("- CAFA_5_BPO_tm_vec_go_terms.pkl")
    print("- CAFA_5_CCO_tm_vec_go_terms.pkl")
    print("- CAFA_5_MFO_filtered_sequences.fasta")
    print("- CAFA_5_BPO_filtered_sequences.fasta")
    print("- CAFA_5_CCO_filtered_sequences.fasta")


if __name__ == "__main__":
    main() 