#!/bin/bash

#SBATCH --job-name=blast_db_creation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=blast_db_creation_%j.out
#SBATCH --error=blast_db_creation_%j.err

## add email to sbatch
#SBATCH --mail-user=hs3434@nyu.edu
#SBATCH --mail-type=ALL

# Load required modules (adjust these based on your HPC environment)
# module load blast+/2.13.0
# module load python/3.9

# Script parameters - adjust these paths as needed
DATA_DIR="${SCRATCH}/for_blast_and_foldseek"  # Directory containing FASTA files to convert
OUTPUT_DIR="blast_databases"
MAKEBLASTDB_BIN="makeblastdb"  # Assuming BLAST+ is in PATH, otherwise specify full path

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Function to create BLAST database from FASTA
create_blast_db() {
    local fasta_file=$1
    local db_name=$2
    
    echo "Creating BLAST database for ${fasta_file}..."
    
    # Build BLAST protein database
    MAKEDB_CMD="${MAKEBLASTDB_BIN} -in ${fasta_file} -dbtype prot -out ${OUTPUT_DIR}/${db_name} -title \"${db_name}\" -parse_seqids"
    
    # Execute database creation
    ${MAKEDB_CMD}
    
    if [ $? -eq 0 ]; then
        echo "Successfully created BLAST database: ${db_name}"
        
        # Get database statistics
        echo "Database statistics for ${db_name}:"
        ${MAKEBLASTDB_BIN} -in ${fasta_file} -dbtype prot -out ${OUTPUT_DIR}/${db_name} -title "${db_name}" -parse_seqids -blastdb_version 5 > /dev/null 2>&1
        
        # Show database info if blastdbcmd is available
        if command -v blastdbcmd &> /dev/null; then
            echo "Number of sequences: $(blastdbcmd -db ${OUTPUT_DIR}/${db_name} -info | grep 'sequences' | awk '{print $1}')"
            echo "Total length: $(blastdbcmd -db ${OUTPUT_DIR}/${db_name} -info | grep 'total length' | awk '{print $4}')"
        fi
    else
        echo "Error: Failed to create BLAST database for ${fasta_file}"
        return 1
    fi
}

# Check if makeblastdb is available
if ! command -v ${MAKEBLASTDB_BIN} &> /dev/null; then
    echo "Error: makeblastdb not found in PATH or at specified location"
    echo "Please ensure BLAST+ is installed and available."
    echo "You may need to load the appropriate module:"
    echo "  module load blast+ or module load ncbi-blast+"
    exit 1
fi

# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: Data directory ${DATA_DIR} not found."
    exit 1
fi

echo "Starting BLAST database creation..."
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "makeblastdb binary: ${MAKEBLASTDB_BIN}"
echo "Number of CPU threads: ${SLURM_CPUS_PER_TASK}"

# Create databases for the curated FASTA files
echo "Processing curated FASTA files from ${DATA_DIR}..."

# Counter for successful database creations
successful_dbs=0
total_dbs=0

# List of FASTA files to process (same as Foldseek script)
declare -A fasta_files=(
    ["process_train.fasta"]="process_train_db"
    ["process_test.fasta"]="process_test_db"
    ["function_train.fasta"]="function_train_db"
    ["function_test.fasta"]="function_test_db"
    ["component_train.fasta"]="component_train_db"
    ["component_test.fasta"]="component_test_db"
)

# Process each FASTA file
for fasta_file in "${!fasta_files[@]}"; do
    db_name="${fasta_files[$fasta_file]}"
    fasta_path="${DATA_DIR}/${fasta_file}"
    
    if [ -f "${fasta_path}" ]; then
        ((total_dbs++))
        echo "=== Processing ${fasta_file} ==="
        if create_blast_db "${fasta_path}" "${db_name}"; then
            ((successful_dbs++))
        fi
    else
        echo "Warning: ${fasta_path} not found, skipping..."
    fi
done

echo "=== BLAST database creation completed! ==="
echo "Successfully created ${successful_dbs} out of ${total_dbs} databases"
echo "Output databases are located in: ${OUTPUT_DIR}/"

# List created databases
echo "Created database files:"
ls -la ${OUTPUT_DIR}/

# Create a summary file
cat > ${OUTPUT_DIR}/database_info.txt << EOF
BLAST Database Creation Summary
===============================

Job ID: ${SLURM_JOB_ID}
Date: $(date)
Node: ${SLURM_JOB_NODELIST}

Input Data Directory: ${DATA_DIR}
Output Directory: ${OUTPUT_DIR}
makeblastdb Binary: ${MAKEBLASTDB_BIN}

Successfully Created: ${successful_dbs}/${total_dbs} databases

Created Databases:
$(ls -1 ${OUTPUT_DIR}/ | grep -v "database_info.txt")

Target FASTA Files Processed:
- process_train.fasta → process_train_db
- process_test.fasta → process_test_db
- function_train.fasta → function_train_db
- function_test.fasta → function_test_db
- component_train.fasta → component_train_db
- component_test.fasta → component_test_db

Usage Instructions:
==================

For protein BLAST searches:
blastp -query query.fasta -db ${OUTPUT_DIR}/database_name -out results.txt

Example searches:
blastp -query test_proteins.fasta -db ${OUTPUT_DIR}/process_train_db -out process_results.txt -outfmt 6
blastp -query test_proteins.fasta -db ${OUTPUT_DIR}/function_train_db -out function_results.txt -outfmt 6 -evalue 1e-5
blastp -query test_proteins.fasta -db ${OUTPUT_DIR}/component_train_db -out component_results.txt -outfmt 6 -max_target_seqs 10

Common BLAST+ output formats:
- outfmt 0: Pairwise alignments
- outfmt 6: Tabular format (default)
- outfmt 7: Tabular with comment lines
- outfmt 8: Tabular (custom format)
- outfmt 10: Comma-separated values
- outfmt 11: BLAST archive format

For parallel BLAST searches:
blastp -query query.fasta -db database -out results.txt -num_threads ${SLURM_CPUS_PER_TASK}

Database Information:
$(if command -v blastdbcmd &> /dev/null; then
    for db_file in "${!fasta_files[@]}"; do
        db_name="${fasta_files[$db_file]}"
        if [ -f "${OUTPUT_DIR}/${db_name}.phr" ]; then
            echo "- ${db_name}:"
            blastdbcmd -db ${OUTPUT_DIR}/${db_name} -info 2>/dev/null | head -5 | sed 's/^/  /'
        fi
    done
else
    echo "- blastdbcmd not available for detailed database info"
fi)
EOF

echo "Database creation summary saved to: ${OUTPUT_DIR}/database_info.txt"

# Verify all expected database files were created
echo "=== Verification ==="
missing_files=0
for fasta_file in "${!fasta_files[@]}"; do
    db_name="${fasta_files[$fasta_file]}"
    # Check for essential BLAST database files
    if [ ! -f "${OUTPUT_DIR}/${db_name}.phr" ] || [ ! -f "${OUTPUT_DIR}/${db_name}.pin" ] || [ ! -f "${OUTPUT_DIR}/${db_name}.psq" ]; then
        echo "Warning: Incomplete database files for ${db_name}"
        ((missing_files++))
    else
        echo "✓ Complete database files for ${db_name}"
    fi
done

if [ ${missing_files} -eq 0 ]; then
    echo "✓ All databases created successfully!"
else
    echo "⚠ ${missing_files} databases have missing files"
fi

echo "BLAST database creation script completed."
