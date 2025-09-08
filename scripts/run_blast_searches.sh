#!/bin/bash

#SBATCH --job-name=blast_search
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=blast_search_%j.out
#SBATCH --error=blast_search_%j.err

## add email to sbatch
#SBATCH --mail-user=hs3434@nyu.edu
#SBATCH --mail-type=ALL

# Load required modules (adjust these based on your HPC environment)
module load blast+/2.15.0
# module load python/intel/3.8.6

# Script parameters
BLASTP_BIN="blastp"  # Assuming BLAST+ is in PATH, otherwise specify full path
DB_DIR="${SCRATCH}/blast_databases"  # Training/test databases location
SWISSPROT_DB="${SCRATCH}/swissprot_blast_db"  # SwissProt BLAST database
OUTPUT_DIR="blast_search_results"
MAX_TARGET_SEQS=100
EVALUE="1e-5"
NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Create output directory structure
mkdir -p ${OUTPUT_DIR}/{swissprot_searches,training_searches}

# Function to run a single BLAST search with error handling
run_blast_search() {
    local query_file=$1
    local target_db=$2
    local output_file=$3
    local search_name=$4
    
    echo "=== Starting BLAST search: ${search_name} ==="
    echo "Query file: ${query_file}"
    echo "Target DB: ${target_db}"
    echo "Output: ${output_file}"
    echo "E-value: ${EVALUE}"
    echo "Max target seqs: ${MAX_TARGET_SEQS}"
    echo "Threads: ${NUM_THREADS}"
    
    # Run BLASTp search
    ${BLASTP_BIN} \
        -query ${query_file} \
        -db ${target_db} \
        -out ${output_file} \
        -outfmt "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen" \
        -evalue ${EVALUE} \
        -max_target_seqs ${MAX_TARGET_SEQS} \
        -num_threads ${NUM_THREADS}
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ Successfully completed: ${search_name}"
        # Check if output file exists and has content
        if [ -s "${output_file}" ]; then
            local line_count=$(wc -l < "${output_file}")
            echo "  Output file contains ${line_count} lines"
            
            # Get some basic statistics
            if [ $line_count -gt 0 ]; then
                local unique_queries=$(cut -f1 "${output_file}" | sort -u | wc -l)
                local unique_targets=$(cut -f2 "${output_file}" | sort -u | wc -l)
                echo "  Unique query sequences: ${unique_queries}"
                echo "  Unique target sequences: ${unique_targets}"
            fi
        else
            echo "  WARNING: Output file is empty or missing"
            return 1
        fi
    else
        echo "✗ Failed: ${search_name} (exit code: ${exit_code})"
        return $exit_code
    fi
    
    echo ""
}

# Function to extract FASTA from BLAST database for query
extract_fasta_from_db() {
    local blast_db=$1
    local output_fasta=$2
    local db_name=$3
    
    echo "Extracting FASTA sequences from ${db_name} database..."
    
    if command -v blastdbcmd &> /dev/null; then
        blastdbcmd -db ${blast_db} -entry all -outfmt "%f" -out ${output_fasta}
        
        if [ $? -eq 0 ] && [ -s "${output_fasta}" ]; then
            local seq_count=$(grep -c "^>" "${output_fasta}")
            echo "✓ Extracted ${seq_count} sequences to ${output_fasta}"
            return 0
        else
            echo "✗ Failed to extract sequences from ${blast_db}"
            return 1
        fi
    else
        echo "✗ blastdbcmd not available - cannot extract FASTA from database"
        return 1
    fi
}

# Check if BLASTp binary exists
if ! command -v ${BLASTP_BIN} &> /dev/null; then
    echo "Error: BLASTp binary not found: ${BLASTP_BIN}"
    echo "Please ensure BLAST+ is installed and available in PATH"
    echo "You may need to load the appropriate module:"
    echo "  module load blast+ or module load ncbi-blast+"
    exit 1
fi

# Check if SwissProt database exists
if [ ! -f "${SWISSPROT_DB}.phr" ]; then
    echo "Error: SwissProt BLAST database not found at ${SWISSPROT_DB}"
    echo "Expected database files: ${SWISSPROT_DB}.phr, .pin, .psq"
    echo "Please ensure the SwissProt BLAST database is created and available"
    exit 1
fi

echo "Starting BLASTp searches..."
echo "BLASTp binary: ${BLASTP_BIN}"
echo "Database directory: ${DB_DIR}"
echo "SwissProt database: ${SWISSPROT_DB}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Max target sequences per search: ${MAX_TARGET_SEQS}"
echo "E-value threshold: ${EVALUE}"
echo "CPU threads: ${NUM_THREADS}"
echo ""

# Define ontologies and their corresponding database names
declare -A ontologies=(
    ["process"]="process"
    ["function"]="function"  
    ["component"]="component"
)

# Track search success/failure
successful_searches=0
total_searches=0

# Create temporary directory for extracted FASTA files
TEMP_FASTA_DIR="${OUTPUT_DIR}/temp_fasta"
mkdir -p ${TEMP_FASTA_DIR}

# Run searches for each ontology
for ontology in "${!ontologies[@]}"; do
    db_prefix="${ontologies[$ontology]}"
    
    # Define database paths
    test_db="${DB_DIR}/${db_prefix}_test_db"
    train_db="${DB_DIR}/${db_prefix}_train_db"
    
    # Check if test and train databases exist
    if [ ! -f "${test_db}.phr" ]; then
        echo "Error: Test database not found: ${test_db}"
        echo "Expected files: ${test_db}.phr, .pin, .psq"
        continue
    fi
    
    if [ ! -f "${train_db}.phr" ]; then
        echo "Error: Training database not found: ${train_db}"
        echo "Expected files: ${train_db}.phr, .pin, .psq"
        continue
    fi
    
    echo "Processing ontology: ${ontology}"
    echo "Test DB: ${test_db}"
    echo "Train DB: ${train_db}"
    echo ""
    
    # Extract test sequences to FASTA for use as queries
    test_fasta="${TEMP_FASTA_DIR}/${ontology}_test.fasta"
    if extract_fasta_from_db "${test_db}" "${test_fasta}" "${ontology} test"; then
        
        # Search 1: Test vs SwissProt
        ((total_searches++))
        if run_blast_search \
            "${test_fasta}" \
            "${SWISSPROT_DB}" \
            "${OUTPUT_DIR}/swissprot_searches/${ontology}_vs_swissprot.tsv" \
            "${ontology} test vs SwissProt"; then
            ((successful_searches++))
        fi
        
        # Search 2: Test vs Training
        ((total_searches++))
        if run_blast_search \
            "${test_fasta}" \
            "${train_db}" \
            "${OUTPUT_DIR}/training_searches/${ontology}_vs_training.tsv" \
            "${ontology} test vs training"; then
            ((successful_searches++))
        fi
    else
        echo "Skipping searches for ${ontology} due to FASTA extraction failure"
        total_searches=$((total_searches + 2))  # Count the skipped searches
    fi
done

echo "=== BLASTp searches completed! ==="
echo "Successfully completed ${successful_searches} out of ${total_searches} searches"
echo ""

# List output files
echo "Created search result files:"
find ${OUTPUT_DIR} -name "*.tsv" -type f | sort

# Create summary report
cat > ${OUTPUT_DIR}/search_summary.txt << EOF
BLASTp Search Summary
=====================

Job ID: ${SLURM_JOB_ID}
Date: $(date)
Node: ${SLURM_JOB_NODELIST}

Configuration:
- BLASTp Binary: ${BLASTP_BIN}
- Database Directory: ${DB_DIR}
- SwissProt Database: ${SWISSPROT_DB}
- E-value Threshold: ${EVALUE}
- Max Target Sequences: ${MAX_TARGET_SEQS}
- CPU Threads: ${NUM_THREADS}

Results:
- Successful Searches: ${successful_searches}/${total_searches}
- Output Directory: ${OUTPUT_DIR}

Search Results:
$(find ${OUTPUT_DIR} -name "*.tsv" -type f -exec sh -c 'echo "- $(basename "$1"): $(wc -l < "$1") hits"' _ {} \;)

Output Format:
The output files use BLAST tabular format (outfmt 6) with the following columns:
1. qseqid    - Query sequence ID
2. sseqid    - Subject (target) sequence ID  
3. pident    - Percentage of identical matches
4. length    - Alignment length
5. mismatch  - Number of mismatches
6. gapopen   - Number of gap openings
7. qstart    - Start position in query
8. qend      - End position in query
9. sstart    - Start position in subject
10. send     - End position in subject
11. evalue   - Expectation value
12. bitscore - Bit score
13. qlen     - Query sequence length
14. slen     - Subject sequence length

Analysis Commands:
==================

To analyze results, you can use commands like:

# Count unique query sequences with hits
cut -f1 ${OUTPUT_DIR}/swissprot_searches/process_vs_swissprot.tsv | sort -u | wc -l

# Get top hits by bit score
sort -k12,12nr ${OUTPUT_DIR}/swissprot_searches/process_vs_swissprot.tsv | head -10

# Filter by identity percentage (e.g., >30%)
awk '\$3 > 30' ${OUTPUT_DIR}/swissprot_searches/process_vs_swissprot.tsv

# Get distribution of E-values
cut -f11 ${OUTPUT_DIR}/swissprot_searches/process_vs_swissprot.tsv | sort -g | uniq -c

System Information:
$(if command -v blastp &> /dev/null; then blastp -version | head -1; else echo "BLASTp version: N/A"; fi)
$(free -h | grep "Mem:")
$(nproc) CPU cores available
EOF

echo "Search summary saved to: ${OUTPUT_DIR}/search_summary.txt"

# Clean up temporary FASTA files
echo "Cleaning up temporary files..."
rm -rf ${TEMP_FASTA_DIR}

# Final status
if [ $successful_searches -eq $total_searches ]; then
    echo "🎉 All BLAST searches completed successfully!"
    exit 0
else
    echo "⚠️ Some searches failed. Check logs for details."
    exit 1
fi
