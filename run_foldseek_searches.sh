#!/bin/bash

#SBATCH --job-name=foldseek_search
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=foldseek_search_%j.out
#SBATCH --error=foldseek_search_%j.err

## add email to sbatch
#SBATCH --mail-user=hs3434@nyu.edu
#SBATCH --mail-type=ALL

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Script parameters
FOLDSEEK_BIN="${SCRATCH}/foldseek/bin/foldseek"
DB_DIR="${SCRATCH}/foldseek_databases"  # Training/test databases location
SWISSPROT_DB="${SCRATCH}/afdb_swissprot_pad"  # SwissProt padded database
OUTPUT_DIR="foldseek_search_results"
MAX_SEQS=100

# Create output directory structure
mkdir -p ${OUTPUT_DIR}/{swissprot_searches,training_searches}

# Function to run a single search with error handling
run_search() {
    local query_db=$1
    local target_db=$2
    local output_file=$3
    local temp_dir=$4
    local search_name=$5
    
    echo "=== Starting search: ${search_name} ==="
    echo "Query DB: ${query_db}"
    echo "Target DB: ${target_db}"
    echo "Output: ${output_file}"
    echo "Temp dir: ${temp_dir}"
    
    # Create temp directory
    mkdir -p ${temp_dir}
    
    # Run FoldSeek search
    ${FOLDSEEK_BIN} easy-search \
        ${query_db} \
        ${target_db} \
        ${output_file} \
        ${temp_dir} \
        --max-seqs ${MAX_SEQS} \
        --gpu 1 \
        --threads ${SLURM_CPUS_PER_TASK}
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ Successfully completed: ${search_name}"
        # Check if output file exists and has content
        if [ -s "${output_file}" ]; then
            local line_count=$(wc -l < "${output_file}")
            echo "  Output file contains ${line_count} lines"
        else
            echo "  WARNING: Output file is empty or missing"
            return 1
        fi
    else
        echo "✗ Failed: ${search_name} (exit code: ${exit_code})"
        return $exit_code
    fi
    
    # Clean up temp directory for this search
    rm -rf ${temp_dir}
    echo ""
}

# Check if FoldSeek binary exists
if [ ! -f "${FOLDSEEK_BIN}" ]; then
    echo "Error: FoldSeek binary not found at ${FOLDSEEK_BIN}"
    exit 1
fi

# Check if SwissProt database exists
if [ ! -f "${SWISSPROT_DB}" ]; then
    echo "Error: SwissProt database not found at ${SWISSPROT_DB}"
    echo "Expected padded database: ${SWISSPROT_DB}"
    exit 1
fi

echo "Starting FoldSeek searches..."
echo "FoldSeek binary: ${FOLDSEEK_BIN}"
echo "Database directory: ${DB_DIR}"
echo "SwissProt database: ${SWISSPROT_DB}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Max sequences per search: ${MAX_SEQS}"
echo "GPU device: ${CUDA_VISIBLE_DEVICES}"
echo "CPU threads: ${SLURM_CPUS_PER_TASK}"
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

# Run searches for each ontology
for ontology in "${!ontologies[@]}"; do
    db_prefix="${ontologies[$ontology]}"
    
    # Define database paths
    test_db="${DB_DIR}/${db_prefix}_test_db_pad"
    train_db="${DB_DIR}/${db_prefix}_train_db_pad"
    
    # Check if test and train databases exist
    if [ ! -f "${test_db}" ]; then
        echo "Error: Test database not found: ${test_db}"
        continue
    fi
    
    if [ ! -f "${train_db}" ]; then
        echo "Error: Training database not found: ${train_db}"
        continue
    fi
    
    echo "Processing ontology: ${ontology}"
    echo "Test DB: ${test_db}"
    echo "Train DB: ${train_db}"
    echo ""
    
    # Search 1: Test vs SwissProt
    ((total_searches++))
    if run_search \
        "${test_db}" \
        "${SWISSPROT_DB}" \
        "${OUTPUT_DIR}/swissprot_searches/${ontology}_vs_swissprot.tsv" \
        "${OUTPUT_DIR}/tmp_${ontology}_swissprot" \
        "${ontology} test vs SwissProt"; then
        ((successful_searches++))
    fi
    
    # Search 2: Test vs Training
    ((total_searches++))
    if run_search \
        "${test_db}" \
        "${train_db}" \
        "${OUTPUT_DIR}/training_searches/${ontology}_vs_training.tsv" \
        "${OUTPUT_DIR}/tmp_${ontology}_training" \
        "${ontology} test vs training"; then
        ((successful_searches++))
    fi
done

echo "=== FoldSeek searches completed! ==="
echo "Successfully completed ${successful_searches} out of ${total_searches} searches"
echo ""

# List output files
echo "Created search result files:"
find ${OUTPUT_DIR} -name "*.tsv" -type f | sort

# Create summary report
cat > ${OUTPUT_DIR}/search_summary.txt << EOF
FoldSeek Search Summary
======================

Job ID: ${SLURM_JOB_ID}
Date: $(date)
Node: ${SLURM_JOB_NODELIST}
GPU: ${CUDA_VISIBLE_DEVICES}

Configuration:
- FoldSeek Binary: ${FOLDSEEK_BIN}
- Database Directory: ${DB_DIR}
- SwissProt Database: ${SWISSPROT_DB}
- Max Sequences: ${MAX_SEQS}
- CPU Threads: ${SLURM_CPUS_PER_TASK}

Results:
- Successful Searches: ${successful_searches}/${total_searches}
- Output Directory: ${OUTPUT_DIR}

Search Results:
$(find ${OUTPUT_DIR} -name "*.tsv" -type f -exec sh -c 'echo "- $(basename "$1"): $(wc -l < "$1") lines"' _ {} \;)

GPU Information:
$(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
EOF

echo "Search summary saved to: ${OUTPUT_DIR}/search_summary.txt"

# Final status
if [ $successful_searches -eq $total_searches ]; then
    echo "🎉 All searches completed successfully!"
    exit 0
else
    echo "⚠️ Some searches failed. Check logs for details."
    exit 1
fi
