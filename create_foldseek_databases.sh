#!/bin/bash

#SBATCH --job-name=foldseek_db_creation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=24:00:00
#SBATCH --output=foldseek_db_creation_%j.out
#SBATCH --error=foldseek_db_creation_%j.err

## add email to sbatch
#SBATCH --mail-user=hs3434@nyu.edu
#SBATCH --mail-type=ALL

# Note: Request A100 or H100 for optimal GPU performance
# Alternative SLURM GPU options:
# #SBATCH --gres=gpu:H100:1     # Best performance
# #SBATCH --gres=gpu:A100:1     # Good performance  
# #SBATCH --gres=gpu:RTX8000:1  # Reduced performance
# #SBATCH --gres=gpu:V100:1     # GPU search NOT supported, CPU only

# Load required modules (adjust these based on your HPC environment)
# module load cuda/11.8
# module load gcc/9.3.0
# module load python/3.9

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Script parameters - adjust these paths as needed
DATA_DIR="${SCRATCH}/for_blast_and_foldseek"  # Directory containing FASTA files to convert
OUTPUT_DIR="foldseek_databases"
PROSTT5_WEIGHTS_DIR="${SCRATCH}/weights"  # ProstT5 weights in SCRATCH directory
FOLDSEEK_BIN="${SCRATCH}/foldseek/bin/foldseek"  # Path to foldseek binary

# Detect GPU capabilities for Foldseek
detect_gpu_support() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        echo "Detected GPU: ${GPU_NAME}"
        
        # Check GPU generation for Foldseek compatibility
        if [[ "${GPU_NAME}" =~ "H100" ]]; then
            echo "H100 detected - GPU search fully supported"
            USE_GPU=true
            CREATE_PADDED=true
        elif [[ "${GPU_NAME}" =~ "A100" ]]; then
            echo "A100 detected - GPU search fully supported"
            USE_GPU=true
            CREATE_PADDED=true
        elif [[ "${GPU_NAME}" =~ "RTX.*8000" ]] || [[ "${GPU_NAME}" =~ "RTX.*6000" ]]; then
            echo "RTX8000/6000 (Turing) detected - GPU search supported with reduced performance"
            USE_GPU=true
            CREATE_PADDED=true
        elif [[ "${GPU_NAME}" =~ "V100" ]]; then
            echo "V100 (Volta) detected - GPU search NOT supported, using CPU only"
            USE_GPU=false
            CREATE_PADDED=false
        else
            echo "Unknown GPU: ${GPU_NAME} - defaulting to CPU only"
            USE_GPU=false
            CREATE_PADDED=false
        fi
    else
        echo "No nvidia-smi found - using CPU only"
        USE_GPU=false
        CREATE_PADDED=false
    fi
}

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Check if ProstT5 weights exist
if [ ! -d "${PROSTT5_WEIGHTS_DIR}" ]; then
    echo "Error: ProstT5 weights directory not found: ${PROSTT5_WEIGHTS_DIR}"
    echo "Please ensure ProstT5 weights are downloaded to ${SCRATCH}/weights/"
    exit 1
fi

# Function to create Foldseek database from FASTA
create_foldseek_db() {
    local fasta_file=$1
    local db_name=$2
    local weights_dir=$3
    
    echo "Creating Foldseek database for ${fasta_file}..."
    
    # Build command based on GPU support
    if [ "$USE_GPU" = true ]; then
        echo "Using GPU acceleration for database creation..."
        CREATE_CMD="${FOLDSEEK_BIN} createdb ${fasta_file} ${OUTPUT_DIR}/${db_name} --prostt5-model ${weights_dir} --gpu 1 --threads ${SLURM_CPUS_PER_TASK}"
    else
        echo "Using CPU-only for database creation..."
        CREATE_CMD="${FOLDSEEK_BIN} createdb ${fasta_file} ${OUTPUT_DIR}/${db_name} --prostt5-model ${weights_dir} --threads ${SLURM_CPUS_PER_TASK}"
    fi
    
    # Execute database creation
    ${CREATE_CMD}
    
    if [ $? -eq 0 ]; then
        echo "Successfully created database: ${db_name}"
        
        # Create padded version only if GPU is supported
        if [ "$CREATE_PADDED" = true ]; then
            echo "Creating padded database for GPU search..."
            ${FOLDSEEK_BIN} makepaddedseqdb ${OUTPUT_DIR}/${db_name} ${OUTPUT_DIR}/${db_name}_pad
            
            if [ $? -eq 0 ]; then
                echo "Successfully created padded database: ${db_name}_pad"
            else
                echo "Error: Failed to create padded database for ${db_name}"
                return 1
            fi
        else
            echo "Skipping padded database creation (not needed for CPU-only searches)"
        fi
    else
        echo "Error: Failed to create database for ${fasta_file}"
        return 1
    fi
}

# Check if foldseek is available
if [ ! -f "${FOLDSEEK_BIN}" ]; then
    echo "Error: foldseek binary not found at ${FOLDSEEK_BIN}"
    echo "Please ensure foldseek is installed and the path is correct."
    exit 1
fi

# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: Data directory ${DATA_DIR} not found."
    exit 1
fi

# Detect GPU capabilities
detect_gpu_support

echo "Starting Foldseek database creation..."
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "ProstT5 weights directory: ${PROSTT5_WEIGHTS_DIR}"
echo "Foldseek binary: ${FOLDSEEK_BIN}"
echo "GPU device: ${CUDA_VISIBLE_DEVICES}"
echo "Number of CPU threads: ${SLURM_CPUS_PER_TASK}"
echo "GPU acceleration: ${USE_GPU}"
echo "Create padded databases: ${CREATE_PADDED}"

# Create databases for the curated FASTA files (with padding for GPU searches)
echo "Processing curated FASTA files from ${DATA_DIR}..."

# Counter for successful database creations
successful_dbs=0
total_dbs=0

# List of FASTA files to process (based on what we found in the directory)
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
        if create_foldseek_db "${fasta_path}" "${db_name}" "${PROSTT5_WEIGHTS_DIR}"; then
            ((successful_dbs++))
        fi
    else
        echo "Warning: ${fasta_path} not found, skipping..."
    fi
done

echo "=== Foldseek database creation completed! ==="
echo "Successfully created ${successful_dbs} out of ${total_dbs} databases"
echo "Output databases are located in: ${OUTPUT_DIR}/"

# List created databases
echo "Created databases:"
ls -la ${OUTPUT_DIR}/

# Create a summary file
cat > ${OUTPUT_DIR}/database_info.txt << EOF
Foldseek Database Creation Summary
================================

Job ID: ${SLURM_JOB_ID}
Date: $(date)
Node: ${SLURM_JOB_NODELIST}
GPU: ${CUDA_VISIBLE_DEVICES}

Input Data Directory: ${DATA_DIR}
Output Directory: ${OUTPUT_DIR}
ProstT5 Weights: ${PROSTT5_WEIGHTS_DIR}
Foldseek Binary: ${FOLDSEEK_BIN}

Successfully Created: ${successful_dbs}/${total_dbs} databases

Created Databases:
$(ls -1 ${OUTPUT_DIR}/ | grep -v "database_info.txt")

GPU Configuration:
- GPU Used: ${USE_GPU}
- Padded Databases Created: ${CREATE_PADDED}
- GPU Name: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "N/A")

Target FASTA Files Processed:
$(if [ "$CREATE_PADDED" = true ]; then
echo "- process_train.fasta → process_train_db + process_train_db_pad"
echo "- process_test.fasta → process_test_db + process_test_db_pad"
echo "- function_train.fasta → function_train_db + function_train_db_pad"
echo "- function_test.fasta → function_test_db + function_test_db_pad"
echo "- component_train.fasta → component_train_db + component_train_db_pad"
echo "- component_test.fasta → component_test_db + component_test_db_pad"
else
echo "- process_train.fasta → process_train_db (CPU only)"
echo "- process_test.fasta → process_test_db (CPU only)"
echo "- function_train.fasta → function_train_db (CPU only)"
echo "- function_test.fasta → function_test_db (CPU only)"
echo "- component_train.fasta → component_train_db (CPU only)"
echo "- component_test.fasta → component_test_db (CPU only)"
fi)

Usage Instructions:
==================

For CPU searches:
${FOLDSEEK_BIN} search query_db target_db result_dir --threads ${SLURM_CPUS_PER_TASK}

For GPU searches (using padded databases):
${FOLDSEEK_BIN} search query_db target_db_pad result_dir --gpu 1

Example GPU search:
CUDA_VISIBLE_DEVICES=0 ${FOLDSEEK_BIN} search process_train_db process_train_db_pad results --gpu 1

Note: These databases contain predicted 3Di structural sequences without 
additional structural details. They support monomer search and clustering 
but do not enable features requiring Cα information (--alignment-type 1, 
TM-score, or LDDT output).
EOF

echo "Database creation summary saved to: ${OUTPUT_DIR}/database_info.txt"
