#!/bin/bash

# Function to display usage instructions
usage() {
    echo "Usage: $0 -t <test_dataset> -l <leftout_dataset> -r <train_dataset> -m <model_path> [-c <code_path>]"
    echo "  -t: Test dataset TSV file (required)"
    echo "  -l: Leftout dataset TSV file (required)"
    echo "  -r: Train dataset TSV file (required)"
    echo "  -m: Model path (required)"
    echo "  -c: Code path (optional, default: ../../code/machine_learning)"
    exit 1
}

# Default code path
CODE="../../code/machine_learning"

# Parse command-line arguments
while getopts ":t:l:r:m:c:" opt; do
    case ${opt} in
        t )
            TEST_DATASET=$OPTARG
            ;;
        l )
            LEFTOUT_DATASET=$OPTARG
            ;;
        r )
            TRAIN_DATASET=$OPTARG
            ;;
        m )
            MODEL=$OPTARG
            ;;
        c )
            CODE=$OPTARG
            ;;
        \? )
            echo "Invalid option: $OPTARG" 1>&2
            usage
            ;;
        : )
            echo "Invalid option: $OPTARG requires an argument" 1>&2
            usage
            ;;
    esac
done
shift $((OPTIND -1))

# Validate required arguments
if [ -z "$TEST_DATASET" ] || [ -z "$LEFTOUT_DATASET" ] || [ -z "$TRAIN_DATASET" ] || [ -z "$MODEL" ]; then
    echo "Error: Missing required arguments" 1>&2
    usage
fi

# Generate output prefixes based on input file names
TEST_DATASET_OUT="encoded_dataset/$(basename "$(dirname "$TEST_DATASET")")/$(basename "$TEST_DATASET" .tsv)"
LEFTOUT_DATASET_OUT="encoded_dataset/$(basename "$(dirname "$LEFTOUT_DATASET")")/$(basename "$LEFTOUT_DATASET" .tsv)"
TRAIN_DATASET_OUT="encoded_dataset/$(basename "$(dirname "$TRAIN_DATASET")")/$(basename "$TRAIN_DATASET" .tsv)"

# Create output directory
mkdir -p "$(dirname "$TEST_DATASET_OUT")"

# Function to check and encode dataset
encode_dataset() {
    local input_file=$1
    local output_prefix=$2
    
    # Check if the dataset and labels .npy files already exist
    if [ ! -f "${output_prefix}_dataset.npy" ] || [ ! -f "${output_prefix}_labels.npy" ]; then
        echo "Encoding dataset: $input_file"
        python "$CODE/encode/binding_2D_matrix_encoder.py" --i_file "$input_file" --o_prefix "$output_prefix"
    else
        echo "Encoded files for $input_file already exist. Skipping encoding."
    fi
}

# Encode datasets
encode_dataset "$TEST_DATASET" "$TEST_DATASET_OUT"
encode_dataset "$LEFTOUT_DATASET" "$LEFTOUT_DATASET_OUT"
encode_dataset "$TRAIN_DATASET" "$TRAIN_DATASET_OUT"

# Train model (check if training dataset files exist)
TRAIN_DATASET_NPY="${TRAIN_DATASET_OUT}_dataset.npy"
TRAIN_LABELS_NPY="${TRAIN_DATASET_OUT}_labels.npy"

# Determine dataset size (can be modified if needed)
DATASET_SIZE=$(wc -l < "$TRAIN_DATASET")

if [ -f "$TRAIN_DATASET_NPY" ] && [ -f "$TRAIN_LABELS_NPY" ]; then
    echo "Training model..."
    python "$CODE/train/CNN_miRBind_2022/miRBind_CNN_training_orig_parameters.py" \
    --data "$TRAIN_DATASET_NPY" \
    --labels "$TRAIN_LABELS_NPY" \
    --dataset_size "$DATASET_SIZE" \
    --ratio 1 \
    --model "$MODEL"
else
    echo "Error: Training dataset or labels file not found. Cannot proceed with training."
    exit 1
fi

echo "Process completed."