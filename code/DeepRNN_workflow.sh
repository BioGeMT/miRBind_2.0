#!/bin/bash



# function to display usage information
usage() {
    echo "Usage: $0 [noreg | withreg] [plot_true | plot_false]"
    echo ""
    echo "Arguments:"
    echo "  noncodingRNA | miRNA      : Specify the micro-rna column name. Allowed values: noncodingRNA | miRNA"
    echo "                            : Note: This was done since datasets where enoucuntered with different column names for miRNA"
    echo "  noreg | withreg           : Specify the regularization type."
    echo "  plot_true | plot_false    : Specify whether to plot the results."
    echo ""
    echo "Options:"
    echo "  -h, --help                : Display usage information."
    exit 0
}

# display usage if help flag is passed
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi



# * START WORKFLOW ---
PATH_TO_BASH_LOG="DeepRNN_Bash_Logs.txt"

timestamp() {
    date '+%H:%M:%S'
}

# function to print error messages in red
print_error() {
    local msg="[$(timestamp)] ERROR: $1"
    echo -e "\e[31m$msg\e[0m"
    echo "$msg" >> "$PATH_TO_BASH_LOG"
}

# function to print success messages in green
print_success() {
    local msg="[$(timestamp)] $1"
    echo -e "\e[32m$msg\e[0m"
    echo "$msg" >> "$PATH_TO_BASH_LOG"
}

# function to print warning messages in yellow
print_warning() {
    local msg="[$(timestamp)] $1"
    echo -e "\e[33m$msg\e[0m"
    echo "$msg" >> "$PATH_TO_BASH_LOG"
}

# function to print echo messages
print_echo() {
    echo "$1"
    echo "$1" >> "$PATH_TO_BASH_LOG"
}



DT=$(date '+%d/%m/%Y --- %H:%M:%S')
echo "DeepRNN with $REG_TYPE workflow started at [$DT]" > "$PATH_TO_BASH_LOG"
print_echo ""



# check if the script is run with two arguments
if [ "$#" -ne 3 ]; then
    print_error "Invalid number of arguments. Use -h or --help for more information."
    exit 1
fi

# set the column name based on user input
if [ "$1" == "noncodingRNA" ]; then
    MIRNA_COL_NAME="noncodingRNA"
elif [ "$1" == "miRNA" ]; then
    MIRNA_COL_NAME="miRNA"
else
    print_error "Invalid argument: '$1' is not recognized. Allowed values: noncodingRNA | miRNA"
    exit 1
fi

# set the training script based on user input
if [ "$2" == "noreg" ]; then
    REG_TYPE="NoReg"
    TRAIN_SCRIPT="DeepRNN_NoReg.py"
elif [ "$2" == "withreg" ]; then
    REG_TYPE="WithReg"
    TRAIN_SCRIPT="DeepRNN_WithReg.py"
else
    print_error "Invalid argument: '$2' is not recognized. Allowed values: noreg | withreg"
    exit 1
fi

# set the plot bool based on user input
if [ "$3" == "plot_true" ]; then
    PLOT_BOOL="true"
elif [ "$3" == "plot_false" ]; then
    PLOT_BOOL="false"
else
    print_error "Invalid argument: '$3' is not recognized. Allowed values: plot_true | plot_false"
    exit 1
fi



# * ENCODE ---
# datasets, prefix, miRNA column_name are required

TRAINING_DATASETS_PATH="data/DeepRNN_data/training"
TESTING_DATASETS_PATH="data/DeepRNN_data/testing"
ENCODER_SCRIPT="code/machine_learning/encode/sequence_encoder_16ntPairs.py"

# ensure training and testing directories exist
if [ ! -d "$TRAINING_DATASETS_PATH" ]; then
    print_error "Training dataset directory '$TRAINING_DATASETS_PATH' not found!"
    exit 1
fi

if [ ! -d "$TESTING_DATASETS_PATH" ]; then
    print_error "Testing dataset directory '$TESTING_DATASETS_PATH' not found!"
    exit 1
fi

# ensure encoder script exists
if [ ! -f "$ENCODER_SCRIPT" ]; then
    print_error "Encoder script '$ENCODER_SCRIPT' not found!"
    exit 1
fi

# get all .tsv files from training and testing directories
TRAINING_DATA_FILES=$(find "$TRAINING_DATASETS_PATH" -type f -name "*.tsv")
TESTING_DATA_FILES=$(find "$TESTING_DATASETS_PATH" -type f -name "*.tsv")

# validate that training datasets were found
if [ -z "$TRAINING_DATA_FILES" ]; then
    print_error "No training datasets found in '$TRAINING_DATASETS_PATH'!"
    exit 1
fi

# validate that testing datasets were found
if [ -z "$TESTING_DATA_FILES" ]; then
    print_error "No testing datasets found in '$TESTING_DATASETS_PATH'!"
    exit 1
fi

# print confirmation
ENCODER_SCRIPT_BASENAME=$(basename "$ENCODER_SCRIPT")
print_success "Found Encoder Script: $ENCODER_SCRIPT_BASENAME"
print_success "Found Training Datasets"
print_success "Found Testing Datasets"
print_success "Encoding..."
print_echo ""

# encode training datasets
for dataset in $TRAINING_DATA_FILES; do
    # get basename of dataset
    dataset_basename=$(basename "$dataset" ".tsv")

    # define expected output file paths
    encoded_dataset="${TRAINING_DATASETS_PATH}/${dataset_basename}_dataset.npy"
    encoded_labels="${TRAINING_DATASETS_PATH}/${dataset_basename}_labels.npy"
    
    # check if encoded files already exist
    if [ -f "$encoded_dataset" ] && [ -f "$encoded_labels" ]; then
        print_success "Encoded files for '$dataset_basename' already exist. Skipping encoding..."
        continue
    fi

    python3 $ENCODER_SCRIPT --i_file $dataset --o_prefix $TRAINING_DATASETS_PATH/$dataset_basename --column_name "$MIRNA_COL_NAME"
    if [ $? -ne 0 ]; then
        print_error "Failed to encode training dataset '$dataset'!"
        exit 1
    fi
done

print_echo ""

# encode testing datasets
for dataset in $TESTING_DATA_FILES; do
    # get basename of dataset
    dataset_basename=$(basename "$dataset" ".tsv")

    # define expected output file paths
    encoded_dataset="${TESTING_DATASETS_PATH}/${dataset_basename}_dataset.npy"
    encoded_labels="${TESTING_DATASETS_PATH}/${dataset_basename}_labels.npy"
    
    # check if encoded files already exist
    if [ -f "$encoded_dataset" ] && [ -f "$encoded_labels" ]; then
        print_success "Encoded files for '$dataset_basename' already exist. Skipping encoding..."
        continue
    fi

    python3 $ENCODER_SCRIPT --i_file $dataset --o_prefix $TESTING_DATASETS_PATH/$dataset_basename --column_name "$MIRNA_COL_NAME"
    if [ $? -ne 0 ]; then
        print_error "Failed to encode testing dataset '$dataset'!"
        exit 1
    fi
done

print_echo ""
print_success "Successfully encoded all datasets"
print_echo ""



# * TRAIN ---
# training_datasets.npy, training_labels.npy are required 
# (-plots and -lr [learning rate] are optional)

# ensure script exists before attempting to execute it
SCRIPT_PATH="code/machine_learning/train/DeepRNN/$TRAIN_SCRIPT"
if [ ! -f "$SCRIPT_PATH" ]; then
    print_error "Training script '$SCRIPT_PATH' not found!"
    exit 1
fi

# output message if script is found and skip line
SCRIPT_BASENAME=$(basename "$SCRIPT_PATH")
print_success "Found Training Script: $TRAINING_SCRIPT_BASENAME"
print_success "Selected plot bool: $PLOT_BOOL"
print_success "Proceeding with execution..."
print_echo ""

# ensure training directory exist
if [ ! -d "$TRAINING_DATASETS_PATH" ]; then
    print_error "Testing dataset directory '$TRAINING_DATASETS_PATH' not found!"
    exit 1
fi

# get all training datasets containing '_dataset.npy'
TRAIN_DATA_FILES=$(find "$TRAINING_DATASETS_PATH" -type f -name "*_dataset.npy" | sort | tr '\n' ',' | sed 's/,$//')
# get all training datasets containing '_label.npy'
TRAIN_LABEL_FILES=$(find "$TRAINING_DATASETS_PATH" -type f -name "*_labels.npy" | sort | tr '\n' ',' | sed 's/,$//')

# validate that training dataset files were found
if [ -z "$TRAIN_DATA_FILES" ]; then
    print_error "No training datasets found in '$TRAINING_DATASETS_PATH'!"
    exit 1
fi

# validate that training label files were found
if [ -z "$TRAIN_LABEL_FILES" ]; then
    print_error "No training labels found in '$TRAINING_DATASETS_PATH'!"
    exit 1
fi

# print confirmation
print_success "Found training datasets"
print_success "Found training labels"
print_success "Training model... - See training logs for more details"
print_echo ""

# run the training script
python3 "$SCRIPT_PATH" --encoded_data "$TRAIN_DATA_FILES" --encoded_labels "$TRAIN_LABEL_FILES" --plot_plots "$PLOT_BOOL"

# print success message
if [ $? -ne 0 ]; then
    print_error "Failed to train model!"
    exit 1 # comment out to continue with the rest of the script when debugging locally
fi

print_success "Training completed successfully"
print_echo ""



# * PREDICT ---
# models and testing_datasets.npy are needed

# ensure testing directory exist
if [ ! -d "$TESTING_DATASETS_PATH" ]; then
    print_error "Testing dataset directory '$TESTING_DATASETS_PATH' not found!"
    exit 1
fi

SAVED_MODELS_PATH="Saves/DeepRNN_Models"
PREDICTIONS_SCRIPT="code/machine_learning/evaluate/DeepRNN/model_predict.py"

# ensure model dir exists
if [ ! -d "$SAVED_MODELS_PATH" ]; then
    print_error "Saved models directory '$SAVED_MODELS_PATH' not found!"
    exit 1
fi

# check if the prediction script exists
if [ ! -f "$PREDICTIONS_SCRIPT" ]; then
    print_error "Predictions script '$PREDICTIONS_SCRIPT' not found!"
    exit 1
fi

# get all testing datasets containing '_dataset.npy'
TEST_DATA_FILES=$(find "$TESTING_DATASETS_PATH" -type f -name "*_dataset.npy" | sort | tr '\n' ',' | sed 's/,$//')
# get all trained models ending in '.keras'
MODEL_FILES=$(find "$SAVED_MODELS_PATH" -type f -name "*.keras" | sort | tr '\n' ',' | sed 's/,$//')

# validate that test dataset files were found
if [ -z "$TEST_DATA_FILES" ]; then
    print_error "No testing datasets found in '$TESTING_DATASETS_PATH'!"
    exit 1
fi

# validate that model files were found
if [ -z "$MODEL_FILES" ]; then
    print_error "No trained models found in '$SAVED_MODELS_PATH'!"
    exit 1
fi

# print confirmation
PREDICTIONS_SCRIPT_BASENAME=$(basename "$PREDICTIONS_SCRIPT")
print_success "Found $PREDICTIONS_SCRIPT_BASENAME"
print_success "Selected regularization type: $REG_TYPE"
print_success "Found testing datasets"
print_success "Found trained models"
print_success "Running predictions..."
print_echo ""

# run the predictions script
python3 "$PREDICTIONS_SCRIPT" --encoded_data "$TEST_DATA_FILES" --trained_models "$MODEL_FILES" --regularization "$REG_TYPE"

# print success message
if [ $? -ne 0 ]; then
    print_error "Failed to generate predictions"
    exit 1
fi

print_success "Predictions obtained successfully"
print_echo ""



# * EVALUATE ---
# testing_datasets.npy, testing_labels.npy, -preds, -models are required
# (-plots is optional)

SAVED_PREDICTIONS_PATH="Saves/DeepRNN_Predictions"
EVALUATIONS_SCRIPT="code/machine_learning/evaluate/DeepRNN/model_evaluate.py"

# ensure predictions dir exists
if [ ! -d "$SAVED_PREDICTIONS_PATH" ]; then
    print_error "Saved models directory '$SAVED_PREDICTIONS_PATH' not found!"
    exit 1
fi

# check if the evaluations script exists
if [ ! -f "$EVALUATIONS_SCRIPT" ]; then
    print_error "Predictions script '$EVALUATIONS_SCRIPT' not found!"
    exit 1
fi

# get all testing datasets containing '_labels.npy'
TEST_LABEL_FILES=$(find "$TESTING_DATASETS_PATH" -type f -name "*_labels.npy" | sort | tr '\n' ',' | sed 's/,$//')
# get all prediction files ending in '.tsv'
PREDICTION_FILES=$(find "$SAVED_PREDICTIONS_PATH" -type f -name "*.tsv" | sort | tr '\n' ',' | sed 's/,$//')

# validate that test label files were found
if [ -z "$TEST_LABEL_FILES" ]; then
    print_error "No testing datasets found in '$TESTING_DATASETS_PATH'!"
    exit 1
fi

# validate that prediction files were found
if [ -z "$PREDICTION_FILES" ]; then
    print_error "No prediction files found in '$SAVED_PREDICTIONS_PATH'!"
    exit 1
fi

# print confirmation
EVALUATIONS_SCRIPT_BASENAME=$(basename "$EVALUATIONS_SCRIPT")
print_success "Found $EVALUATIONS_SCRIPT_BASENAME"
print_success "Selected regularization type: $REG_TYPE"
print_success "Selected plot bool: $PLOT_BOOL"
print_success "Found testing datasets"
print_success "Found testing labels"
print_success "Found predictions"
print_success "Running evaluations... - See evaluation logs for more details"
echo ""

# run the evaluations script
python3 "$EVALUATIONS_SCRIPT" --encoded_data "$TEST_DATA_FILES" --encoded_labels "$TEST_LABEL_FILES" --predictions "$PREDICTION_FILES" --regularization "$REG_TYPE" --plot_plots "$PLOT_BOOL"

# print success message
if [ $? -ne 0 ]; then
    print_error "Failed to evaluate model!"
    exit 1
fi

print_echo ""
print_success "Evaluations obtained successfully"
print_echo ""
print_success "DeepRNN $REG_TYPE pipeline completed successfully"
print_warning "NOTE: Please remove, rename or move the 'Saves' directory to avoid conflicts with future runs"

mv "$PATH_TO_BASH_LOG" "Saves/"

exit 0
