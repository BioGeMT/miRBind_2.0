# mirBind Model optimisation pipeline

This repository contains scripts for training and evaluating a deep learning model based on variations and tuning of the miRBind architecture for miRNA-binding prediction using eCLIP data from Manakov

1. ../encode_dataset.sh
Converts AGO2 eCLIP datasets from Manakov2022 into the 2D matrix format.
Before running the script, data needs to be place in this directory (same for test and leftout dataset): "miRBind_2.0/data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_train.tsv"

2. hyperparam_optimization.sh
Performs hyperparameter optimization for the model. Saves the best model checkpoint, architecture description, training stats, and metrics.

3. train_model.sh
Trains the model using the optimized hyperparameters until convergence. Saves model checkpoints and training results.
Requires setting the name (timestamp) for your model

4. evaluate_model.sh
Evaluates the trained model on test and left-out datasets. Requires setting the name (timestamp) of your trained model. Generates performance metrics and plots.
Saves results.

../hyperparam_optimization_pipeline.sh
Orchestrates the (almost) entire workflow (except training until convergence with found hyperpara.) in a single execution, combining dataset encoding, hyperparameter optimization, and model evaluation.