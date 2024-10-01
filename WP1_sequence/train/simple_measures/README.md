## Overview

This project implements two machine learning approaches, **Logistic Regression** and **Random Forest**, to predict miRNA-target binding interactions. Both models utilize k-mer reverse complement features derived from miRNA and target gene sequences, providing a comprehensive framework for identifying potential miRNA-target pairs.

## Methods

### Data Preprocessing and Feature Extraction

**Sequence Standardization:**  
miRNA sequences were standardized by truncating or padding them to a fixed length of 20 nucleotides. Sequences longer than 20 nucleotides were truncated to the first 20 bases, while shorter sequences were padded with 'N' to achieve the desired length. Only canonical nucleotides ('A', 'T', 'G', 'C') were retained.

**K-mer Reverse Complement Feature Extraction:**  
For each miRNA-target gene pair, k-mer reverse complement counts were computed for k-values ranging from 2 to 12. Specifically, for each position within the miRNA sequence, every possible k-mer was extracted. The reverse complement of each k-mer was then determined and counted within the corresponding target gene sequence. Features were labeled in the format `pos{position}_k{k}`, indicating the position within the miRNA sequence and the k-mer length.

**Feature Matrix Construction:**  
A feature matrix was constructed where each row represents a miRNA-target pair and each column corresponds to a specific k-mer reverse complement feature. Counts were aggregated across all positions and k-values. To manage computational resources, a random subset of 500,000 samples was selected from the training dataset of 2.5 million samples. Stratified sampling was employed to preserve the class distribution within the subset.

### Model Training

**Logistic Regression:**  
A Logistic Regression model with L2 regularization was trained using the extracted k-mer reverse complement features. The model was configured with class balancing (`class_weight='balanced'`) to address potential class imbalances within the training data. Feature scaling was performed using `StandardScaler` to normalize the feature values, enhancing model performance and convergence.

**Random Forest:**  
In parallel, a Random Forest classifier was trained using the same feature set. Configured with 100 decision trees (`n_estimators=100`), the Random Forest model also employed class balancing to mitigate class imbalance issues. This ensemble approach leverages multiple decision trees to improve predictive accuracy and control overfitting. Feature scaling was not applied to the Random Forest model, as it is inherently insensitive to feature scaling.

### Model Evaluation

Both models were evaluated using a separate validation set, with the following performance metrics computed:

- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):** Assesses the model's ability to distinguish between positive and negative classes.
- **PR-AUC (Precision-Recall Area Under Curve):** Evaluates the trade-off between precision and recall, especially relevant for imbalanced datasets.
- **Precision:** The ratio of true positive predictions to the total positive predictions made.
- **Recall:** The ratio of true positive predictions to the actual number of positive instances.
- **F1-Score:** The harmonic mean of precision and recall, providing a balance between the two.

Additionally, feature importance analyses were conducted:

- **Logistic Regression:** The top five features with the highest and lowest coefficients were identified, reflecting their influence on the model's predictions. For each feature, the percentage of positive and negative samples containing the feature was calculated to assess its discriminative power.
  
- **Random Forest:** The top five features with the highest and lowest importances were determined based on the model's feature importance scores. Corresponding percentages of positive and negative samples possessing these features were computed to understand their contribution to model predictions.

### Prediction and Inference

For inference, both trained models generated probability scores indicating the likelihood of miRNA-target binding interactions for new data. These scores were appended to the inference dataset, enabling threshold-based classification and downstream analysis. The script facilitates saving the trained models and scaler for future use, ensuring reproducibility and ease of deployment.

### Implementation Details

The predictive models were implemented in Python 3, utilizing the following libraries:

- **pandas:** For data manipulation and processing.
- **numpy:** For numerical operations.
- **scikit-learn:** For machine learning modeling and evaluation.
- **tqdm:** For displaying progress bars during data processing.
- **joblib:** For model persistence (saving and loading trained models).

## Usage

### Requirements

Ensure the following Python packages are installed:

- pandas
- numpy
- scikit-learn
- tqdm
- joblib

Install the required packages using pip:

```bash
pip install pandas numpy scikit-learn tqdm joblib
```

Run the script using the following command:
```bash
python3 kmer_count/kmer_predictor_lr_rf.py --train data/training_data.tsv --inference data/inference_data.tsv --output kmer_count/inference_with_scores_lr_rf.tsv --evaluate --save_models --model_dir kmer_count/saved_models/
```

