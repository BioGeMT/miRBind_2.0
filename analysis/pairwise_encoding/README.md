# Pairwise encoding for CNN
The analyses implements a CNN model for miRNA binding site prediction. It:

1) Takes sequences of miRNA and target RNA as input
2) Encodes nucleotide pairs into indices (16 possible pairs + N padding)
3) Processes these through:
- Embedding layer (configurable dimension)
- Three convolutional layers with batch normalization, max pooling, and dropout
- Two fully connected layers
4) Outputs binding probability (0-1)

Training involves:

- Random train/validation split
- Monitors train/validation/test metrics
- Logs configuration (JSON) and training metrics (TSV) with unique timestamp identifiers

CLI arguments allow configuration of model parameters, training settings, and data paths.