#!/usr/bin/env bash

set -euo pipefail

# Parameters
REG_TYPE="noncodingRNA"
PLOT_FLAG="plot_true"
SEED="0"

# ResNet, for each size and each reg-setting - 6 models in total
for SIZE in small medium large; do
  for REG in noreg withreg; do
    echo ">>> Running ResNet: size=${SIZE}, reg=${REG}"
    code/ResNet_workflow.sh "$REG_TYPE" "$SIZE" "$REG" "$PLOT_FLAG" "$SEED"
  done
done

# DeepRNN, with and without regularization - 2 models in total
for REG in noreg withreg; do
  echo ">>> Running DeepRNN: reg=${REG}"
  code/DeepRNN_workflow.sh "$REG_TYPE" "$REG" "$PLOT_FLAG" "$SEED"
done

# BiLSTM, with and without regularization - 2 models in total
for REG in noreg withreg; do
  echo ">>> Running BiLSTM: reg=${REG}"
  code/BiLSTM_workflow.sh "$REG_TYPE" "$REG" "$PLOT_FLAG" "$SEED"
done

echo ">>> All workflows completed."