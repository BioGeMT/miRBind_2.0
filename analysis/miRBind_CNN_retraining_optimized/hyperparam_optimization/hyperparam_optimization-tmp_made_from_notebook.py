# Hyperparam optimization notebook
"""
In this notebook, we will try to optimize the hyperparameters of the miRBind CNN model. Quick guide to what is a [parameter vs. hyperparameter](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/).

We will use [Optuna](https://optuna.org/) framework for this. It will try for us a bunch of different hyperparameter settings and see what combination works the best.

Let's try to optimize number of blocks with convolution layer, kernel size of the convolution, size of the pooling layer, number of blocks with the dense layer and learning rate - these are our hyperparameters.

Our metrics to optimize will be the AU PRC on the validation set (we split the train set into actual training set and validation set).
"""

import numpy as np
from tensorflow import keras as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

import plotly
import logging
import optuna
import optuna.visualization as vis
from optuna.integration import TFKerasPruningCallback

import sys
sys.path.append("../../../code/machine_learning/train/CNN_miRBind_2022/")

from miRBind_CNN_architecture import miRBind_CNN

# it's here for to be able to display plots in jupyter notebook
plotly.io.renderers.default = 'iframe'

def compile_model(model, lr):

    opt = Adam(
        learning_rate=lr,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam")

    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy', K.metrics.AUC(curve='PR')] # adding the metrics on which we want to optimize
        )
    return model

class DataGenerator(Sequence):
    def __init__(self, data_path, labels_path, dataset_size, batch_size=32, validation_split=0.1, is_validation=False, shuffle=True):
        # preload the encoded numpy data
        # the size needed to properly load the array
        self.size = dataset_size

        self.data = np.memmap(data_path, dtype='float32', mode='r', shape=(self.size, 50, 20, 1))
        self.labels = np.memmap(labels_path, dtype='float32', mode='r', shape=(self.size,))
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Determine number of train and validation samples
        self.validation_split = validation_split
        self.num_samples = len(self.data)
        self.num_validation_samples = int(self.num_samples * validation_split)
        self.num_train_samples = self.num_samples - self.num_validation_samples

        # Determine indices for validation and training
        indices = np.arange(self.num_samples)
        if shuffle:
            np.random.shuffle(indices)

        if is_validation:
            self.indices = indices[self.num_train_samples:]
        else:
            self.indices = indices[:self.num_train_samples]

        # Shuffle the data initially
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.ceil(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, idx):
        # Generate one batch of data
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = self.data[batch_indices]
        batch_labels = self.labels[batch_indices]
        return batch_data, batch_labels

    def on_epoch_end(self):
        # Updates indices after each epoch for shuffling
        if self.shuffle:
            np.random.shuffle(self.indices)

"""------------------------------
Choose a dataset on which you want to train. It has to be already encoded with the ```binding_2D_matrix_encoder.py```
"""

# DATASET = "../../../AmiRBench/code/dataset_vOct/Manakov_1_train_dataset.npy"
# DATASET = '../miRBind_CNN_retraining_orig_parameters/encoded_dataset/AGO2_eCLIP_Manakov2022_1_train_dataset.npy'
DATASET = '../../miRBind_CNN_retraining_orig_parameters/encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_1_train_dataset.npy'

# LABELS = "../../../AmiRBench/code/dataset_vOct/Manakov_1_train_labels.npy"
# LABELS = '../miRBind_CNN_retraining_orig_parameters/encoded_dataset/AGO2_eCLIP_Manakov2022_1_train_labels.npy'
LABELS = '../../miRBind_CNN_retraining_orig_parameters/encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_1_train_labels.npy'

DATASET_RATIO = 1
# DATASET_SIZE = 2524246
DATASET_SIZE = 2516195

import pandas as pd
pd.read_csv("../../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_train.tsv", sep='\t')

train_data_gen = DataGenerator(DATASET, LABELS, dataset_size=DATASET_SIZE, validation_split=0.1, is_validation=False)

val_data_gen = DataGenerator(DATASET, LABELS, dataset_size=DATASET_SIZE, validation_split=0.1, is_validation=True)

"""----------------------------
This is the function that creates a model with suggested hyperparameters, trains it and sees how well it performs on the validation set

**Some explanations**

`trial` is the object that "carries the information" about the hyperparameter optimization. `trial.suggest_<something>` means "give me some value for the hyperparameter" that might work well for the model.

`TFKerasPruningCallback` is another hack, where you can stop unpromising training in the middle and scratch it. E.g. when you are training the model with some hyperparameters and after few epochs you see the model doesn't learn anything, you can simply stop the training, remember that this set of hyperparameters didn't work well and you don't have to waste time with worthless training finishing.

`K.callbacks.EarlyStopping` - early stopping method helps to train for the right amount of epochs. It monitors the performance on the validation set and when the model starts overfitting and performing worse, it stops the training.
"""

best_model = None
best_val_auc = 0

def objective(trial):
    global best_model, best_val_auc

    K.backend.clear_session()

    # build the model based on suggested hyperparameters
    cnn_num = trial.suggest_int('cnn_layers_num', 2, 10)
    kernel_size = trial.suggest_int('kernel_size', 3, 10)
    pool_size = trial.suggest_int('pool_size', 1, 8)
    dense_num = trial.suggest_int('dense_layers_num', 2, cnn_num)
    model = miRBind_CNN(cnn_num=cnn_num, kernel_size=kernel_size, pool_size=pool_size, dense_num=dense_num).model
    lr = trial.suggest_float('learning_rate', 0.00001, 0.1)
    model = compile_model(model, lr=lr)

    model_history = model.fit(
        train_data_gen,
        validation_data=val_data_gen,
        epochs=50,
        class_weight={0: 1, 1: DATASET_RATIO},
        callbacks=[TFKerasPruningCallback(trial, "val_auc"),  # get rid of attempts with unpromising hyperparam combination
                   K.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        )

    num_epochs_trained = np.argmax(model_history.history['val_auc'])
    val_auc = model_history.history['val_auc'][num_epochs_trained]

    # check performance of this trial
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model = model  # save the current best model
        model.save('best_model.keras')  # save the model to disk
        logger.info(f"New best model found and saved with Validation AUC: {val_auc}")

    print(f"Validation AU PRC: {val_auc}")

    return val_auc

"""Set up a logger for logging the optimization process to a file"""

logger = logging.getLogger('optuna')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('hyperparam_optimization.log', 'w')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(file_handler)

"""This is the place where we start running the optimization process"""

study = optuna.create_study(direction='maximize', study_name='miRBind_CNN')
study.optimize(objective, n_trials=20)

logger.info("\n")
logger.info(f"Best hyperparameters: {study.best_params}")
logger.info(f"Best value (validation AU PRC): {study.best_value}")

"""Let's plot now how the optimization process went, what was the best set of hyperparameters etc."""

vis.plot_optimization_history(study)

vis.plot_optimization_history(study)

vis.plot_contour(study)

vis.plot_param_importances(study)

vis.plot_slice(study)

"""If you want to, you can save the plots like this:"""

fig = vis.plot_optimization_history(study)
fig.write_image("optimization_history.png")

