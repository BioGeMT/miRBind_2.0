import numpy as np
import tensorflow as tf
import random
import logging
from tensorflow.keras.optimizers import Adam
from tensorflow import keras as K


def set_seeds(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    
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
        metrics=['accuracy', K.metrics.AUC(curve='PR')]
    )
    return model


def setup_logger(log_file, logger_name):
    """Set up a logger to file and console"""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file, 'w')
    console_handler = logging.StreamHandler()
    
    # Create formatters and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger