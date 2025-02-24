import numpy as np
import tensorflow as tf
import random
import logging


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


def setup_logger(log_file):
    """Configure logging to file and console."""
    logger = logging.getLogger('mirbind_train')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(console_handler)
    
    return logger