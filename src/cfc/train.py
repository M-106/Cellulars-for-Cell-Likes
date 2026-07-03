# -----------
# > Imports <
# -----------
import time
import os
import shutil

import torch

from cfc.utils.config import save_config



# --------------
# > Train Loop <
# --------------

def train(
        model_name, 
        dataset_name, 
        num_epochs, 
        batch_size, 
        learning_rate, 
        weight_decay, 
        criterion,
        optimizer, 
        scheduler, 
        output_dir,
        exp_name
    ):
    """
    Train a model on a specified dataset.

    Args:
        model_name (str): Name of the model to train.
        dataset_name (str): Name of the dataset to use.
        num_epochs (int): Number of epochs to train for.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        criterion (str): Loss/criterion to use.
        optimizer (str): Optimizer to use.
        scheduler (str): Learning rate scheduler to use.
        output_dir (str): Directory to save the trained model, plots, and logs.
        exp_name (str): Name of the experiment for logging and saving purposes.
    """
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.accelerator.current_accelerator()

    # get data
    train_data = get_data(dataset_name, batch_size, partition="train")
    val_data = get_data(
        dataset_name=dataset_name, 
        batch_size=batch_size, 
        partition="val",
        shuffle=False
    )








# -------------
# > Execution <
# -------------

def main(config):

    # extract configs
    model_name = config.model.name
    dataset_name = config.data.name
    num_epochs = config.train.num_epochs
    batch_size = config.train.batch_size
    learning_rate = config.train.learning_rate
    weight_decay = config.train.weight_decay
    criterion = config.train.criterion
    optimizer = config.train.optimizer
    scheduler = config.train.scheduler
    output_dir = config.train.output_dir
    exp_name = config.train.exp_name


    # create exp output folder
    output_dir = f"{output_dir}/{time.strftime('%Y-%m-%d_%H-%M-%S')}_{exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


    # track config for this run
    save_config(config, f"{output_dir}/config.yaml")


    # start training
    train(
        model_name=model_name,
        dataset_name=dataset_name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        exp_name=exp_name
    )







