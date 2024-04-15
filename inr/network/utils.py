"""
Module containing utility functions for INR networks.

Functions
---------
get_loss_string(loss)
    Get a loss string in float or scientific format.

print_batch_progress(epoch, epochs, batch, batches, loss, start_time)
    Print the progress of the training with one line per epoch.

print_batch_training_summary(epoch, epochs, batches, loss, start_time)
    Print the summary of the training one epoch.

print_progress(epoch, epochs, loss, start_time)
    Print the progress of the training.

print_training_summary(epochs, loss, start_time)
    Print the summary of the training.
"""

import datetime
import sys
import time


def _get_progress_string(
        progress: float,
        bar_length: int = 40
    ) -> str:
    """
    Get a progress bar string.

    Parameters
    ----------
    progress : float
        The progress of the training.
    bar_length : int, optional
        The length of the progress bar, by default 40.

    Returns
    -------
    str
        The progress bar string.
    """

    progress_int = int(progress * bar_length)
    progress_string = f'[{progress_int * "="}>'
    progress_string += f'{(bar_length - progress_int - 1) * "."}]'
    return progress_string

def get_loss_string(
        loss: float
    ) -> str:
    """
    Get a loss string in float or scientific format.

    Parameters
    ----------
    loss : float
        The loss.

    Returns
    -------
    str
        The loss string.
    """

    if loss < 1e-3:
        loss_string = f'{loss:.4e}'
    else:
        loss_string = f'{loss:.4f}'
    return loss_string

def print_progress(
        epoch: int,
        epochs: int,
        loss: float,
        start_time: float
    ) -> None:
    """
    Print the progress of the training.

    Parameters
    ----------
    epoch : int
        The current epoch.

    epochs : int
        The total number of epochs.

    loss : float
        The current training loss.

    start_time : float
        The start time of the training.
    """

    time_elapsed = time.time() - start_time
    time_per_epoch = time_elapsed / (epoch + 1)
    time_remaining = (epochs - epoch - 1) * time_per_epoch
    progress = epoch / epochs

    print_string = f'Epoch {epoch + 1}/{epochs}: '
    print_string += f'{_get_progress_string(progress)} - '
    print_string += f'ETA: {int(time_remaining)}s - '
    print_string += f'Loss: {get_loss_string(loss)}'

    sys.stdout.write(f"\r\033[K{print_string}")
    sys.stdout.flush()

def print_training_summary(
        epochs: int,
        loss: float,
        start_time: float
    ) -> None:
    """
    Print the summary of the training.

    Parameters
    ----------
    epochs : int
        The total number of epochs.

    loss : float
        The final training loss.

    start_time : float
        The start time of the training.
    """

    time_elapsed = time.time() - start_time

    print_string = f'Epoch {epochs}/{epochs}: '
    print_string += f'[{"="*40}] - '
    print_string += f'{datetime.timedelta(seconds=time_elapsed)} - '
    print_string += f'Loss: {get_loss_string(loss)}'

    sys.stdout.write(f"\r\033[K{print_string}\n")
    sys.stdout.flush()

def print_batch_progress(
        epoch: int,
        epochs: int,
        batch: int,
        batches: int,
        loss: float,
        start_time: float
    ) -> None:
    """
    Print the progress of the training.

    Parameters
    ----------
    epoch : int
        The current epoch.

    epochs : int
        The total number of epochs.

    batch : int
        The current batch.

    batches : int
        The total number of batches.

    loss : float
        The current training loss.

    start_time : float
        The start time of the training.
    """

    time_elapsed = time.time() - start_time
    time_per_batch = time_elapsed / (batch + 1)
    time_remaining = (batches - batch - 1) * time_per_batch
    progress = (epoch + batch / batches) / epochs

    print_string = f'Epoch {epoch + 1}/{epochs} - Batch {batch + 1}/{batches}: '
    print_string += f'{_get_progress_string(progress)} - '
    print_string += f'ETA: {int(time_remaining)}s - '
    print_string += f'Loss: {get_loss_string(loss)}'

    sys.stdout.write(f"\r\033[K{print_string}")
    sys.stdout.flush()

def print_batch_training_summary(
        epoch: int,
        epochs: int,
        batches: int,
        loss: float,
        start_time: float
    ) -> None:
    """
    Print the summary of the training.

    Parameters
    ----------
    epoch : int
        The current epoch.

    epochs : int
        The total number of epochs.

    batches : int
        The total number of batches.

    loss : float
        The final training loss.

    start_time : float
        The start time of the training.
    """

    time_elapsed = time.time() - start_time

    print_string = f'Epoch {epoch + 1}/{epochs} - Batch {batches}/{batches}: '
    print_string += f'[{"="*40}] - '
    print_string += f'{datetime.timedelta(seconds=time_elapsed)} - '
    print_string += f'Loss: {get_loss_string(loss)}'

    sys.stdout.write(f"\r\033[K{print_string}\n")
    sys.stdout.flush()
