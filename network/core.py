"""
Module containing the base class for INR networks.

Classes
-------
BaseNetwork
    Base class for INR networks.
"""

import time
from typing import Optional

import numpy as np
import torch
import torch.utils.data

from .utils import (get_loss_string, print_batch_progress,
                    print_batch_training_summary, print_progress,
                    print_training_summary)


class BaseNetwork(torch.nn.Module):
    """
    Base class for INR networks.

    Attributes
    ----------
    None

    Methods
    -------
    forward(x)
        Forward pass of the network.
        Must be implemented in the derived class.

    fit(input_dataloader, epochs, optimizer, **kwargs)
        Fit the network to the data.

    generate(input_dataloader, **kwargs)
        Generate the output of the network.
    """
    def __init__(
            self,
            **kwargs
        ):
        """
        Base class for INR networks.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the derived class.
        """

        raise NotImplementedError

    def fit(
            self,
            input_dataloader: torch.utils.data.DataLoader,
            epochs: int,
            optimizer: torch.optim.Optimizer,
            **kwargs
        ):
        """
        Fit the network to the data.

        Parameters
        ----------
        input_dataloader : torch.utils.data.DataLoader
            DataLoader for the input data.

        epochs : int
            Number of epochs to train the network.

        optimizer : torch.optim.Optimizer
            Optimizer for the network.

        Keyword Arguments
        -----------------
        verbose : bool, optional
            Whether to print additional information, by default False.

        save_path : str, optional
            Path to save the best model, by default None.

        scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Learning rate scheduler, by default None.

        device : torch.device, optional
            Device to use for training, by default torch.device('cpu').

        loss_fn : torch.nn.modules.loss._Loss, optional
            Loss function to use, by default torch.nn.MSELoss(reduction='mean').
        """

        verbose = kwargs.get('verbose', False)
        save_path = kwargs.get('save_path', None)

        if verbose:
            if save_path is not None:
                print(f'Saving best model to {save_path}')

        if len(input_dataloader) == 1:
            loss = self._single_batch_training(
                input_dataloader,
                epochs,
                optimizer,
                **kwargs
            )
        else:
            loss = self._batch_training(
                input_dataloader,
                epochs,
                optimizer,
                **kwargs
            )

        if verbose:
            if save_path is not None:
                print(f"Best loss: {get_loss_string(loss.min())}")

            else:
                print(f"Final loss: {get_loss_string(loss[-1])}")

        return loss

    def generate(
            self,
            input_dataloader: torch.utils.data.DataLoader,
            **kwargs
        ) -> torch.Tensor:
        """
        Generate the output of the network.

        Parameters
        ----------
        input_dataloader : torch.utils.data.DataLoader
            DataLoader for the input data.
            Assumes that only the input data is present.

        Keyword Arguments
        -----------------
        device : torch.device, optional
            Device to use for training, by default torch.device('cpu').

        Returns
        -------
        torch.Tensor
            Output of the network.
        """

        device = kwargs.get('device', torch.device('cpu'))

        output = []
        for input_batch in input_dataloader:
            input_batch = input_batch.to(device)
            output_batch = self(input_batch)
            output.append(output_batch.detach().cpu())
        return torch.cat(output)


    def _single_batch_training(
            self,
            input_dataloader: torch.utils.data.DataLoader,
            epochs: int,
            optimizer: torch.optim.Optimizer,
            save_path: Optional[str] = None,
            **kwargs
        ) -> np.ndarray:
        """
        Train the network on a single batch of data.

        Parameters
        ----------
        input_dataloader : torch.utils.data.DataLoader
            DataLoader for the input data.

        epochs : int
            Number of epochs to train the network.

        optimizer : torch.optim.Optimizer
            Optimizer for the network.

        save_path : str, optional
            Path to save the best model, by default None.

        Keyword Arguments
        -----------------
        scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Learning rate scheduler, by default None.

        device : torch.device, optional
            Device to use for training, by default torch.device('cpu').

        loss_fn : torch.nn.modules.loss._Loss, optional
            Loss function to use, by default torch.nn.MSELoss(reduction='mean').

        Returns
        -------
        np.ndarray
            Loss after each epoch.
        """

        scheduler = kwargs.get('scheduler', None)
        scheduler_args = kwargs.get('scheduler_args', None)
        device = kwargs.get('device', torch.device('cpu'))

        start_time = time.time()

        input_data, target_data = next(iter(input_dataloader))
        input_data = input_data.to(device)
        target_data = target_data.to(device)

        loss_list = []

        if save_path is not None:
            best_loss = float('inf')
        for epoch in range(epochs):
            loss = self._training_step(
                input_data,
                target_data,
                optimizer,
                **kwargs
            )

            if scheduler is not None:
                if scheduler_args is None:
                    scheduler.step()
                elif scheduler_args == 'loss':
                    scheduler.step(loss)
                else:
                    raise ValueError(
                        f"Invalid value for scheduler_args: {scheduler_args}"
                    )

            print_progress(
                epoch,
                epochs,
                loss,
                start_time
            )

            if save_path is not None:
                if loss < best_loss:
                    best_loss = loss
                    torch.save(self.state_dict(), save_path)

            loss_list.append(loss)

        print_training_summary(
            epochs,
            loss,
            start_time
        )

        return np.array(loss_list)

    def _batch_training(
            self,
            input_dataloader: torch.utils.data.DataLoader,
            epochs: int,
            optimizer: torch.optim.Optimizer,
            save_path: Optional[str] = None,
            **kwargs
        ) -> np.ndarray:
        """
        Train the network on data with more than one batch.

        Parameters
        ----------
        input_dataloader : torch.utils.data.DataLoader
            DataLoader for the input data.

        epochs : int
            Number of epochs to train the network.

        optimizer : torch.optim.Optimizer
            Optimizer for the network.

        save_path : str, optional
            Path to save the best model, by default None.

        Keyword Arguments
        -----------------
        scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Learning rate scheduler, by default None.

        device : torch.device, optional
            Device to use for training, by default torch.device('cpu').

        loss_fn : torch.nn.modules.loss._Loss, optional
            Loss function to use, by default torch.nn.MSELoss(reduction='mean').

        Returns
        -------
        np.ndarray
            Loss after each epoch.
        """

        scheduler = kwargs.get('scheduler', None)
        scheduler_args = kwargs.get('scheduler_args', None)
        device = kwargs.get('device', torch.device('cpu'))

        loss_list = []
        if save_path is not None:
            best_loss = float('inf')
        for epoch in range(epochs):
            start_time = time.time()
            loss = 0
            for i, (input_data, target_data) in enumerate(input_dataloader):
                input_data = input_data.to(device)
                target_data = target_data.to(device)

                loss += self._training_step(
                    input_data,
                    target_data,
                    optimizer,
                    **kwargs
                ) * input_data.shape[0]

                if scheduler is not None:
                    if scheduler_args is None:
                        scheduler.step()
                    elif scheduler_args == 'loss':
                        scheduler.step(loss / ((i+1) * input_data.shape[0]))
                    else:
                        raise ValueError(
                            f"Invalid value for scheduler_args: {scheduler_args}"
                        )

                print_batch_progress(
                    epoch,
                    epochs,
                    i,
                    len(input_dataloader),
                    loss / ((i+1) * input_data.shape[0]),
                    start_time
                )

            loss /= len(input_dataloader.dataset)
            loss_list.append(loss)

            if save_path is not None:
                if loss < best_loss:
                    best_loss = loss
                    torch.save(self.state_dict(), save_path)

            print_batch_training_summary(
                epoch,
                epochs,
                len(input_dataloader),
                loss,
                start_time
            )

        return np.array(loss_list)

    def _training_step(
            self,
            input_batch: torch.Tensor,
            target_batch: torch.Tensor,
            optimizer: torch.optim.Optimizer,
            **kwargs
        ) -> torch.Tensor:
        """
        Perform a single training step.

        Parameters
        ----------
        input_batch : torch.Tensor
            Input batch.

        target_batch : torch.Tensor
            Target batch.

        optimizer : torch.optim.Optimizer
            Optimizer for the network.

        Keyword Arguments
        -----------------
        loss_fn : torch.nn.modules.loss._Loss, optional
            Loss function to use, by default torch.nn.MSELoss(reduction='mean').

        Returns
        -------
        torch.Tensor
            Loss of the batch.

        Notes
        -----
        The function assumes that the inputs and targets are already on the
        correct device.
        """

        loss_fn = kwargs.get('loss_fn', torch.nn.MSELoss(reduction='mean'))

        optimizer.zero_grad(set_to_none=True)

        output_batch = self(input_batch)
        loss = loss_fn(output_batch, target_batch)

        loss.backward()
        optimizer.step()

        return loss.item()
