"""
Base model class for PyTorch models
"""

import torch
import numpy as np
from pathlib import Path
from errors import MissingNetworkError


class PytorchModelBase:
    """Base class for PyTorch models providing common functionality."""

    def __init__(self, name=None, description=None):
        """
        Initialize model parameters.

        Args:
            name (str): Name of the model
            description (str): Description of the model
        """
        self.name = name or "Untitled Model"
        self.description = description or "No description provided"

        # Model components
        self._model_instance = None
        self._optimizer = None

        # Training state
        self._is_training = False
        self._output_offset = None

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build(self):
        """Build the network. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement build()")

    def train(self):
        """Set model to training mode."""
        if self._model_instance is not None:
            self._model_instance.train()
            self._is_training = True
            torch.set_grad_enabled(True)

    def eval(self):
        """Set model to evaluation mode."""
        if self._model_instance is not None:
            self._model_instance.eval()
            self._is_training = False
            torch.set_grad_enabled(False)

    def save(self, filepath):
        """
        Save model state to file.

        Args:
            filepath (str): Path to save the model
        """
        if self._model_instance is None:
            raise MissingNetworkError()

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model state, optimizer state, and custom data
        save_dict = {
            'model_state_dict': self._model_instance.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict() if self._optimizer else None,
            'custom_data': self._customdata(),
            'name': self.name,
            'description': self.description
        }

        torch.save(save_dict, filepath)

    def load(self, filepath):
        """
        Load model state from file.

        Args:
            filepath (str): Path to load the model from
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        # Restore custom data first (needed for build)
        if 'custom_data' in checkpoint:
            self._setcustomdata(checkpoint['custom_data'])

        # Build model if not already built
        if self._model_instance is None:
            self.build()

        # Load state dicts
        self._model_instance.load_state_dict(checkpoint['model_state_dict'])

        if self._optimizer and checkpoint.get('optimizer_state_dict'):
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore metadata
        self.name = checkpoint.get('name', self.name)
        self.description = checkpoint.get('description', self.description)

    def _customdata(self):
        """
        Get custom data to save with the model.
        Should be implemented by subclasses.

        Returns:
            dict: Custom data
        """
        return {}

    def _setcustomdata(self, data_maps):
        """
        Set custom data loaded from file.
        Should be implemented by subclasses.

        Args:
            data_maps (dict): Custom data
        """
        pass

    def _matchdatatonetwork(self, data):
        """
        Match data format to network requirements.

        Args:
            data: Input data (numpy array or tensor)

        Returns:
            torch.Tensor: Data on correct device
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data).float()

        return data.to(self.device)

    def _forwardstatshook(self, module, inpt, outpt):
        """
        Hook for collecting forward pass statistics.
        Can be extended by subclasses.

        Args:
            module: The module
            inpt: Input to the module
            outpt: Output from the module
        """
        pass

    def _getreconstructioninformationforlayers(self, input_shape, modules):
        """
        Calculate reconstruction information for a sequence of layers.

        Args:
            input_shape (tuple): Input shape (height, width)
            modules (list): List of modules

        Returns:
            tuple: (offset, downsample_factor, interpolation_factor)
        """
        # Simplified implementation - returns dummy values
        # In full implementation, this would track how each layer affects spatial dimensions
        offset = np.array([0, 0])
        downsample_factor = np.array([1, 1])
        interpolation_factor = np.array([1, 1])

        return offset, downsample_factor, interpolation_factor

    def parameters(self):
        """Get model parameters."""
        if self._model_instance is None:
            raise MissingNetworkError()
        return self._model_instance.parameters()

    def to(self, device):
        """
        Move model to device.

        Args:
            device: Target device
        """
        self.device = device
        if self._model_instance is not None:
            self._model_instance.to(device)
        return self
