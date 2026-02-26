"""
Multi-Head CNN for OOD
"""

from .mheads import MHeads, DenseNetMHeads
from .densenet_components import Bottleneck, SingleLayer, Transition, make_dense_block
from .base_model import PytorchModelBase
from .errors import (
    ModelError,
    InvalidInputShapeError,
    InvalidModelClassCountError,
    MissingNetworkError
)

__version__ = "1.0.0"
__all__ = [
    'MHeads',
    'DenseNetMHeads',
    'Bottleneck',
    'SingleLayer',
    'Transition',
    'make_dense_block',
    'PytorchModelBase',
    'ModelError',
    'InvalidInputShapeError',
    'InvalidModelClassCountError',
    'MissingNetworkError',
]
