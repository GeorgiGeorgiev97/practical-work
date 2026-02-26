"""
Custom error classes
"""


class ModelError(Exception):
    """Base class for model-related errors."""
    pass


class InvalidInputShapeError(ModelError):
    """Raised when the input shape is invalid."""

    def __init__(self, input_shape):
        self.input_shape = input_shape
        super().__init__(f"Invalid input shape: {input_shape}. "
                         f"Expected 3D shape (channels, height, width) with positive dimensions.")


class InvalidModelClassCountError(ModelError):
    """Raised when the number of classes is invalid."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__(f"Invalid number of classes: {num_classes}. "
                         f"Must be greater than 1.")


class MissingNetworkError(ModelError):
    """Raised when attempting operations without a built network."""

    def __init__(self):
        super().__init__("Network has not been built. Call build() first.")
