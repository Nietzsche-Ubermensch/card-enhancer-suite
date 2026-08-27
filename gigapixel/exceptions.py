class GigapixelException(Exception):
    """Base exception for Gigapixel automation."""
    pass


class NotFile(GigapixelException):
    """Raised when the provided path is not a file."""
    pass


class FileAlreadyExists(GigapixelException):
    """Raised when an output file would be overwritten."""
    pass


class ElementNotFound(GigapixelException):
    """Raised when a UI element cannot be located."""
    pass
