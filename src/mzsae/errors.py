"""Custom exception classes for MZSAE."""


class MZSAEError(Exception):
    """Base exception for all MZSAE errors."""

    pass


class BackendNotAvailableError(MZSAEError):
    """Raised when the requested execution backend cannot be initialized."""

    pass


class ConfigError(MZSAEError):
    """Raised when configuration validation or resolution fails."""

    pass


class DeviceMismatchError(MZSAEError):
    """Raised when input tensors do not match the expected execution device."""

    pass
