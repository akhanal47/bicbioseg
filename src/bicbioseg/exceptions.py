class BicBioSegError(Exception):
    """Base exception for bicbioseg."""


class DatasetError(BicBioSegError):
    """Raised when dataset inputs or structure are invalid."""


class ModelError(BicBioSegError):
    """Raised when model construction or training configuration is invalid."""


class InferenceError(BicBioSegError):
    """Raised when inference inputs or outputs are invalid."""
