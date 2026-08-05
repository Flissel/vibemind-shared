"""Immutable Brain-to-OpenFang V1 contract closure."""

from .brain_openfang import (
    ContractValidationError,
    validate_brain_openfang_handoff_bundle,
)

__all__ = [
    "ContractValidationError",
    "validate_brain_openfang_handoff_bundle",
]
