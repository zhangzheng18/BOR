#!/usr/bin/env python3
"""
MMIO handler package exports.

Historically this package defaulted to an older experimental handler defined in
__init__.py itself. The current scripts use EnhancedMMIOHandler as the primary
entrypoint, so package-level imports now resolve to the maintained
implementation.
"""

from .enhanced_mmio_handler import EnhancedMMIOHandler

MMIOHandler = EnhancedMMIOHandler

__all__ = ["EnhancedMMIOHandler", "MMIOHandler"]
