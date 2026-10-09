#!/usr/bin/env python3
"""
LSGEmu local analysis subset.

This package contains the local analysis modules required by LSGEmu, so the
runner can resolve its analysis stack entirely from inside lsgemu/.
"""

__all__ = [
    "FirmwareAnalyzer",
    "AnalysisResult",
    "IntelligentEmulator",
]


def __getattr__(name):
    if name in {"FirmwareAnalyzer", "AnalysisResult"}:
        from .firmware_analyzer import FirmwareAnalyzer, AnalysisResult

        exports = {
            "FirmwareAnalyzer": FirmwareAnalyzer,
            "AnalysisResult": AnalysisResult,
        }
        return exports[name]

    if name == "IntelligentEmulator":
        from .intelligent_emulator import IntelligentEmulator

        return IntelligentEmulator

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
