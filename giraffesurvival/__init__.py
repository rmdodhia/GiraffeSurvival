"""
GiraffeSurvival: modular giraffe growth analysis package.

This package organizes the prior monolithic script into clear modules:
- models: curve definitions, fitting utilities, and model selection
- data: loading and preparation helpers for wild and zoo datasets
- age: age estimation/refinement utilities
- fitting: configuration and batch fitting helpers
- plotting: visualization utilities
- pipeline: the end-to-end `main()` orchestrator

Convenient entry points:
- `from giraffesurvival.pipeline import main, AnalysisConfig`
"""

from .pipeline import main
from .fitting import AnalysisConfig, MeasurementConfig, MEASUREMENTS

__all__ = [
    "main",
    "AnalysisConfig",
    "MeasurementConfig",
    "MEASUREMENTS",
]
