"""
Validation Module

Provides automated quality validation for synthetic images:
- quality_metrics.py: LPIPS, FID, anomaly detection
- physics_validator.py: Physical plausibility checks
"""

from .quality_metrics import QualityValidator, QualityScore, Anomaly
from .physics_validator import PhysicsValidator, PhysicsViolation, Position, Object

__all__ = [
    'QualityValidator',
    'QualityScore',
    'Anomaly',
    'PhysicsValidator',
    'PhysicsViolation',
    'Position',
    'Object',
]
