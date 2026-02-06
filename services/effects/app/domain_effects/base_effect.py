"""
Base Effect Class

Provides the base class for all domain-specific effects.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class EffectResult:
    """Result of applying an effect."""
    image: np.ndarray
    success: bool = True
    effect_id: str = ""
    parameters_used: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    error: Optional[str] = None


class BaseEffect(ABC):
    """
    Base class for domain-specific effects.

    All effects should inherit from this class and implement
    the apply() method.
    """

    # Effect metadata (override in subclasses)
    effect_id: str = "base_effect"
    display_name: str = "Base Effect"
    description: str = "Base effect class"
    domains: List[str] = []  # Empty means universal
    is_universal: bool = False

    # Default parameters (override in subclasses)
    default_params: Dict[str, Any] = {}

    def __init__(self, **kwargs):
        """Initialize effect with optional parameter overrides."""
        self.params = {**self.default_params, **kwargs}

    @abstractmethod
    def apply(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
        **kwargs
    ) -> EffectResult:
        """
        Apply the effect to an image.

        Args:
            image: Input image (BGR format)
            mask: Optional object mask
            depth_map: Optional depth map
            **kwargs: Additional effect-specific parameters

        Returns:
            EffectResult with the processed image
        """
        pass

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and merge parameters with defaults."""
        merged = {**self.default_params, **self.params, **params}
        return merged

    def is_enabled_for_domain(self, domain_id: str) -> bool:
        """Check if this effect is enabled for a specific domain."""
        if self.is_universal:
            return True
        if not self.domains:
            return True  # No restriction = universal
        return domain_id in self.domains

    def get_info(self) -> Dict[str, Any]:
        """Get effect information."""
        return {
            "effect_id": self.effect_id,
            "display_name": self.display_name,
            "description": self.description,
            "domains": self.domains,
            "is_universal": self.is_universal,
            "default_params": self.default_params,
        }
