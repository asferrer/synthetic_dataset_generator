"""
Effect Registry

Central registry for managing domain-specific effects.
Supports dynamic loading and filtering of effects by domain.
"""

import logging
from typing import Dict, List, Any, Optional, Type, Callable
from dataclasses import dataclass, field

from .base_effect import BaseEffect, EffectResult

logger = logging.getLogger(__name__)


@dataclass
class EffectDefinition:
    """Definition of a registered effect."""
    effect_id: str
    effect_class: Type[BaseEffect]
    domains: List[str] = field(default_factory=list)
    is_universal: bool = False
    enabled: bool = True


class EffectRegistry:
    """
    Registry for domain-specific effects.

    Allows registering effects and filtering them by domain.
    """

    def __init__(self):
        self._effects: Dict[str, EffectDefinition] = {}
        self._instances: Dict[str, BaseEffect] = {}
        self._initialized = False

    def register(
        self,
        effect_id: str,
        effect_class: Type[BaseEffect],
        domains: List[str] = None,
        is_universal: bool = False,
    ) -> None:
        """
        Register an effect.

        Args:
            effect_id: Unique identifier for the effect
            effect_class: Effect class (must inherit from BaseEffect)
            domains: List of domain IDs this effect applies to
            is_universal: If True, effect applies to all domains
        """
        if not issubclass(effect_class, BaseEffect):
            raise TypeError(f"{effect_class} must inherit from BaseEffect")

        self._effects[effect_id] = EffectDefinition(
            effect_id=effect_id,
            effect_class=effect_class,
            domains=domains or [],
            is_universal=is_universal,
        )
        logger.debug(f"Registered effect: {effect_id}")

    def unregister(self, effect_id: str) -> bool:
        """Unregister an effect."""
        if effect_id in self._effects:
            del self._effects[effect_id]
            if effect_id in self._instances:
                del self._instances[effect_id]
            return True
        return False

    def get_effect(self, effect_id: str, **params) -> Optional[BaseEffect]:
        """
        Get an effect instance.

        Args:
            effect_id: Effect identifier
            **params: Parameters to pass to the effect constructor

        Returns:
            Effect instance or None if not found
        """
        if effect_id not in self._effects:
            logger.warning(f"Effect not found: {effect_id}")
            return None

        definition = self._effects[effect_id]

        # Create new instance with params or return cached
        if params or effect_id not in self._instances:
            self._instances[effect_id] = definition.effect_class(**params)

        return self._instances[effect_id]

    def get_effects_for_domain(self, domain_id: str) -> List[BaseEffect]:
        """
        Get all effects available for a domain.

        Args:
            domain_id: Domain identifier

        Returns:
            List of effect instances for the domain
        """
        effects = []
        for effect_id, definition in self._effects.items():
            if definition.is_universal or domain_id in definition.domains or not definition.domains:
                effect = self.get_effect(effect_id)
                if effect:
                    effects.append(effect)
        return effects

    def get_domain_specific_effects(self, domain_id: str) -> List[BaseEffect]:
        """
        Get only domain-specific effects (not universal).

        Args:
            domain_id: Domain identifier

        Returns:
            List of domain-specific effect instances
        """
        effects = []
        for effect_id, definition in self._effects.items():
            if not definition.is_universal and domain_id in definition.domains:
                effect = self.get_effect(effect_id)
                if effect:
                    effects.append(effect)
        return effects

    def get_universal_effects(self) -> List[BaseEffect]:
        """Get all universal effects."""
        effects = []
        for effect_id, definition in self._effects.items():
            if definition.is_universal:
                effect = self.get_effect(effect_id)
                if effect:
                    effects.append(effect)
        return effects

    def list_effects(self, domain_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all registered effects.

        Args:
            domain_id: Optional filter by domain

        Returns:
            List of effect info dictionaries
        """
        result = []
        for effect_id, definition in self._effects.items():
            if domain_id:
                if not (definition.is_universal or domain_id in definition.domains or not definition.domains):
                    continue

            effect = self.get_effect(effect_id)
            if effect:
                info = effect.get_info()
                info["enabled"] = definition.enabled
                result.append(info)

        return result

    def is_effect_enabled_for_domain(self, effect_id: str, domain_id: str) -> bool:
        """Check if effect is enabled for a domain."""
        if effect_id not in self._effects:
            return False

        definition = self._effects[effect_id]
        if not definition.enabled:
            return False

        return definition.is_universal or domain_id in definition.domains or not definition.domains

    def set_effect_enabled(self, effect_id: str, enabled: bool) -> bool:
        """Enable or disable an effect."""
        if effect_id in self._effects:
            self._effects[effect_id].enabled = enabled
            return True
        return False

    def initialize_all(self) -> None:
        """Initialize all registered effects."""
        if self._initialized:
            return

        # Import and register all built-in effects
        self._register_builtin_effects()
        self._initialized = True
        logger.info(f"Effect registry initialized with {len(self._effects)} effects")

    def _register_builtin_effects(self) -> None:
        """Register all built-in effects."""
        # Import universal effects
        try:
            from .universal import (
                ColorCorrectionEffect,
                BlurMatchingEffect,
                LightingEffect,
                ShadowsEffect,
                EdgeSmoothingEffect,
                MotionBlurEffect,
            )

            self.register("color_correction", ColorCorrectionEffect, is_universal=True)
            self.register("blur_matching", BlurMatchingEffect, is_universal=True)
            self.register("lighting", LightingEffect, is_universal=True)
            self.register("shadows", ShadowsEffect, is_universal=True)
            self.register("edge_smoothing", EdgeSmoothingEffect, is_universal=True)
            self.register("motion_blur", MotionBlurEffect, is_universal=True)
        except ImportError as e:
            logger.warning(f"Could not load universal effects: {e}")

        # Import underwater effects
        try:
            from .underwater import (
                UnderwaterTintEffect,
                CausticsEffect,
                WaterAttenuationEffect,
            )

            self.register("underwater_tint", UnderwaterTintEffect, domains=["underwater"])
            self.register("caustics", CausticsEffect, domains=["underwater"])
            self.register("water_attenuation", WaterAttenuationEffect, domains=["underwater"])
        except ImportError as e:
            logger.warning(f"Could not load underwater effects: {e}")

        # Import fire/smoke effects
        try:
            from .fire_smoke import (
                HeatDistortionEffect,
                SmokeHazeEffect,
                FireGlowEffect,
                EmberGlowEffect,
            )

            self.register("heat_distortion", HeatDistortionEffect, domains=["fire_smoke"])
            self.register("smoke_haze", SmokeHazeEffect, domains=["fire_smoke"])
            self.register("fire_glow", FireGlowEffect, domains=["fire_smoke"])
            self.register("ember_glow", EmberGlowEffect, domains=["fire_smoke"])
        except ImportError as e:
            logger.warning(f"Could not load fire_smoke effects: {e}")

        # Import aerial effects
        try:
            from .aerial import (
                AtmosphericHazeEffect,
                DepthFogEffect,
            )

            self.register("atmospheric_haze", AtmosphericHazeEffect, domains=["aerial_birds"])
            self.register("depth_fog", DepthFogEffect, domains=["aerial_birds"])
        except ImportError as e:
            logger.warning(f"Could not load aerial effects: {e}")


# Singleton instance
_registry: Optional[EffectRegistry] = None


def get_effect_registry() -> EffectRegistry:
    """Get the singleton effect registry instance."""
    global _registry
    if _registry is None:
        _registry = EffectRegistry()
    return _registry
