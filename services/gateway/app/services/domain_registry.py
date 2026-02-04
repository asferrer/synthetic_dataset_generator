"""
Domain Registry Service

Manages domain configurations for the multi-domain synthetic dataset generator.
Supports loading built-in domains and custom user-defined domains.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DOMAINS_PATH = os.environ.get(
    "DOMAINS_PATH",
    "/app/config/domains"
)
USER_DOMAINS_PATH = os.environ.get(
    "USER_DOMAINS_PATH",
    "/shared/domains"
)


@dataclass
class DomainRegion:
    """Represents a scene region within a domain."""
    id: str
    name: str
    display_name: str
    color_rgb: List[int] = field(default_factory=lambda: [128, 128, 128])
    sam3_prompt: Optional[str] = None
    detection_heuristics: Optional[Dict] = None


@dataclass
class DomainObject:
    """Represents an object type within a domain."""
    class_name: str
    real_world_size_meters: float
    display_name: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    physics_properties: Dict = field(default_factory=dict)


@dataclass
class DomainEffects:
    """Effects configuration for a domain."""
    domain_specific: List[Dict] = field(default_factory=list)
    disabled: List[str] = field(default_factory=list)
    universal_overrides: Dict = field(default_factory=dict)


@dataclass
class PhysicsConfig:
    """Physics configuration for a domain."""
    physics_type: str = "neutral"
    medium_density: float = 1.0
    float_threshold: Optional[float] = None
    sink_threshold: Optional[float] = None
    surface_zone: Optional[float] = None
    bottom_zone: Optional[float] = None
    gravity_direction: str = "down"


@dataclass
class Domain:
    """Complete domain configuration."""
    domain_id: str
    name: str
    version: str
    regions: List[DomainRegion]
    objects: List[DomainObject]
    compatibility_matrix: Dict[str, Dict[str, float]]
    effects: DomainEffects
    physics: PhysicsConfig
    description: str = ""
    icon: str = "Box"
    is_builtin: bool = False
    presets: List[Dict] = field(default_factory=list)
    labeling_templates: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert domain to dictionary for JSON serialization."""
        return {
            "domain_id": self.domain_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "icon": self.icon,
            "is_builtin": self.is_builtin,
            "regions": [
                {
                    "id": r.id,
                    "name": r.name,
                    "display_name": r.display_name,
                    "color_rgb": r.color_rgb,
                    "sam3_prompt": r.sam3_prompt,
                    "detection_heuristics": r.detection_heuristics
                }
                for r in self.regions
            ],
            "objects": [
                {
                    "class_name": o.class_name,
                    "display_name": o.display_name or o.class_name,
                    "real_world_size_meters": o.real_world_size_meters,
                    "keywords": o.keywords,
                    "physics_properties": o.physics_properties
                }
                for o in self.objects
            ],
            "compatibility_matrix": self.compatibility_matrix,
            "effects": {
                "domain_specific": self.effects.domain_specific,
                "disabled": self.effects.disabled,
                "universal_overrides": self.effects.universal_overrides
            },
            "physics": {
                "physics_type": self.physics.physics_type,
                "medium_density": self.physics.medium_density,
                "float_threshold": self.physics.float_threshold,
                "sink_threshold": self.physics.sink_threshold,
                "surface_zone": self.physics.surface_zone,
                "bottom_zone": self.physics.bottom_zone,
                "gravity_direction": self.physics.gravity_direction
            },
            "presets": self.presets,
            "labeling_templates": self.labeling_templates
        }


class DomainRegistry:
    """
    Registry for managing domain configurations.

    Handles loading, saving, and activating domains.
    Supports both built-in domains (read-only) and user-defined domains.
    """

    def __init__(
        self,
        builtin_path: str = DEFAULT_DOMAINS_PATH,
        user_path: str = USER_DOMAINS_PATH
    ):
        self.builtin_path = Path(builtin_path)
        self.user_path = Path(user_path)
        self._domains: Dict[str, Domain] = {}
        self._active_domain_id: str = "underwater"  # Default
        self._loaded = False

        # Ensure user domains directory exists
        self.user_path.mkdir(parents=True, exist_ok=True)

    def _load_domain_from_file(self, filepath: Path, is_builtin: bool = False) -> Optional[Domain]:
        """Load a domain from a JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Skip template files
            if data.get("domain_id", "").startswith("_") or "$comment" in data:
                return None

            # Parse regions
            regions = [
                DomainRegion(
                    id=r["id"],
                    name=r["name"],
                    display_name=r["display_name"],
                    color_rgb=r.get("color_rgb", [128, 128, 128]),
                    sam3_prompt=r.get("sam3_prompt"),
                    detection_heuristics=r.get("detection_heuristics")
                )
                for r in data.get("regions", [])
            ]

            # Parse objects
            objects = [
                DomainObject(
                    class_name=o["class_name"],
                    display_name=o.get("display_name"),
                    real_world_size_meters=o["real_world_size_meters"],
                    keywords=o.get("keywords", []),
                    physics_properties=o.get("physics_properties", {})
                )
                for o in data.get("objects", [])
            ]

            # Parse effects
            effects_data = data.get("effects", {})
            effects = DomainEffects(
                domain_specific=effects_data.get("domain_specific", []),
                disabled=effects_data.get("disabled", []),
                universal_overrides=effects_data.get("universal_overrides", {})
            )

            # Parse physics
            physics_data = data.get("physics", {})
            physics = PhysicsConfig(
                physics_type=physics_data.get("physics_type", "neutral"),
                medium_density=physics_data.get("medium_density", 1.0),
                float_threshold=physics_data.get("float_threshold"),
                sink_threshold=physics_data.get("sink_threshold"),
                surface_zone=physics_data.get("surface_zone"),
                bottom_zone=physics_data.get("bottom_zone"),
                gravity_direction=physics_data.get("gravity_direction", "down")
            )

            return Domain(
                domain_id=data["domain_id"],
                name=data["name"],
                description=data.get("description", ""),
                version=data.get("version", "1.0.0"),
                icon=data.get("icon", "Box"),
                is_builtin=is_builtin or data.get("is_builtin", False),
                regions=regions,
                objects=objects,
                compatibility_matrix=data.get("compatibility_matrix", {}),
                effects=effects,
                physics=physics,
                presets=data.get("presets", []),
                labeling_templates=data.get("labeling_templates", [])
            )

        except Exception as e:
            logger.error(f"Error loading domain from {filepath}: {e}")
            return None

    def load_all_domains(self) -> None:
        """Load all domains from built-in and user directories."""
        self._domains = {}

        # Load built-in domains
        if self.builtin_path.exists():
            for filepath in self.builtin_path.glob("*.json"):
                if filepath.name.startswith("_"):
                    continue  # Skip templates
                domain = self._load_domain_from_file(filepath, is_builtin=True)
                if domain:
                    self._domains[domain.domain_id] = domain
                    logger.info(f"Loaded built-in domain: {domain.domain_id}")

        # Load user domains (can override built-in with same ID)
        if self.user_path.exists():
            for filepath in self.user_path.glob("*.json"):
                if filepath.name.startswith("_"):
                    continue
                domain = self._load_domain_from_file(filepath, is_builtin=False)
                if domain:
                    if domain.domain_id in self._domains:
                        logger.info(f"User domain overrides built-in: {domain.domain_id}")
                    self._domains[domain.domain_id] = domain
                    logger.info(f"Loaded user domain: {domain.domain_id}")

        self._loaded = True
        logger.info(f"Loaded {len(self._domains)} domains total")

    def ensure_loaded(self) -> None:
        """Ensure domains are loaded."""
        if not self._loaded:
            self.load_all_domains()

    def list_domains(self) -> List[Dict[str, Any]]:
        """List all available domains with summary info."""
        self.ensure_loaded()
        return [
            {
                "domain_id": d.domain_id,
                "name": d.name,
                "description": d.description,
                "icon": d.icon,
                "version": d.version,
                "is_builtin": d.is_builtin,
                "region_count": len(d.regions),
                "object_count": len(d.objects)
            }
            for d in self._domains.values()
        ]

    def get_domain(self, domain_id: str) -> Optional[Domain]:
        """Get a domain by ID."""
        self.ensure_loaded()
        return self._domains.get(domain_id)

    def get_domain_dict(self, domain_id: str) -> Optional[Dict[str, Any]]:
        """Get a domain as a dictionary."""
        domain = self.get_domain(domain_id)
        return domain.to_dict() if domain else None

    def get_active_domain(self) -> Optional[Domain]:
        """Get the currently active domain."""
        return self.get_domain(self._active_domain_id)

    def get_active_domain_id(self) -> str:
        """Get the ID of the currently active domain."""
        return self._active_domain_id

    def set_active_domain(self, domain_id: str) -> bool:
        """Set the active domain."""
        self.ensure_loaded()
        if domain_id in self._domains:
            self._active_domain_id = domain_id
            logger.info(f"Active domain set to: {domain_id}")
            return True
        logger.warning(f"Domain not found: {domain_id}")
        return False

    def create_domain(self, domain_data: Dict[str, Any]) -> Optional[Domain]:
        """Create a new user domain."""
        domain_id = domain_data.get("domain_id")
        if not domain_id:
            raise ValueError("domain_id is required")

        # Check if domain already exists as built-in
        existing = self.get_domain(domain_id)
        if existing and existing.is_builtin:
            raise ValueError(f"Cannot overwrite built-in domain: {domain_id}")

        # Save to user domains directory
        filepath = self.user_path / f"{domain_id}.json"

        # Add metadata
        domain_data["is_builtin"] = False
        if "version" not in domain_data:
            domain_data["version"] = "1.0.0"

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(domain_data, f, indent=2, ensure_ascii=False)

            # Reload to parse and validate
            domain = self._load_domain_from_file(filepath, is_builtin=False)
            if domain:
                self._domains[domain_id] = domain
                logger.info(f"Created domain: {domain_id}")
                return domain
            else:
                filepath.unlink()  # Remove invalid file
                raise ValueError("Invalid domain configuration")

        except Exception as e:
            logger.error(f"Error creating domain: {e}")
            raise

    def update_domain(self, domain_id: str, domain_data: Dict[str, Any]) -> Optional[Domain]:
        """Update an existing user domain."""
        existing = self.get_domain(domain_id)
        if not existing:
            raise ValueError(f"Domain not found: {domain_id}")

        if existing.is_builtin:
            raise ValueError(f"Cannot modify built-in domain: {domain_id}")

        # Merge with existing and save
        domain_data["domain_id"] = domain_id  # Ensure ID doesn't change
        domain_data["is_builtin"] = False

        # Increment version if not specified
        if "version" not in domain_data:
            parts = existing.version.split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            domain_data["version"] = ".".join(parts)

        filepath = self.user_path / f"{domain_id}.json"

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(domain_data, f, indent=2, ensure_ascii=False)

            domain = self._load_domain_from_file(filepath, is_builtin=False)
            if domain:
                self._domains[domain_id] = domain
                logger.info(f"Updated domain: {domain_id}")
                return domain
            else:
                raise ValueError("Invalid domain configuration")

        except Exception as e:
            logger.error(f"Error updating domain: {e}")
            raise

    def delete_domain(self, domain_id: str) -> bool:
        """Delete a user domain."""
        existing = self.get_domain(domain_id)
        if not existing:
            return False

        if existing.is_builtin:
            raise ValueError(f"Cannot delete built-in domain: {domain_id}")

        filepath = self.user_path / f"{domain_id}.json"

        try:
            if filepath.exists():
                filepath.unlink()

            if domain_id in self._domains:
                del self._domains[domain_id]

            # Reset active domain if deleted
            if self._active_domain_id == domain_id:
                self._active_domain_id = "underwater"

            logger.info(f"Deleted domain: {domain_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting domain: {e}")
            raise

    def export_domain(self, domain_id: str) -> Optional[Dict[str, Any]]:
        """Export a domain as JSON-serializable dict."""
        domain = self.get_domain(domain_id)
        if domain:
            data = domain.to_dict()
            data["exported_at"] = datetime.utcnow().isoformat()
            return data
        return None

    def import_domain(self, domain_data: Dict[str, Any], overwrite: bool = False) -> Optional[Domain]:
        """Import a domain from JSON data."""
        domain_id = domain_data.get("domain_id")
        if not domain_id:
            raise ValueError("domain_id is required")

        existing = self.get_domain(domain_id)
        if existing:
            if existing.is_builtin:
                raise ValueError(f"Cannot overwrite built-in domain: {domain_id}")
            if not overwrite:
                raise ValueError(f"Domain already exists: {domain_id}. Use overwrite=True to replace.")

        # Remove export metadata
        domain_data.pop("exported_at", None)

        return self.create_domain(domain_data)

    def get_compatibility_score(
        self,
        object_class: str,
        region_id: str,
        domain_id: Optional[str] = None
    ) -> float:
        """Get compatibility score for object in region."""
        domain = self.get_domain(domain_id) if domain_id else self.get_active_domain()
        if not domain:
            return 0.5  # Default neutral score

        matrix = domain.compatibility_matrix

        # Direct lookup
        if object_class in matrix and region_id in matrix[object_class]:
            return matrix[object_class][region_id]

        # Try keyword matching
        for obj in domain.objects:
            if obj.class_name == object_class:
                continue
            if object_class.lower() in [k.lower() for k in obj.keywords]:
                if obj.class_name in matrix and region_id in matrix[obj.class_name]:
                    return matrix[obj.class_name][region_id]

        return 0.5  # Default neutral score

    def get_object_size(self, class_name: str, domain_id: Optional[str] = None) -> float:
        """Get real-world size for an object class."""
        domain = self.get_domain(domain_id) if domain_id else self.get_active_domain()
        if not domain:
            return 0.25  # Default size

        # Direct lookup
        for obj in domain.objects:
            if obj.class_name.lower() == class_name.lower():
                return obj.real_world_size_meters

        # Try keyword matching
        for obj in domain.objects:
            if class_name.lower() in [k.lower() for k in obj.keywords]:
                return obj.real_world_size_meters

        return 0.25  # Default size

    def get_sam3_prompts(self, domain_id: Optional[str] = None) -> List[tuple]:
        """Get SAM3 prompts for all regions in a domain."""
        domain = self.get_domain(domain_id) if domain_id else self.get_active_domain()
        if not domain:
            return []

        return [
            (r.sam3_prompt, r.id)
            for r in domain.regions
            if r.sam3_prompt
        ]

    def get_domain_effects(self, domain_id: Optional[str] = None) -> Dict[str, Any]:
        """Get effects configuration for a domain."""
        domain = self.get_domain(domain_id) if domain_id else self.get_active_domain()
        if not domain:
            return {"domain_specific": [], "disabled": [], "universal_overrides": {}}

        return {
            "domain_specific": domain.effects.domain_specific,
            "disabled": domain.effects.disabled,
            "universal_overrides": domain.effects.universal_overrides
        }


# Singleton instance
_registry: Optional[DomainRegistry] = None


def get_domain_registry() -> DomainRegistry:
    """Get the singleton domain registry instance."""
    global _registry
    if _registry is None:
        _registry = DomainRegistry()
    return _registry
