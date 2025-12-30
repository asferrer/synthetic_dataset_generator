"""
Configuration manager for object sizes and parameters.
Allows runtime modification of object real-world sizes for better realism.
"""
import json
import os
from typing import Dict, Optional
from pathlib import Path
import threading


class ObjectSizeConfig:
    """Manages object size configuration with thread-safe access."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to config JSON file. If None, uses default location.
        """
        if config_path is None:
            # Default path: config/object_sizes.json
            config_path = os.environ.get(
                "OBJECT_SIZES_CONFIG",
                "/app/config/object_sizes.json"
            )

        self.config_path = config_path
        self._lock = threading.Lock()
        self._config = None
        self._load_config()

    def _load_config(self):
        """Load configuration from JSON file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self._config = json.load(f)
            else:
                # Create default config if file doesn't exist
                self._config = self._get_default_config()
                self._save_config()
        except Exception as e:
            print(f"Error loading config from {self.config_path}: {e}")
            self._config = self._get_default_config()

    def _save_config(self):
        """Save configuration to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            print(f"Error saving config to {self.config_path}: {e}")

    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            "description": "Real-world object sizes in meters",
            "reference_capture_distance": 2.0,
            "sizes": {
                "fish": 0.25,
                "shark": 2.5,
                "plastic_bottle": 0.25,
                "can": 0.12,
                "tire": 0.65,
                "default": 0.25
            },
            "keyword_mappings": {
                "fish": ["fish"],
                "shark": ["shark"],
                "plastic_bottle": ["bottle"],
                "can": ["can"],
                "tire": ["tire"]
            }
        }

    def get_size(self, object_class: str) -> float:
        """
        Get the real-world size for an object class.

        Args:
            object_class: Object class name

        Returns:
            Size in meters
        """
        with self._lock:
            class_lower = object_class.lower().strip()
            sizes = self._config.get("sizes", {})

            # Direct match
            if class_lower in sizes:
                return sizes[class_lower]

            # Keyword matching
            keyword_map = self._config.get("keyword_mappings", {})
            for key, keywords in keyword_map.items():
                if any(kw in class_lower for kw in keywords):
                    return sizes.get(key, sizes.get("default", 0.25))

            # Default
            return sizes.get("default", 0.25)

    def get_all_sizes(self) -> Dict[str, float]:
        """Get all configured sizes."""
        with self._lock:
            return self._config.get("sizes", {}).copy()

    def set_size(self, object_class: str, size: float):
        """
        Set the real-world size for an object class.

        Args:
            object_class: Object class name
            size: Size in meters (must be > 0)
        """
        if size <= 0:
            raise ValueError("Size must be positive")

        with self._lock:
            if "sizes" not in self._config:
                self._config["sizes"] = {}

            class_lower = object_class.lower().strip()
            self._config["sizes"][class_lower] = size
            self._save_config()

    def update_sizes(self, sizes: Dict[str, float]):
        """
        Update multiple sizes at once.

        Args:
            sizes: Dictionary of {class_name: size_in_meters}
        """
        with self._lock:
            if "sizes" not in self._config:
                self._config["sizes"] = {}

            for class_name, size in sizes.items():
                if size > 0:
                    class_lower = class_name.lower().strip()
                    self._config["sizes"][class_lower] = size

            self._save_config()

    def delete_size(self, object_class: str):
        """
        Remove a size configuration.

        Args:
            object_class: Object class name to remove
        """
        with self._lock:
            class_lower = object_class.lower().strip()
            sizes = self._config.get("sizes", {})

            if class_lower in sizes:
                del sizes[class_lower]
                self._save_config()

    def get_reference_distance(self) -> float:
        """Get reference capture distance in meters."""
        with self._lock:
            return self._config.get("reference_capture_distance", 2.0)

    def set_reference_distance(self, distance: float):
        """
        Set reference capture distance.

        Args:
            distance: Distance in meters (must be > 0)
        """
        if distance <= 0:
            raise ValueError("Distance must be positive")

        with self._lock:
            self._config["reference_capture_distance"] = distance
            self._save_config()

    def reload(self):
        """Reload configuration from file."""
        with self._lock:
            self._load_config()

    def get_config_dict(self) -> dict:
        """Get the entire configuration as a dictionary."""
        with self._lock:
            return self._config.copy()


# Global instance
_config_instance: Optional[ObjectSizeConfig] = None
_instance_lock = threading.Lock()


def get_object_size_config() -> ObjectSizeConfig:
    """Get the global ObjectSizeConfig instance (singleton)."""
    global _config_instance

    if _config_instance is None:
        with _instance_lock:
            if _config_instance is None:
                _config_instance = ObjectSizeConfig()

    return _config_instance
