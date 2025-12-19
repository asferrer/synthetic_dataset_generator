"""
Segmentation Service Client
===========================
HTTP client for communicating with the Segmentation microservice.
"""

import os
import logging
import httpx
import base64
import numpy as np
import cv2
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


def decode_region_map_base64(region_map_b64: str) -> Optional[np.ndarray]:
    """Decode region map from base64 PNG string.

    Returns numpy array with region values:
    0=unknown, 1=open_water, 2=seafloor, 3=surface, 4=vegetation, 5=rocky, 6=sandy, 7=murky
    """
    try:
        if not region_map_b64:
            logger.warning("Empty region_map_base64 string provided")
            return None

        png_bytes = base64.b64decode(region_map_b64)
        if len(png_bytes) == 0:
            logger.warning("Decoded region_map bytes are empty")
            return None

        nparr = np.frombuffer(png_bytes, np.uint8)
        region_map = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # Validate decode result
        if region_map is None:
            logger.error("cv2.imdecode returned None - PNG data may be corrupted")
            return None

        if region_map.size == 0:
            logger.error("Decoded region_map has zero size")
            return None

        logger.debug(f"Region map decoded: {region_map.shape}, unique values: {np.unique(region_map)}")
        return region_map

    except Exception as e:
        logger.warning(f"Failed to decode region_map: {e}")
        return None


class SceneRegion(str, Enum):
    """Scene region types matching the Segmentation service"""
    OPEN_WATER = "open_water"
    SEAFLOOR = "seafloor"
    SURFACE = "surface"
    VEGETATION = "vegetation"
    ROCKY = "rocky"
    SANDY = "sandy"
    MURKY = "murky"


@dataclass
class SceneAnalysis:
    """Scene analysis result from segmentation service"""
    dominant_region: SceneRegion
    region_scores: Dict[str, float]
    depth_zones: Dict[str, Tuple[float, float]]
    scene_brightness: float
    water_clarity: str
    color_temperature: str
    region_map: Optional[np.ndarray] = None


@dataclass
class DebugInfo:
    """Debug information from scene analysis"""
    analysis_method: str
    processing_time_ms: float
    sam3_prompts_used: List[str]
    region_confidences: Dict[str, float]
    decision_log: List[str]
    visualization_path: Optional[str] = None


@dataclass
class PlacementDecision:
    """Placement decision with debug information"""
    decision: str  # "accepted", "rejected", "relocated"
    original_position: Tuple[int, int]
    final_position: Optional[Tuple[int, int]]
    score: float
    reason: str
    region_at_position: str
    alternative_positions: List[Tuple[int, int, float]]
    object_class: str = ""  # Object class name
    compatibility_score: float = 0.0  # Alias for score for compatibility


class SegmentationClient:
    """
    HTTP client for the Segmentation microservice.

    Provides methods for scene analysis and object-scene compatibility checking
    via HTTP calls to the centralized Segmentation service.
    """

    def __init__(
        self,
        service_url: Optional[str] = None,
        timeout: int = 30,
        debug: bool = False,
        debug_output_dir: Optional[str] = None,
    ):
        """
        Initialize the segmentation client.

        Args:
            service_url: URL of the segmentation service (default: from env var)
            timeout: Request timeout in seconds
            debug: Enable debug mode
            debug_output_dir: Directory for debug output files
        """
        self.service_url = service_url or os.environ.get(
            "SEGMENTATION_SERVICE_URL", "http://segmentation:8002"
        )
        self.timeout = httpx.Timeout(timeout, connect=10.0)
        self.debug = debug
        self.debug_output_dir = debug_output_dir
        self._client: Optional[httpx.AsyncClient] = None

        logger.info(f"SegmentationClient initialized with URL: {self.service_url}")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self):
        """Close the HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def health_check(self) -> Dict[str, Any]:
        """Check if segmentation service is healthy"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.service_url}/health")
            if response.status_code == 200:
                return response.json()
            return {"status": "unhealthy", "error": f"Status {response.status_code}"}
        except Exception as e:
            logger.error(f"Segmentation service health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def analyze_scene(
        self,
        image: np.ndarray,
        image_path: Optional[str] = None,
    ) -> SceneAnalysis:
        """
        Analyze scene using the Segmentation service.

        Args:
            image: Image as numpy array (BGR format)
            image_path: Path to saved image (if available)

        Returns:
            SceneAnalysis with region information
        """
        try:
            # If no path provided, save image temporarily to shared volume
            if image_path is None:
                import tempfile
                import time
                temp_path = f"/shared/temp/scene_{int(time.time() * 1000)}.jpg"
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                cv2.imwrite(temp_path, image)
                image_path = temp_path

            client = await self._get_client()

            # Call the analyze endpoint
            response = await client.post(
                f"{self.service_url}/analyze",
                json={"image_path": image_path}
            )
            if response.status_code != 200:
                raise RuntimeError(f"Segmentation service error: {response.text}")

            result = response.json()

            if not result.get("success", False):
                raise RuntimeError(result.get("error", "Unknown error"))

            # Convert region string to enum
            try:
                dominant_region = SceneRegion(result["dominant_region"])
            except ValueError:
                dominant_region = SceneRegion.OPEN_WATER

            # Convert depth zones to tuples
            depth_zones = {
                k: tuple(v) for k, v in result.get("depth_zones", {}).items()
            }

            # Decode region_map if present
            region_map = None
            if result.get("region_map_base64"):
                region_map = decode_region_map_base64(result["region_map_base64"])

            return SceneAnalysis(
                dominant_region=dominant_region,
                region_scores=result.get("region_scores", {}),
                depth_zones=depth_zones,
                scene_brightness=result.get("scene_brightness", 0.5),
                water_clarity=result.get("water_clarity", "moderate"),
                color_temperature=result.get("color_temperature", "neutral"),
                region_map=region_map,
            )

        except Exception as e:
            logger.error(f"Scene analysis failed: {e}")
            # Return basic analysis based on image properties
            return self._create_basic_analysis_from_image(image)

    async def analyze_scene_with_debug(
        self,
        image: np.ndarray,
        image_path: Optional[str] = None,
        save_visualization: bool = True,
        image_id: Optional[str] = None,
    ) -> Tuple[SceneAnalysis, DebugInfo]:
        """
        Analyze scene with full debug information.

        Args:
            image: Image as numpy array
            image_path: Path to saved image
            save_visualization: Whether to save debug visualization
            image_id: Identifier for debug files

        Returns:
            Tuple of (SceneAnalysis, DebugInfo)
        """
        try:
            if image_path is None:
                import time
                temp_path = f"/shared/temp/scene_debug_{int(time.time() * 1000)}.jpg"
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                cv2.imwrite(temp_path, image)
                image_path = temp_path

            client = await self._get_client()

            response = await client.post(
                f"{self.service_url}/debug/analyze",
                json={
                    "image_path": image_path,
                    "save_visualization": save_visualization,
                    "image_id": image_id,
                }
            )
            if response.status_code != 200:
                raise RuntimeError(f"Debug analyze failed: {response.text}")

            result = response.json()

            if not result.get("success", False):
                raise RuntimeError(result.get("error", "Unknown error"))

            # Parse analysis
            try:
                dominant_region = SceneRegion(result["dominant_region"])
            except ValueError:
                dominant_region = SceneRegion.OPEN_WATER

            # Decode region_map if present
            region_map = None
            if result.get("region_map_base64"):
                region_map = decode_region_map_base64(result["region_map_base64"])

            analysis = SceneAnalysis(
                dominant_region=dominant_region,
                region_scores=result.get("region_scores", {}),
                depth_zones={},
                scene_brightness=result.get("scene_brightness", 0.5),
                water_clarity=result.get("water_clarity", "moderate"),
                color_temperature=result.get("color_temperature", "neutral"),
                region_map=region_map,
            )

            debug_info = DebugInfo(
                analysis_method=result.get("analysis_method", "unknown"),
                processing_time_ms=result.get("processing_time_ms", 0),
                sam3_prompts_used=result.get("sam3_prompts_used", []),
                region_confidences=result.get("region_confidences", {}),
                decision_log=result.get("decision_log", []),
                visualization_path=result.get("visualization_path"),
            )

            return analysis, debug_info

        except Exception as e:
            logger.error(f"Debug scene analysis failed: {e}")
            # Return defaults on error
            analysis = SceneAnalysis(
                dominant_region=SceneRegion.OPEN_WATER,
                region_scores={"open_water": 1.0},
                depth_zones={},
                scene_brightness=0.5,
                water_clarity="moderate",
                color_temperature="neutral",
            )
            debug_info = DebugInfo(
                analysis_method="error",
                processing_time_ms=0,
                sam3_prompts_used=[],
                region_confidences={},
                decision_log=[f"Error: {str(e)}"],
            )
            return analysis, debug_info

    async def check_compatibility(
        self,
        object_class: str,
        position: Tuple[int, int],
        image: np.ndarray,
        image_path: Optional[str] = None,
    ) -> Tuple[float, str]:
        """
        Check if object placement is compatible with scene.

        Args:
            object_class: Class name of the object
            position: (x, y) position for placement
            image: Background image
            image_path: Path to saved image

        Returns:
            Tuple of (compatibility_score, reason)
        """
        try:
            if image_path is None:
                import time
                temp_path = f"/shared/temp/compat_{int(time.time() * 1000)}.jpg"
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                cv2.imwrite(temp_path, image)
                image_path = temp_path

            client = await self._get_client()

            response = await client.post(
                f"{self.service_url}/check-compatibility",
                json={
                    "image_path": image_path,
                    "object_class": object_class,
                    "position_x": position[0],
                    "position_y": position[1],
                }
            )
            if response.status_code != 200:
                return 0.6, "Service error - using default"

            result = response.json()

            if not result.get("success", False):
                return 0.6, result.get("error", "Unknown error")

            return result.get("score", 0.6), result.get("reason", "No reason provided")

        except Exception as e:
            logger.error(f"Compatibility check failed: {e}")
            return 0.6, f"Error: {str(e)}"

    async def check_compatibility_with_debug(
        self,
        object_class: str,
        position: Tuple[int, int],
        image: np.ndarray,
        image_path: Optional[str] = None,
    ) -> Tuple[float, str, PlacementDecision]:
        """
        Check compatibility with full debug information.

        Args:
            object_class: Class name of the object
            position: (x, y) position for placement
            image: Background image
            image_path: Path to saved image

        Returns:
            Tuple of (score, reason, PlacementDecision)
        """
        try:
            if image_path is None:
                import time
                temp_path = f"/shared/temp/compat_debug_{int(time.time() * 1000)}.jpg"
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                cv2.imwrite(temp_path, image)
                image_path = temp_path

            client = await self._get_client()

            response = await client.post(
                f"{self.service_url}/debug/compatibility",
                json={
                    "image_path": image_path,
                    "object_class": object_class,
                    "position_x": position[0],
                    "position_y": position[1],
                }
            )
            if response.status_code != 200:
                raise RuntimeError(f"Debug compatibility failed: {response.text}")

            result = response.json()

            score = result.get("score", 0.6)
            reason = result.get("reason", "No reason")

            # Parse alternatives
            alternatives = [
                (int(a[0]), int(a[1]), float(a[2]))
                for a in result.get("alternatives", [])
            ]

            decision = PlacementDecision(
                decision=result.get("decision", "accepted"),
                original_position=position,
                final_position=position if result.get("is_compatible") else None,
                score=score,
                reason=reason,
                region_at_position=result.get("region_at_position", "unknown"),
                alternative_positions=alternatives,
                object_class=object_class,
                compatibility_score=score,
            )

            return score, reason, decision

        except Exception as e:
            logger.error(f"Debug compatibility check failed: {e}")
            decision = PlacementDecision(
                decision="error",
                original_position=position,
                final_position=position,
                score=0.6,
                reason=f"Error: {str(e)}",
                region_at_position="unknown",
                alternative_positions=[],
                object_class=object_class,
                compatibility_score=0.6,
            )
            return 0.6, str(e), decision

    async def suggest_placement(
        self,
        object_class: str,
        object_size: Tuple[int, int],
        image: np.ndarray,
        image_path: Optional[str] = None,
        existing_positions: Optional[List[Tuple[int, int]]] = None,
        min_distance: int = 50,
    ) -> Optional[Tuple[int, int]]:
        """
        Get placement suggestion from segmentation service.

        Args:
            object_class: Class name of object
            object_size: (width, height) of object
            image: Background image
            image_path: Path to saved image
            existing_positions: Already placed object positions
            min_distance: Minimum distance between objects

        Returns:
            Suggested (x, y) position or None
        """
        try:
            if image_path is None:
                import time
                temp_path = f"/shared/temp/suggest_{int(time.time() * 1000)}.jpg"
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                cv2.imwrite(temp_path, image)
                image_path = temp_path

            client = await self._get_client()

            response = await client.post(
                f"{self.service_url}/suggest-placement",
                json={
                    "image_path": image_path,
                    "object_class": object_class,
                    "object_width": object_size[0],
                    "object_height": object_size[1],
                    "existing_positions": existing_positions or [],
                    "min_distance": min_distance,
                }
            )
            if response.status_code != 200:
                return None

            result = response.json()

            if not result.get("success", False):
                return None

            return (result["position_x"], result["position_y"])

        except Exception as e:
            logger.error(f"Placement suggestion failed: {e}")
            return None

    def get_best_placement_region(
        self,
        object_class: str,
        scene_analysis: SceneAnalysis,
    ) -> Optional[SceneRegion]:
        """
        Get best placement region for an object class.
        Local computation based on compatibility rules.

        Args:
            object_class: Object class name
            scene_analysis: Previous scene analysis result

        Returns:
            Best region for placement
        """
        # Compatibility rules (same as in scene_analyzer.py)
        SCENE_COMPATIBILITY = {
            "fish": {
                SceneRegion.OPEN_WATER: 1.0,
                SceneRegion.SEAFLOOR: 0.4,
                SceneRegion.SURFACE: 0.6,
                SceneRegion.VEGETATION: 0.8,
            },
            "can": {
                SceneRegion.SEAFLOOR: 1.0,
                SceneRegion.SANDY: 0.9,
                SceneRegion.ROCKY: 0.7,
                SceneRegion.OPEN_WATER: 0.2,
            },
            "plastic": {
                SceneRegion.SURFACE: 0.9,
                SceneRegion.OPEN_WATER: 0.8,
                SceneRegion.SEAFLOOR: 0.5,
            },
            "bottle": {
                SceneRegion.SURFACE: 0.8,
                SceneRegion.OPEN_WATER: 0.7,
                SceneRegion.SEAFLOOR: 0.6,
            },
            "debris": {
                SceneRegion.SEAFLOOR: 0.9,
                SceneRegion.SANDY: 0.85,
                SceneRegion.ROCKY: 0.7,
            },
        }

        # Find matching category
        obj_lower = object_class.lower()
        compatibility = None

        for key, compat in SCENE_COMPATIBILITY.items():
            if key in obj_lower:
                compatibility = compat
                break

        if compatibility is None:
            # Default: open water is fine for unknown objects
            return SceneRegion.OPEN_WATER

        # Find best region from available ones
        best_region = None
        best_score = -1

        for region_str, score in scene_analysis.region_scores.items():
            try:
                region = SceneRegion(region_str)
                compat_score = compatibility.get(region, 0.3)
                weighted_score = score * compat_score

                if weighted_score > best_score:
                    best_score = weighted_score
                    best_region = region
            except ValueError:
                continue

        return best_region or SceneRegion.OPEN_WATER

    def _create_basic_analysis_from_image(self, image: np.ndarray) -> SceneAnalysis:
        """
        Create basic scene analysis from image properties when service fails.

        This is a simple fallback that analyzes basic color properties to provide
        better-than-nothing defaults instead of assuming OPEN_WATER.

        Args:
            image: BGR numpy array

        Returns:
            SceneAnalysis with basic estimates
        """
        try:
            h, w = image.shape[:2]

            # Analyze image color distribution
            b, g, r = cv2.split(image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Calculate basic metrics
            brightness = gray.mean() / 255.0
            blue_ratio = b.mean() / max(r.mean(), 1)

            # Analyze vertical gradient (bottom darker = seafloor likely)
            top_half = gray[:h//2, :].mean()
            bottom_half = gray[h//2:, :].mean()

            # Estimate dominant region
            region_scores = {
                "open_water": 0.3,
                "seafloor": 0.2,
                "surface": 0.2,
                "vegetation": 0.1,
                "murky": 0.1,
                "sandy": 0.1,
            }

            # Adjust scores based on image properties
            if blue_ratio > 1.2:
                region_scores["open_water"] += 0.3
            if bottom_half < top_half * 0.8:
                region_scores["seafloor"] += 0.3
                region_scores["open_water"] -= 0.1
            if brightness < 0.3:
                region_scores["murky"] += 0.2

            # Determine clarity
            contrast = gray.std() / 127.0
            if contrast > 0.3:
                water_clarity = "clear"
            elif contrast > 0.15:
                water_clarity = "moderate"
            else:
                water_clarity = "murky"

            # Color temperature
            if blue_ratio > 1.3:
                color_temperature = "cool"
            elif blue_ratio < 0.8:
                color_temperature = "warm"
            else:
                color_temperature = "neutral"

            # Normalize and find dominant
            total = sum(region_scores.values())
            region_scores = {k: v/total for k, v in region_scores.items()}
            dominant = max(region_scores.items(), key=lambda x: x[1])[0]

            try:
                dominant_region = SceneRegion(dominant)
            except ValueError:
                dominant_region = SceneRegion.OPEN_WATER

            logger.info(f"Basic analysis fallback: dominant={dominant}, brightness={brightness:.2f}")

            return SceneAnalysis(
                dominant_region=dominant_region,
                region_scores=region_scores,
                depth_zones={"mid_water": (0.3, 0.7)},
                scene_brightness=brightness,
                water_clarity=water_clarity,
                color_temperature=color_temperature,
            )

        except Exception as e:
            logger.error(f"Basic analysis fallback also failed: {e}")
            # Ultimate fallback
            return SceneAnalysis(
                dominant_region=SceneRegion.OPEN_WATER,
                region_scores={"open_water": 0.5, "seafloor": 0.3, "surface": 0.2},
                depth_zones={"mid_water": (0.3, 0.7)},
                scene_brightness=0.5,
                water_clarity="moderate",
                color_temperature="neutral",
            )
