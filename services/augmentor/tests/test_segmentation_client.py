"""
Test segmentation client utility functions.

These tests verify:
1. Base64 decoding handles edge cases
2. Enum values are correct
3. Pure functions work without network
"""

import pytest
from app.segmentation_client import (
    decode_region_map_base64,
    SceneRegion,
    SceneAnalysis,
    PlacementDecision,
    SegmentationClient,
)


class TestSceneRegion:
    """Test SceneRegion enum."""

    def test_enum_values(self):
        """Verify all enum values are correct."""
        assert SceneRegion.OPEN_WATER.value == "open_water"
        assert SceneRegion.SEAFLOOR.value == "seafloor"
        assert SceneRegion.SURFACE.value == "surface"
        assert SceneRegion.VEGETATION.value == "vegetation"
        assert SceneRegion.ROCKY.value == "rocky"
        assert SceneRegion.SANDY.value == "sandy"
        assert SceneRegion.MURKY.value == "murky"

    def test_enum_count(self):
        """Verify all expected regions exist."""
        assert len(SceneRegion) == 7


class TestDecodeRegionMap:
    """Test decode_region_map_base64 function."""

    def test_empty_string(self):
        """Empty string should return None."""
        result = decode_region_map_base64("")
        assert result is None

    def test_none_input(self):
        """None input should return None (with error handling)."""
        # The function checks for empty string first, but handles None gracefully
        try:
            result = decode_region_map_base64(None)
            assert result is None
        except TypeError:
            # This is also acceptable behavior
            pass

    def test_invalid_base64(self):
        """Invalid base64 should return None."""
        result = decode_region_map_base64("not-valid-base64!!!")
        assert result is None

    def test_empty_decoded_bytes(self):
        """Base64 that decodes to empty should return None."""
        import base64
        empty_b64 = base64.b64encode(b"").decode()
        result = decode_region_map_base64(empty_b64)
        assert result is None


class TestSceneAnalysis:
    """Test SceneAnalysis dataclass."""

    def test_creation(self):
        """Create SceneAnalysis instance."""
        analysis = SceneAnalysis(
            dominant_region=SceneRegion.OPEN_WATER,
            region_scores={"open_water": 0.8, "seafloor": 0.2},
            depth_zones={"mid_water": (0.3, 0.7)},
            scene_brightness=0.6,
            water_clarity="clear",
            color_temperature="neutral",
        )
        assert analysis.dominant_region == SceneRegion.OPEN_WATER
        assert analysis.region_scores["open_water"] == 0.8
        assert analysis.scene_brightness == 0.6
        assert analysis.region_map is None  # default


class TestPlacementDecision:
    """Test PlacementDecision dataclass."""

    def test_creation(self):
        """Create PlacementDecision instance."""
        decision = PlacementDecision(
            decision="accepted",
            original_position=(100, 200),
            final_position=(100, 200),
            score=0.9,
            reason="Compatible with open water",
            region_at_position="open_water",
            alternative_positions=[(150, 180, 0.85), (200, 220, 0.8)],
            object_class="fish",
            compatibility_score=0.9,
        )
        assert decision.decision == "accepted"
        assert decision.original_position == (100, 200)
        assert decision.score == 0.9
        assert len(decision.alternative_positions) == 2


class TestSegmentationClientInit:
    """Test SegmentationClient initialization."""

    def test_default_url(self):
        """Client should use default URL from env or hardcoded."""
        client = SegmentationClient()
        assert "segmentation" in client.service_url or "8002" in client.service_url

    def test_custom_url(self):
        """Client should accept custom URL."""
        client = SegmentationClient(service_url="http://custom:9000")
        assert client.service_url == "http://custom:9000"

    def test_debug_mode(self):
        """Client should accept debug settings."""
        client = SegmentationClient(debug=True, debug_output_dir="/tmp/debug")
        assert client.debug is True
        assert client.debug_output_dir == "/tmp/debug"


class TestGetBestPlacementRegion:
    """Test get_best_placement_region method."""

    def test_fish_prefers_open_water(self):
        """Fish should prefer open water."""
        client = SegmentationClient()
        analysis = SceneAnalysis(
            dominant_region=SceneRegion.OPEN_WATER,
            region_scores={"open_water": 0.5, "seafloor": 0.3, "surface": 0.2},
            depth_zones={},
            scene_brightness=0.5,
            water_clarity="clear",
            color_temperature="neutral",
        )
        result = client.get_best_placement_region("fish", analysis)
        assert result == SceneRegion.OPEN_WATER

    def test_can_prefers_seafloor(self):
        """Cans should prefer seafloor."""
        client = SegmentationClient()
        analysis = SceneAnalysis(
            dominant_region=SceneRegion.SEAFLOOR,
            region_scores={"open_water": 0.3, "seafloor": 0.5, "surface": 0.2},
            depth_zones={},
            scene_brightness=0.5,
            water_clarity="clear",
            color_temperature="neutral",
        )
        result = client.get_best_placement_region("metal_can", analysis)
        assert result == SceneRegion.SEAFLOOR

    def test_unknown_class_defaults_to_open_water(self):
        """Unknown objects should default to open water."""
        client = SegmentationClient()
        analysis = SceneAnalysis(
            dominant_region=SceneRegion.SEAFLOOR,
            region_scores={"seafloor": 0.8},
            depth_zones={},
            scene_brightness=0.5,
            water_clarity="clear",
            color_temperature="neutral",
        )
        result = client.get_best_placement_region("unknown_object_xyz", analysis)
        assert result == SceneRegion.OPEN_WATER
