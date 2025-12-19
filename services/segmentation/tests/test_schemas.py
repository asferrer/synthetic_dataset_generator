"""
Test Pydantic schema validation.

These tests verify that:
1. Schemas accept valid data
2. Schemas have correct default values
3. Validation rules work properly
"""

import pytest
from app.models.schemas import (
    SceneRegionType,
    AnalyzeSceneRequest,
    AnalyzeSceneResponse,
    CompatibilityCheckRequest,
    CompatibilityCheckResponse,
    SuggestPlacementRequest,
    SuggestPlacementResponse,
    SegmentTextRequest,
    SegmentTextResponse,
    HealthResponse,
    DebugAnalyzeRequest,
    DebugAnalyzeResponse,
    PlacementDecisionInfo,
)


class TestSceneRegionType:
    """Test SceneRegionType enum."""

    def test_enum_values(self):
        """Verify all enum values are correct."""
        assert SceneRegionType.OPEN_WATER.value == "open_water"
        assert SceneRegionType.SEAFLOOR.value == "seafloor"
        assert SceneRegionType.SURFACE.value == "surface"
        assert SceneRegionType.VEGETATION.value == "vegetation"
        assert SceneRegionType.ROCKY.value == "rocky"
        assert SceneRegionType.SANDY.value == "sandy"
        assert SceneRegionType.MURKY.value == "murky"
        assert SceneRegionType.UNKNOWN.value == "unknown"

    def test_enum_count(self):
        """Verify all expected regions exist."""
        assert len(SceneRegionType) == 8


class TestAnalyzeSceneRequest:
    """Test AnalyzeSceneRequest schema."""

    def test_minimal_creation(self):
        """Create with only required fields."""
        req = AnalyzeSceneRequest(image_path="/test/image.jpg")
        assert req.image_path == "/test/image.jpg"
        assert req.use_sam3 is False  # default

    def test_with_sam3(self):
        """Create with SAM3 enabled."""
        req = AnalyzeSceneRequest(image_path="/test/image.jpg", use_sam3=True)
        assert req.use_sam3 is True


class TestAnalyzeSceneResponse:
    """Test AnalyzeSceneResponse schema."""

    def test_success_response(self):
        """Create successful response."""
        resp = AnalyzeSceneResponse(
            success=True,
            dominant_region="open_water",
            region_scores={"open_water": 0.8, "seafloor": 0.2},
            depth_zones={"mid_water": [0.3, 0.7]},
            scene_brightness=0.6,
            water_clarity="clear",
            color_temperature="neutral",
            processing_time_ms=150.5,
        )
        assert resp.success is True
        assert resp.dominant_region == "open_water"
        assert resp.region_scores["open_water"] == 0.8
        assert resp.error is None

    def test_error_response(self):
        """Create error response."""
        resp = AnalyzeSceneResponse(
            success=False,
            dominant_region="unknown",
            region_scores={},
            depth_zones={},
            scene_brightness=0.0,
            water_clarity="unknown",
            color_temperature="unknown",
            processing_time_ms=0,
            error="Failed to analyze scene",
        )
        assert resp.success is False
        assert resp.error == "Failed to analyze scene"


class TestCompatibilityCheckRequest:
    """Test CompatibilityCheckRequest schema."""

    def test_creation(self):
        """Create compatibility check request."""
        req = CompatibilityCheckRequest(
            image_path="/bg.jpg",
            object_class="fish",
            position_x=100,
            position_y=200,
        )
        assert req.image_path == "/bg.jpg"
        assert req.object_class == "fish"
        assert req.position_x == 100
        assert req.position_y == 200


class TestCompatibilityCheckResponse:
    """Test CompatibilityCheckResponse schema."""

    def test_compatible_response(self):
        """Create compatible response."""
        resp = CompatibilityCheckResponse(
            success=True,
            is_compatible=True,
            score=0.9,
            reason="Fish compatible with open water",
        )
        assert resp.is_compatible is True
        assert resp.score == 0.9

    def test_incompatible_response(self):
        """Create incompatible response."""
        resp = CompatibilityCheckResponse(
            success=True,
            is_compatible=False,
            score=0.2,
            reason="Can not compatible with surface",
            suggested_region="seafloor",
        )
        assert resp.is_compatible is False
        assert resp.suggested_region == "seafloor"


class TestSuggestPlacementRequest:
    """Test SuggestPlacementRequest schema."""

    def test_minimal_creation(self):
        """Create with minimal fields."""
        req = SuggestPlacementRequest(
            image_path="/bg.jpg",
            object_class="fish",
            object_width=100,
            object_height=50,
        )
        assert req.object_width == 100
        assert req.existing_positions == []  # default
        assert req.min_distance == 50  # default

    def test_with_existing_positions(self):
        """Create with existing positions."""
        req = SuggestPlacementRequest(
            image_path="/bg.jpg",
            object_class="fish",
            object_width=100,
            object_height=50,
            existing_positions=[[50, 50], [200, 200]],
            min_distance=80,
        )
        assert len(req.existing_positions) == 2
        assert req.min_distance == 80


class TestSegmentTextRequest:
    """Test SegmentTextRequest schema."""

    def test_minimal_creation(self):
        """Create with minimal fields."""
        req = SegmentTextRequest(
            image_path="/image.jpg",
            text_prompt="water surface",
        )
        assert req.text_prompt == "water surface"
        assert req.threshold == 0.5  # default

    def test_with_threshold(self):
        """Create with custom threshold."""
        req = SegmentTextRequest(
            image_path="/image.jpg",
            text_prompt="seafloor",
            threshold=0.7,
        )
        assert req.threshold == 0.7

    def test_threshold_bounds(self):
        """Verify threshold bounds."""
        # Valid thresholds
        req = SegmentTextRequest(image_path="/i.jpg", text_prompt="x", threshold=0.0)
        assert req.threshold == 0.0

        req = SegmentTextRequest(image_path="/i.jpg", text_prompt="x", threshold=1.0)
        assert req.threshold == 1.0

        # Invalid thresholds
        with pytest.raises(Exception):
            SegmentTextRequest(image_path="/i.jpg", text_prompt="x", threshold=-0.1)

        with pytest.raises(Exception):
            SegmentTextRequest(image_path="/i.jpg", text_prompt="x", threshold=1.1)


class TestHealthResponse:
    """Test HealthResponse schema."""

    def test_healthy_response(self):
        """Create healthy response."""
        health = HealthResponse(
            status="healthy",
            sam3_available=True,
            gpu_available=True,
            model_loaded=True,
        )
        assert health.status == "healthy"
        assert health.sam3_available is True
        assert health.version == "1.0.0"  # default

    def test_unhealthy_response(self):
        """Create unhealthy response."""
        health = HealthResponse(
            status="unhealthy",
            sam3_available=False,
            gpu_available=False,
            model_loaded=False,
        )
        assert health.status == "unhealthy"


class TestDebugAnalyzeRequest:
    """Test DebugAnalyzeRequest schema."""

    def test_minimal_creation(self):
        """Create with minimal fields."""
        req = DebugAnalyzeRequest(image_path="/image.jpg")
        assert req.image_path == "/image.jpg"
        assert req.save_visualization is True  # default

    def test_with_options(self):
        """Create with all options."""
        req = DebugAnalyzeRequest(
            image_path="/image.jpg",
            image_id="img_001",
            save_visualization=False,
        )
        assert req.image_id == "img_001"
        assert req.save_visualization is False


class TestPlacementDecisionInfo:
    """Test PlacementDecisionInfo schema."""

    def test_creation(self):
        """Create placement decision info."""
        info = PlacementDecisionInfo(
            object_class="fish",
            requested_position=[100, 200],
            region_at_position="open_water",
            compatibility_score=0.9,
            reason="Compatible with region",
            decision="accepted",
            alternatives=[],
        )
        assert info.object_class == "fish"
        assert info.decision == "accepted"
        assert info.compatibility_score == 0.9
