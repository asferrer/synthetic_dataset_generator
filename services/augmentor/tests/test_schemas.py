"""
Test Pydantic schema validation.

These tests verify that:
1. Schemas accept valid data
2. Schemas have correct default values
3. Validation rules work properly
"""

import pytest
from app.models.schemas import (
    EffectType,
    LightType,
    DepthZone,
    WaterClarity,
    ObjectPlacement,
    AnnotationBox,
    EffectsConfig,
    LightSourceInfo,
    QualityScoreInfo,
    PhysicsViolationInfo,
    LightingInfo,
    ComposeRequest,
    ValidateRequest,
    HealthResponse,
)


class TestEnums:
    """Test enum values are correct."""

    def test_effect_type_values(self):
        """Verify EffectType enum has expected values."""
        assert EffectType.COLOR_CORRECTION.value == "color_correction"
        assert EffectType.BLUR_MATCHING.value == "blur_matching"
        assert EffectType.SHADOWS.value == "shadows"
        assert EffectType.CAUSTICS.value == "caustics"

    def test_light_type_values(self):
        """Verify LightType enum has expected values."""
        assert LightType.DIRECTIONAL.value == "directional"
        assert LightType.POINT.value == "point"
        assert LightType.AREA.value == "area"

    def test_depth_zone_values(self):
        """Verify DepthZone enum has expected values."""
        assert DepthZone.NEAR.value == "near"
        assert DepthZone.MID.value == "mid"
        assert DepthZone.FAR.value == "far"

    def test_water_clarity_values(self):
        """Verify WaterClarity enum has expected values."""
        assert WaterClarity.CLEAR.value == "clear"
        assert WaterClarity.MURKY.value == "murky"
        assert WaterClarity.VERY_MURKY.value == "very_murky"


class TestObjectPlacement:
    """Test ObjectPlacement schema."""

    def test_minimal_creation(self):
        """Create with only required fields."""
        obj = ObjectPlacement(image_path="/test.png", class_name="fish")
        assert obj.image_path == "/test.png"
        assert obj.class_name == "fish"
        assert obj.position is None
        assert obj.scale is None
        assert obj.rotation is None
        assert obj.material == "plastic"

    def test_full_creation(self):
        """Create with all fields."""
        obj = ObjectPlacement(
            image_path="/test.png",
            class_name="can",
            position=(100, 200),
            scale=1.5,
            rotation=45.0,
            material="metal",
        )
        assert obj.position == (100, 200)
        assert obj.scale == 1.5
        assert obj.rotation == 45.0
        assert obj.material == "metal"

    def test_scale_bounds(self):
        """Verify scale validation bounds."""
        # Valid scale
        obj = ObjectPlacement(image_path="/t.png", class_name="x", scale=0.1)
        assert obj.scale == 0.1

        obj = ObjectPlacement(image_path="/t.png", class_name="x", scale=5.0)
        assert obj.scale == 5.0

        # Invalid scale
        with pytest.raises(Exception):
            ObjectPlacement(image_path="/t.png", class_name="x", scale=0.05)

        with pytest.raises(Exception):
            ObjectPlacement(image_path="/t.png", class_name="x", scale=6.0)


class TestAnnotationBox:
    """Test AnnotationBox schema."""

    def test_creation(self):
        """Create annotation box."""
        box = AnnotationBox(
            x=10, y=20, width=100, height=50, class_name="fish"
        )
        assert box.x == 10
        assert box.y == 20
        assert box.width == 100
        assert box.height == 50
        assert box.class_name == "fish"
        assert box.confidence == 1.0  # default
        assert box.area == 0  # default


class TestEffectsConfig:
    """Test EffectsConfig schema."""

    def test_defaults(self):
        """Verify default values are correct (BUG #11 revised defaults)."""
        config = EffectsConfig()
        assert config.color_intensity == 0.12  # Revised from 0.15 (BUG #11)
        assert config.blur_strength == 0.5
        assert config.underwater_intensity == 0.15
        assert config.caustics_intensity == 0.10  # Revised from 0.15 (BUG #11)
        assert config.shadow_opacity == 0.10  # Revised from 0.12 (BUG #11)
        assert config.shadow_blur == 25  # Revised from 31 (BUG #11)
        assert config.edge_feather == 4
        assert config.water_clarity == WaterClarity.CLEAR
        # New fields from bug fixes
        assert config.validate_identity is False  # Disabled by default (too strict for underwater)
        assert config.use_binary_alpha is True
        assert config.caustics_deterministic is True
        assert config.recalculate_bbox_after_global is True

    def test_custom_values(self):
        """Create with custom values."""
        config = EffectsConfig(
            color_intensity=0.5,
            blur_strength=1.0,
            shadow_opacity=0.3,
        )
        assert config.color_intensity == 0.5
        assert config.blur_strength == 1.0
        assert config.shadow_opacity == 0.3


class TestLightSourceInfo:
    """Test LightSourceInfo schema."""

    def test_creation(self):
        """Create light source info."""
        light = LightSourceInfo(
            light_type=LightType.DIRECTIONAL,
            position=(45.0, 60.0, 100.0),
            intensity=0.8,
            color=(255, 255, 200),
        )
        assert light.light_type == LightType.DIRECTIONAL
        assert light.position == (45.0, 60.0, 100.0)
        assert light.intensity == 0.8
        assert light.color == (255, 255, 200)


class TestQualityScoreInfo:
    """Test QualityScoreInfo schema."""

    def test_creation(self):
        """Create quality score info."""
        score = QualityScoreInfo(
            perceptual_quality=0.85,
            anomaly_score=0.9,
            composition_score=0.8,
            overall_score=0.85,
            overall_pass=True,
        )
        assert score.perceptual_quality == 0.85
        assert score.overall_pass is True


class TestComposeRequest:
    """Test ComposeRequest schema."""

    def test_minimal_creation(self):
        """Create with minimal fields."""
        req = ComposeRequest(
            background_path="/bg.jpg",
            objects=[ObjectPlacement(image_path="/obj.png", class_name="fish")],
            output_path="/out.jpg",
        )
        assert req.background_path == "/bg.jpg"
        assert len(req.objects) == 1
        assert req.validate_quality is False
        assert req.validate_physics is False


class TestHealthResponse:
    """Test HealthResponse schema."""

    def test_creation(self):
        """Create health response."""
        health = HealthResponse(
            status="healthy",
            gpu_available=True,
            gpu_name="RTX 4090",
            validators_loaded=True,
        )
        assert health.status == "healthy"
        assert health.gpu_available is True
