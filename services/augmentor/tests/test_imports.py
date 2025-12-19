"""
Test that all modules import correctly.

These tests verify that:
1. All dependencies are installed correctly
2. No syntax errors in application code
3. Module structure is valid
"""


def test_main_app_imports():
    """Verify main app module imports without errors."""
    from app.main import app
    assert app is not None


def test_schemas_import():
    """Verify all schemas import correctly."""
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
        ComposeBatchRequest,
        ValidateRequest,
        LightingRequest,
        ComposeResponse,
        ComposeBatchResponse,
        ValidateResponse,
        LightingResponse,
        HealthResponse,
        InfoResponse,
    )
    # Verify enums have values
    assert len(EffectType) > 0
    assert len(LightType) > 0
    assert len(DepthZone) > 0
    assert len(WaterClarity) > 0


def test_segmentation_client_import():
    """Verify segmentation client imports correctly."""
    from app.segmentation_client import (
        SegmentationClient,
        SceneRegion,
        SceneAnalysis,
        PlacementDecision,
        decode_region_map_base64,
    )
    assert SegmentationClient is not None
    assert len(SceneRegion) > 0


def test_composer_import():
    """Verify composer module imports correctly."""
    from app.composer import ImageComposer
    assert ImageComposer is not None
